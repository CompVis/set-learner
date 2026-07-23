"""Generate query-conditioned images from four visual concept examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
import webdataset as wds
from PIL import Image, ImageDraw

from vicis.data import (
    DIRECT_SUBCLASSES,
    HIERARCHY_ATTR_TO_CLASSES,
    SYN_NAME_TO_ID,
    expected_hierarchy_level,
)

HUB_REPOSITORY = "CompVis/set-learner"


def add_generation_args(parser):
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=50)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)

    images = modes.add_parser("images", help="Generate from explicit context and query image paths")
    images.add_argument(
        "--context", nargs=4, type=Path, required=True, metavar=("IMAGE1", "IMAGE2", "IMAGE3", "IMAGE4")
    )
    images.add_argument("--query", type=Path, required=True)
    add_generation_args(images)

    imagenet = modes.add_parser("imagenet", help="Select a hierarchy-valid example from ImageNet shards")
    imagenet.add_argument("--data-dir", type=Path, required=True)
    imagenet.add_argument("--concept", required=True, help="Shared extended_v1 concept, e.g. vehicle.n.01")
    imagenet.add_argument("--query-class", required=True, help="ImageNet leaf class for the query")
    imagenet.add_argument("--split", choices=("train", "val"), default="val")
    add_generation_args(imagenet)
    return parser.parse_args()


def image_transform():
    return T.Compose([T.ToImage(), T.Resize(256), T.CenterCrop(256), T.ToDtype(torch.float32, scale=True)])


def load_image(path):
    return image_transform()(Image.open(path).convert("RGB")) * 2 - 1


def to_pil(tensor):
    tensor = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).byte()
    return T.ToPILImage()(tensor)


def labeled_grid(images, labels, concept=None, expected_level=None):
    width, height, title, caption = 256, 256, 48, 24
    canvas = Image.new("RGB", (width * len(images), height + title + caption), "white")
    draw = ImageDraw.Draw(canvas)
    if concept is None:
        draw.text((6, 14), "User-provided context set and query", fill="black")
    else:
        draw.text((6, 5), f"Shared context concept: {concept}", fill="black")
        draw.text((6, 22), f"Expected generated hierarchy level: {expected_level}", fill="black")
    for index, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(image.resize((width, height)), (index * width, title + caption))
        draw.text((index * width + 4, title + 4), label, fill="black")
    return canvas


def select_imagenet_inputs(data_dir, split, concept, query_class, output_dir):
    """Select four distinct direct children plus the requested query leaf."""
    expected_level = expected_hierarchy_level(concept, query_class)
    query_id = SYN_NAME_TO_ID.get(query_class)
    if query_id is None:
        raise ValueError(f"Query class {query_class} is not an ImageNet leaf class")

    context_children = [child for child in DIRECT_SUBCLASSES[concept] if child != expected_level][:4]
    if len(context_children) < 4:
        raise ValueError(f"Concept {concept} does not have four context children besides {expected_level}")
    context_leaves = []
    for child in context_children:
        candidates = HIERARCHY_ATTR_TO_CLASSES.get(child, [child])
        leaf = next((candidate for candidate in candidates if candidate in SYN_NAME_TO_ID), None)
        if leaf is None:
            raise ValueError(f"No ImageNet leaf found below {child}")
        context_leaves.append(leaf)

    requested = context_leaves + [query_class]
    requested_ids = {SYN_NAME_TO_ID[name]: name for name in requested}
    shards = sorted(str(path) for path in data_dir.glob(f"*{split}*.tar"))
    if not shards:
        raise FileNotFoundError(f"No *{split}*.tar shards found in {data_dir}")
    found = {}
    for sample in wds.WebDataset(shards, shardshuffle=False).decode("pil"):
        class_id = int(sample["cls"])
        if class_id in requested_ids and class_id not in found:
            found[class_id] = sample.get("jpg", sample.get("jpeg")).convert("RGB")
        if len(found) == len(requested_ids):
            break
    if missing := requested_ids.keys() - found.keys():
        raise RuntimeError(f"Could not find ImageNet class ids: {sorted(missing)}")

    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    resize = T.Compose([T.Resize(256), T.CenterCrop(256)])
    for index, name in enumerate(requested):
        role = f"context_{index + 1}" if index < 4 else "query"
        path = inputs_dir / f"{role}_{name.replace('.', '_')}.png"
        resize(found[SYN_NAME_TO_ID[name]]).save(path)
        paths.append(path)
    labels = [
        f"ctx {index + 1}: {leaf.split('.')[0].replace('_', ' ')} [{child}]"
        for index, (child, leaf) in enumerate(zip(context_children, context_leaves))
    ]
    labels.append(f"query: {query_class.split('.')[0].replace('_', ' ')} [{expected_level}]")
    return paths[:4], paths[4], labels, expected_level


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("VICIS inference requires a CUDA GPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "images":
        context_paths, query_path = args.context, args.query
        concept = expected_level = None
        input_labels = [path.stem.replace("_", " ") for path in context_paths]
        input_labels.append(query_path.stem.replace("_", " "))
    else:
        context_paths, query_path, input_labels, expected_level = select_imagenet_inputs(
            args.data_dir, args.split, args.concept, args.query_class, args.output_dir
        )
        concept = args.concept

    model = torch.hub.load(HUB_REPOSITORY, "vicis", pretrained=True, device="cuda")
    context = torch.stack([load_image(path) for path in context_paths], dim=1)[None].cuda().bfloat16()
    query = load_image(query_path)[None].cuda().bfloat16()
    samples, metrics = model.generate(
        context,
        query,
        seed=args.seed,
        num_samples=args.num_samples,
        cfg_scale=args.cfg_scale,
        sample_steps=args.sample_steps,
    )
    outputs = []
    for index, sample in enumerate(samples):
        output = to_pil(sample)
        output.save(args.output_dir / f"sample_{index:02}.png")
        outputs.append(output)

    inputs = [Image.open(path).convert("RGB") for path in context_paths] + [Image.open(query_path).convert("RGB")]
    sample_labels = [
        f"sample {index + 1}" if expected_level is None else f"sample {index + 1}: expected {expected_level}"
        for index in range(len(outputs))
    ]
    labeled_grid(inputs + outputs, input_labels + sample_labels, concept, expected_level).save(
        args.output_dir / "grid.png"
    )
    torch.save({name: value.float().cpu() for name, value in metrics.items()}, args.output_dir / "conditioning.pt")
    print(f"Saved {len(outputs)} samples and grid.png to {args.output_dir}")


if __name__ == "__main__":
    main()
