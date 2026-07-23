"""Run the VICIS ImageNet hierarchy evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from tqdm.auto import tqdm

from vicis.data import (
    DIRECT_SUBCLASSES,
    HIERARCHY_ATTR_TO_CLASSES,
    ID_TO_SYN_NAME,
    SUPER_CLASS_DIR,
    SYN_NAME_TO_ID,
)
from scripts.inference import HUB_REPOSITORY, image_transform

EVAL_CONCEPT = "animal.n.01"
SPLIT = "val"
N_CONTEXT = 4
IMAGES_PER_ATTR = 100
SEED = 123
CFG_SCALE = 4.0
SAMPLE_STEPS = 50
ATTR_DEF = "variation"
DEVICE = "cuda"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing ImageNet WebDataset shards")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional local VICIS inference checkpoint. If omitted, torch.hub loads the released model.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke-test", action="store_true", help="Run two classes with at most two query samples each")
    return parser.parse_args()


def shard_paths(data_dir: Path, split: str) -> list[str]:
    paths = sorted(str(path) for path in data_dir.glob(f"*{split}*.tar"))
    if not paths:
        raise FileNotFoundError(f"No *{split}*.tar shards found in {data_dir}")
    return paths


def image_key(sample):
    if "jpg" in sample:
        return "jpg"
    if "jpeg" in sample:
        return "jpeg"
    raise KeyError("Sample has neither jpg nor jpeg image data")


def class_id(sample) -> int:
    value = sample["cls"]
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, bytes):
        return int(value.decode("utf-8"))
    return int(value)


def required_eval_leaves(eval_classes: Iterable[str]) -> set[str]:
    required = set(eval_classes)
    for cls in eval_classes:
        for set_direction in SUPER_CLASS_DIR[cls]:
            for child in DIRECT_SUBCLASSES.get(set_direction, []):
                required.update(HIERARCHY_ATTR_TO_CLASSES.get(child, [child]))
    return required


def load_imagenet_images(
    data_dir: Path,
    split: str,
    eval_classes: Iterable[str],
    *,
    max_images_per_class: int | None = None,
) -> dict[str, list[torch.Tensor]]:
    required = required_eval_leaves(eval_classes)
    required_ids = {SYN_NAME_TO_ID[name] for name in required if name in SYN_NAME_TO_ID}
    transform = image_transform()
    images = defaultdict(list)
    complete_ids = set()

    dataset = wds.WebDataset(shard_paths(data_dir, split), shardshuffle=False).decode("torch")
    for sample in tqdm(dataset, desc=f"Indexing ImageNet {split}"):
        cid = class_id(sample)
        if cid not in required_ids:
            continue
        name = ID_TO_SYN_NAME[cid]
        if max_images_per_class is not None and len(images[name]) >= max_images_per_class:
            continue
        images[name].append((transform(sample[image_key(sample)]) * 2 - 1).clamp(-1, 1).cpu())
        if max_images_per_class is not None and len(images[name]) == max_images_per_class:
            complete_ids.add(cid)
            if complete_ids == required_ids:
                break

    missing = sorted(name for name in required if name in SYN_NAME_TO_ID and not images[name])
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise RuntimeError(f"No images found for required classes: {preview}{suffix}")
    return dict(images)


def sample_leaf_under(child: str, images: dict[str, list[torch.Tensor]], rng: random.Random) -> str:
    candidates = [leaf for leaf in HIERARCHY_ATTR_TO_CLASSES.get(child, [child]) if images.get(leaf)]
    if not candidates:
        raise RuntimeError(f"No loaded ImageNet leaves are available below {child}")
    return rng.choice(candidates)


def sample_context_set(
    set_direction: str,
    images: dict[str, list[torch.Tensor]],
    rng: random.Random,
) -> torch.Tensor:
    children = DIRECT_SUBCLASSES.get(set_direction)
    if not children:
        raise ValueError(f"Cannot build a context set for leaf/non-hierarchy node: {set_direction}")

    if len(children) >= N_CONTEXT:
        selected_children = rng.sample(children, k=N_CONTEXT)
    else:
        selected_children = list(children) + rng.choices(children, k=N_CONTEXT - len(children))

    leaves = [sample_leaf_under(child, images, rng) for child in selected_children]
    rng.shuffle(leaves)
    return torch.stack([rng.choice(images[leaf]) for leaf in leaves], dim=1)


def sample_query_batch(
    query_class: str,
    images: dict[str, list[torch.Tensor]],
    batch_size: int,
    rng: random.Random,
) -> torch.Tensor:
    return torch.stack([rng.choice(images[query_class]) for _ in range(batch_size)], dim=0)


def load_generation_model(args):
    if args.checkpoint is not None:
        from vicis.model import load_model

        return load_model(args.checkpoint, device=DEVICE, dtype=torch.bfloat16)

    return torch.hub.load(HUB_REPOSITORY, "vicis", pretrained=True, device=DEVICE).eval()


@torch.no_grad()
def generate_batch(
    model,
    context_set: torch.Tensor,
    query_images: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    model.eval()
    model.set_learner.dynamic_set_size = False
    directions = model.set_learner(context_set, attr_def=[ATTR_DEF]).expand(query_images.shape[0], -1, -1)
    projections = model.project_query(directions, query_images)
    context = model.direction_in_proj(directions * projections[..., None]).mean(dim=1).to(torch.bfloat16)

    generator = torch.Generator(device=context.device).manual_seed(seed)
    latent = torch.randn(
        context.shape[0],
        4,
        32,
        32,
        device=context.device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    dt = 1.0 / SAMPLE_STEPS
    for step in range(SAMPLE_STEPS, 0, -1):
        time = latent.new_full((latent.shape[0],), step / SAMPLE_STEPS)
        pos = model._position(latent)
        conditional = model.unet(latent, pos, model._conditioning(time, context, dropout=False))
        unconditional = model.unet(latent, pos, model._conditioning(time, torch.zeros_like(context), dropout=False))
        latent = latent - dt * (unconditional + CFG_SCALE * (conditional - unconditional))
    return model.ae.decode(latent)


def classifier_inputs(images: torch.Tensor) -> torch.Tensor:
    images = (images.float().clamp(-1, 1) + 1) / 2
    if images.shape[-2:] != (256, 256):
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
    top = (images.shape[-2] - 224) // 2
    left = (images.shape[-1] - 224) // 2
    images = images[:, :, top : top + 224, left : left + 224]
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images - mean) / std


def load_classifier(device: str):
    import timm

    classifier = timm.create_model("vit_large_patch16_224", pretrained=True)
    return classifier.eval().to(device)


@torch.no_grad()
def classify_generated(images: torch.Tensor, classifier) -> list[str]:
    logits = classifier(classifier_inputs(images))
    top1_ids = logits.argmax(dim=1).detach().cpu().tolist()
    return [ID_TO_SYN_NAME[index] for index in top1_ids]


def run_generation_eval(args, eval_classes: list[str], images_per_attr: int):
    if not torch.cuda.is_available():
        raise RuntimeError("VICIS eval generation requires a CUDA GPU")

    images = load_imagenet_images(
        args.data_dir,
        SPLIT,
        eval_classes,
        max_images_per_class=1 if args.smoke_test else None,
    )
    model = load_generation_model(args)
    classifier = load_classifier(DEVICE)

    counts_by_class = {}
    for class_index, query_class in enumerate(tqdm(eval_classes, desc="Evaluating classes")):
        rng = random.Random(SEED + class_index)
        counts_by_set_direction = {}
        for set_index, set_direction in enumerate(SUPER_CLASS_DIR[query_class]):
            predictions_for_direction = Counter()
            remaining = images_per_attr
            batch_index = 0
            while remaining > 0:
                current_batch = min(args.batch_size, remaining)
                context = sample_context_set(set_direction, images, rng)
                context = context[None].to(DEVICE, dtype=torch.bfloat16)
                query_images = sample_query_batch(query_class, images, current_batch, rng)
                query_images = query_images.to(DEVICE, dtype=torch.bfloat16)
                seed = SEED + class_index * 100_000 + set_index * 1_000 + batch_index
                generated = generate_batch(
                    model,
                    context,
                    query_images,
                    seed=seed,
                )
                predictions_for_direction.update(classify_generated(generated, classifier))
                remaining -= current_batch
                batch_index += 1
            counts_by_set_direction[set_direction] = predictions_for_direction
        counts_by_class[query_class] = counts_by_set_direction

    return counts_by_class


def categorical_entropy(probs, base=np.e):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs) / np.log(base)))


def target_cluster(set_direction: str, query_class: str) -> str:
    candidates = set(DIRECT_SUBCLASSES.get(set_direction, [])) & set(SUPER_CLASS_DIR[query_class])
    return sorted(candidates)[0] if candidates else query_class


def compute_accuracy_and_entropy(pred_counts: Counter, q_cluster: str, q_cls: str):
    total = sum(pred_counts.values())
    if total == 0:
        raise ValueError(f"Empty predictions for query class {q_cls}")

    pred_prob_dict = {img_cls: float(pred_ct) / total for img_cls, pred_ct in pred_counts.items()}
    cluster_classes = HIERARCHY_ATTR_TO_CLASSES[q_cluster] if q_cluster in HIERARCHY_ATTR_TO_CLASSES else [q_cluster]
    accuracy = sum(value for key, value in pred_counts.items() if key in cluster_classes) / float(total)

    entropies = {}
    for super_class in SUPER_CLASS_DIR[q_cls]:
        if super_class == "animal.n.01":
            continue
        leaves = HIERARCHY_ATTR_TO_CLASSES[super_class]
        entropies[super_class] = categorical_entropy([pred_prob_dict.get(key, 0.0) for key in leaves])
    return float(accuracy), entropies


def eval_outputs_per_set_direction(counts_by_class, eval_classes: list[str]) -> list[dict[str, float | int | str]]:
    accuracies = defaultdict(list)
    entropy_ratios_by_set_direction = defaultdict(list)

    for query_class in eval_classes:
        count_dict = counts_by_class[query_class]
        entropies = {}
        for set_direction, pred_counts in count_dict.items():
            q_cluster = target_cluster(set_direction, query_class)
            accuracy, ent = compute_accuracy_and_entropy(
                pred_counts,
                q_cluster=q_cluster,
                q_cls=query_class,
            )
            accuracies[set_direction].append(accuracy)
            entropies[set_direction] = ent

        for set_direction in count_dict:
            direct_sub = sorted(set(DIRECT_SUBCLASSES.get(set_direction, [])) & set(count_dict))
            if direct_sub:
                child = direct_sub[0]
                entropy_abstract = entropies[set_direction][child]
                entropy_specific = entropies[child][child]
                entropy_ratios_by_set_direction[set_direction].append(max(entropy_abstract / (entropy_specific + 1), 0))

    rows = []
    for set_direction in sorted(accuracies):
        entropy_ratios = entropy_ratios_by_set_direction.get(set_direction, [])
        rows.append(
            {
                "set_direction": set_direction,
                "accuracy": float(np.mean(accuracies[set_direction])),
                "diversity": float(np.mean(entropy_ratios)) if entropy_ratios else math.nan,
                "n_set_query_combinations": len(accuracies[set_direction]),
                "n_entropy_ratios": len(entropy_ratios),
            }
        )
    return rows


def eval_aggregate_metrics(per_set_direction_rows: list[dict[str, float | int | str]]):
    if not per_set_direction_rows:
        raise ValueError("No per-set-direction metrics to aggregate")
    accuracy = np.asarray([row["accuracy"] for row in per_set_direction_rows], dtype=np.float64)
    weights = np.asarray([row["n_set_query_combinations"] for row in per_set_direction_rows], dtype=np.float64)
    div_rows = [row for row in per_set_direction_rows if row["n_entropy_ratios"] > 0 and not math.isnan(row["diversity"])]
    diversity = np.asarray([row["diversity"] for row in div_rows], dtype=np.float64)
    diversity_weights = np.asarray([row["n_entropy_ratios"] for row in div_rows], dtype=np.float64)
    return {
        "accuracy_per_concept": float(np.mean(accuracy)),
        "accuracy_per_instantiation": float(np.average(accuracy, weights=weights)),
        "diversity": float(np.average(diversity, weights=diversity_weights)) if len(diversity) else math.nan,
    }


def write_summary_file(args, eval_classes: list[str], counts_by_class):
    per_set_direction = eval_outputs_per_set_direction(counts_by_class, eval_classes)
    aggregate = eval_aggregate_metrics(per_set_direction)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"metrics": aggregate}, handle, indent=2)
    return aggregate


def main():
    args = parse_args()
    eval_classes = list(HIERARCHY_ATTR_TO_CLASSES[EVAL_CONCEPT])
    images_per_attr = IMAGES_PER_ATTR
    if args.smoke_test:
        eval_classes = eval_classes[:2]
        images_per_attr = 2

    counts_by_class = run_generation_eval(args, eval_classes, images_per_attr)
    aggregate = write_summary_file(args, eval_classes, counts_by_class)
    print("Aggregate metrics:")
    for key, value in aggregate.items():
        print(f"  {key}: {value}")
    print(f"Wrote eval outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
