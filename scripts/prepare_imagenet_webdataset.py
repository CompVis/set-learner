"""Convert a local ImageNet folder split to the WebDataset format used by VICIS.

The input directory is expected to contain one subdirectory per class. Class
directories may be named by ImageNet class index, by the synset names used in
``extended_v1.json`` (for example ``tench.n.01``), or by WordNet ID if a mapping
file is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
with (REPO_ROOT / "vicis" / "extended_v1.json").open("r", encoding="utf-8") as handle:
    SYN_NAME_TO_ID = json.load(handle)["name_to_id"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="ImageNet split directory or dataset root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated WebDataset shards")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--wnid-map",
        type=Path,
        help=(
            "Optional text or JSON mapping from ImageNet WordNet IDs to canonical class indices "
            "or extended_v1 synset names."
        ),
    )
    parser.add_argument("--maxcount", type=int, default=10_000, help="Maximum samples per output shard")
    return parser.parse_args()


def load_wnid_map(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.items()
    else:
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            left, right, *_ = line.replace(",", " ").split()
            items.append((left, right))

    mapping = {}
    for wnid, target in items:
        if isinstance(target, int) or str(target).isdigit():
            mapping[wnid] = int(target)
        elif target in SYN_NAME_TO_ID:
            mapping[wnid] = SYN_NAME_TO_ID[target]
        else:
            raise ValueError(f"Could not map {wnid!r} to an ImageNet class id: {target!r}")
    return mapping


def split_root(input_dir: Path, split: str) -> Path:
    return input_dir / split if (input_dir / split).is_dir() else input_dir


def class_id_for_dir(class_dir: Path, wnid_map: dict[str, int]) -> int:
    name = class_dir.name
    if name.isdigit():
        return int(name)
    if name in SYN_NAME_TO_ID:
        return SYN_NAME_TO_ID[name]
    if name in wnid_map:
        return wnid_map[name]
    raise ValueError(
        f"Cannot infer ImageNet class id for {class_dir}. Use class-index or extended_v1 synset "
        "directory names, or pass --wnid-map for WordNet ID directories."
    )


def iter_images(root: Path, wnid_map: dict[str, int]):
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in {root}")

    for class_dir in class_dirs:
        class_id = class_id_for_dir(class_dir, wnid_map)
        if not 0 <= class_id < 1000:
            raise ValueError(f"ImageNet class id for {class_dir} is outside [0, 999]: {class_id}")
        paths = sorted(path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        for path in paths:
            yield path, class_id


def main():
    args = parse_args()
    import webdataset as wds

    root = split_root(args.input_dir, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wnid_map = load_wnid_map(args.wnid_map)

    shard_pattern = str(args.output_dir / f"{args.split}-%06d.tar")
    with wds.ShardWriter(shard_pattern, maxcount=args.maxcount) as sink:
        for index, (path, class_id) in enumerate(tqdm(iter_images(root, wnid_map), desc=f"Writing {args.split}")):
            sink.write(
                {
                    "__key__": f"{args.split}_{class_id:04d}_{index:08d}",
                    "jpg": path.read_bytes(),
                    "cls": str(class_id).encode("utf-8"),
                }
            )


if __name__ == "__main__":
    main()
