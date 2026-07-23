"""Standalone `ImageNetModule` data pipeline for the fixed VICIS release.

This is the hierarchy-only path of
``diffusion.data.images_with_attrs.ImageNetModule``. The bundled hierarchy is
the exact ``extended_v1`` table used by the accepted model.
"""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset


HIERARCHY_NAME = "extended_v1"
with Path(__file__).with_name("extended_v1.json").open("r", encoding="utf-8") as handle:
    _HIERARCHY = json.load(handle)
HIERARCHY_ATTR_TO_CLASSES = _HIERARCHY["classes"]
DIRECT_SUBCLASSES = _HIERARCHY["direct"]
SUPER_CLASS_DIR = _HIERARCHY["super"]
ID_TO_SYN_NAME = {int(key): value for key, value in _HIERARCHY["id_to_name"].items()}
SYN_NAME_TO_ID = _HIERARCHY["name_to_id"]


def class_attributes(class_id: int) -> dict[str, str | int]:
    """Map an ImageNet class to each extended_v1 concept and its direct child."""
    class_name = ID_TO_SYN_NAME[class_id]
    attributes: dict[str, str | int] = {"cls_id": class_id}
    for concept, leaf_classes in HIERARCHY_ATTR_TO_CLASSES.items():
        if class_name not in leaf_classes:
            continue
        if class_name in DIRECT_SUBCLASSES[concept]:
            attributes[concept] = class_name
            continue
        child = next((parent for parent in SUPER_CLASS_DIR[class_name] if parent in DIRECT_SUBCLASSES[concept]), None)
        if child is not None:
            attributes[concept] = child
    return attributes


def expected_hierarchy_level(concept: str, query_class: str) -> str:
    """Return the query's direct child under a shared extended_v1 concept."""
    if concept not in HIERARCHY_ATTR_TO_CLASSES:
        raise ValueError(f"Unknown extended_v1 concept: {concept}")
    if query_class not in HIERARCHY_ATTR_TO_CLASSES[concept]:
        raise ValueError(f"{query_class} is not a descendant of {concept} in extended_v1")
    if query_class in DIRECT_SUBCLASSES[concept]:
        return query_class
    expected = next(
        (parent for parent in SUPER_CLASS_DIR[query_class] if parent in DIRECT_SUBCLASSES[concept]), None
    )
    if expected is None:
        raise ValueError(f"Could not resolve the direct child of {concept} for {query_class}")
    return expected


def preprocess_sample(sample, image_size=256):
    class_id = int(sample["cls"])
    image_key = "jpg" if "jpg" in sample else "jpeg"
    transform = T.Compose(
        [T.Resize(image_size), T.CenterCrop(image_size), T.Normalize([0.5] * 3, [0.5] * 3)]
    )
    image = transform(sample[image_key].contiguous().cpu().detach().clone()).clamp(-1, 1).bfloat16()
    if torch.isnan(image).any() or not image.numel():
        raise ValueError("Invalid image in ImageNet shard")
    return image, class_attributes(class_id)


class ImageNetDynamicWebDataset(IterableDataset):
    """Dynamic pool sampler matching images_with_attrs.ImageNetDynamicWebDataset."""

    def __init__(self, shards, *, pool_size, n_context, n_queries, batch_size, image_size=256, seed=42):
        super().__init__()
        self.pool_size, self.n_ctx, self.n_queries, self.bs = pool_size, n_context, n_queries, batch_size
        self.image_size, self.seed = image_size, seed
        self.valid_class_ids = {
            SYN_NAME_TO_ID[name] for classes in HIERARCHY_ATTR_TO_CLASSES.values() for name in classes
        }
        self.source = wds.WebDataset(shards, nodesplitter=None, workersplitter=None, shardshuffle=100).decode("torch").repeat()

    def _next_source(self):
        while True:
            image, attributes = preprocess_sample(next(self.source_iterator), self.image_size)
            if attributes["cls_id"] in self.valid_class_ids:
                return image.clone(), copy.deepcopy(attributes)

    @staticmethod
    def _sample_values(attributes):
        return set(attributes) | set(attributes.values())

    def _update_counts(self, attributes, delta):
        for value in self._sample_values(attributes):
            self.counts[value] += delta

    def _valid_concepts(self):
        return [
            concept
            for concept in HIERARCHY_ATTR_TO_CLASSES
            if self.counts[concept] > self.n_ctx + self.n_queries
            and max(self.counts[child] for child in DIRECT_SUBCLASSES[concept]) >= 2
        ]

    def _should_replace(self, attributes):
        for concept, child in attributes.items():
            if concept == "cls_id":
                continue
            if self.counts[child] - 1 <= (self.n_ctx + self.n_queries) // len(DIRECT_SUBCLASSES[concept]):
                return False
        return True

    def _refill(self, used_indices):
        for index in used_indices:
            if not self._should_replace(self.pool[index][1]):
                continue
            self._update_counts(self.pool[index][1], -1)
            self.pool[index] = self._next_source()
            self._update_counts(self.pool[index][1], 1)

    def _choose_specification(self, rng):
        concepts = rng.choices(self.valid_concepts, k=self.bs)
        context_classes, query_target_values = [], []
        for concept in concepts:
            available_context = [child for child in DIRECT_SUBCLASSES[concept] if self.counts[child] >= 1]
            selected_context = rng.choices(available_context, k=self.n_ctx)
            while any(selected_context.count(child) > self.counts[child] for child in set(selected_context)):
                selected_context = rng.choices(available_context, k=self.n_ctx)
            context_classes.append(selected_context)

            available_query = [child for child in DIRECT_SUBCLASSES[concept] if self.counts[child] >= 2]
            selected_query_children = rng.choices(available_query, k=self.n_queries)
            values = []
            for child in selected_query_children:
                if child in HIERARCHY_ATTR_TO_CLASSES:
                    possible_leaves = [leaf for leaf in HIERARCHY_ATTR_TO_CLASSES[child] if self.counts[leaf] >= 1]
                else:
                    possible_leaves = [child]
                values.extend(rng.choices(possible_leaves, k=2))
            query_target_values.append(values)
        return concepts, context_classes, query_target_values

    def _search_pool(self, context_classes, query_target_values):
        remaining_context = [classes.copy() for classes in context_classes]
        remaining_values = [values.copy() for values in query_target_values]
        context_unique = [set(values) for values in remaining_context]
        value_unique = [set(values) for values in remaining_values]
        found_values = [{value: [] for value in values} for values in value_unique]
        context = torch.full((self.bs, 3, self.n_ctx, self.image_size, self.image_size), float("nan"))
        used = set()

        for pool_index, sample in enumerate(self.pool):
            image, attributes = sample
            sample_values = self._sample_values(attributes)
            for batch_index in range(self.bs):
                matching_context = sample_values & context_unique[batch_index]
                if matching_context:
                    selected = next(iter(matching_context))
                    position = self.n_ctx - len(remaining_context[batch_index])
                    context[batch_index, :, position] = image
                    remaining_context[batch_index].remove(selected)
                    context_unique[batch_index] = set(remaining_context[batch_index])
                    used.add(pool_index)

                matching_values = list(sample_values & value_unique[batch_index])
                if matching_values:
                    selected = matching_values[0]
                    found_values[batch_index][selected].append(sample)
                    remaining_values[batch_index].remove(selected)
                    value_unique[batch_index] = set(remaining_values[batch_index])
                    used.add(pool_index)

        if torch.isnan(context).any():
            raise RuntimeError(f"Could not assemble context set; missing {[len(values) for values in remaining_context]}")
        return context, found_values, used

    def _assemble_query_targets(self, found, requested, rng):
        query = torch.empty(self.bs, 3, self.n_queries, self.image_size, self.image_size)
        target = torch.empty_like(query)
        previously_used = [{value: [] for value in element} for element in found]
        for batch_index in range(self.bs):
            for request_index, value in enumerate(requested[batch_index]):
                if found[batch_index][value]:
                    sample = found[batch_index][value].pop(0)
                    previously_used[batch_index][value].append(sample)
                else:
                    sample = rng.choice(previously_used[batch_index][value])
                destination = query if request_index % 2 == 0 else target
                destination[batch_index, :, request_index // 2] = sample[0]
        return query, target

    def __iter__(self):
        rng = random.Random(self.seed + torch.initial_seed())
        self.source_iterator = iter(self.source)
        self.pool, self.counts = [], defaultdict(int)
        for _ in range(self.pool_size):
            sample = self._next_source()
            self.pool.append(sample)
            self._update_counts(sample[1], 1)
        self.valid_concepts = self._valid_concepts()
        if not self.valid_concepts:
            raise RuntimeError("Pool contains no valid extended_v1 concepts")

        iteration = 0
        while True:
            concepts, context_classes, requested = self._choose_specification(rng)
            context, found, used = self._search_pool(context_classes, requested)
            query, target = self._assemble_query_targets(found, requested, rng)
            yield {"context_set": context, "query": query, "target": target}
            self._refill(used)
            self.valid_concepts = self._valid_concepts()
            iteration += 1
            if iteration % 10 == 0:
                rng.shuffle(self.pool)


class ImageNetModule:
    """Fixed release equivalent of diffusion.data.images_with_attrs.ImageNetModule."""

    def __init__(
        self,
        data_dir,
        *,
        batch_size=5,
        n_context=6,
        n_queries=30,
        val_batch_size=8,
        val_n_queries=3,
        pool_size=100_000,
        val_pool_size=10_000,
        hierarchy_name=HIERARCHY_NAME,
        smoke_test=False,
    ):
        if hierarchy_name != HIERARCHY_NAME:
            raise ValueError(f"The released model requires hierarchy_name={HIERARCHY_NAME!r}")
        self.data_dir, self.smoke_test = Path(data_dir), smoke_test
        self.batch_size, self.n_context, self.n_queries = batch_size, n_context, n_queries
        self.val_batch_size, self.val_n_queries = val_batch_size, val_n_queries
        self.pool_size, self.val_pool_size = pool_size, val_pool_size

    def _dataloader(self, split):
        shards = sorted(str(path) for path in self.data_dir.glob(f"*{split}*.tar"))
        if not shards:
            raise FileNotFoundError(f"No *{split}*.tar shards found in {self.data_dir}")
        dataset = ImageNetDynamicWebDataset(
            shards,
            pool_size=2_000 if self.smoke_test else (self.pool_size if split == "train" else self.val_pool_size),
            n_context=2 if self.smoke_test else self.n_context,
            n_queries=1 if self.smoke_test else (self.n_queries if split == "train" else self.val_n_queries),
            batch_size=1 if self.smoke_test else (self.batch_size if split == "train" else self.val_batch_size),
        )
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")


def make_dataloader(data_dir, *, split="train", smoke_test=False):
    module = ImageNetModule(data_dir, hierarchy_name="extended_v1", smoke_test=smoke_test)
    return module.train_dataloader() if split == "train" else module.val_dataloader()
