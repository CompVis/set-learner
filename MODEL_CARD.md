---
language:
- en
pipeline_tag: image-to-image
tags:
- image-generation
- visual-concept-learning
- set-learning
- rectified-flow
- computer-vision
- pytorch
- torch-hub
---

# VICIS — Show Me Examples: Inferring Visual Concepts from Image Sets

[![Paper](https://img.shields.io/badge/arXiv-2607.02402-b31b1b)](https://arxiv.org/abs/2607.02402)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/CompVis/set-learner)
[![Venue](https://img.shields.io/badge/ECCV-2026-green)](https://arxiv.org/abs/2607.02402)

VICIS infers a visual concept directly from a set of example images and applies it to a query image. It does not require a text description: four context images define the concept, while one query image determines how that concept should be instantiated.

## Paper

VICIS was introduced in the ECCV 2026 paper **Show Me Examples: Inferring Visual Concepts from Image Sets** by Nick Stracke, Kolja Bauer, Stefan Andreas Baumann, Miguel Angel Bautista, Josh Susskind, and Björn Ommer.

The model learns normalized concept directions from a context set, projects the DINOv2 representation of a query onto those directions, and conditions a latent rectified-flow transformer on the resulting query-specific concept embedding. It was trained at 256 × 256 resolution using the `extended_v1` ImageNet/WordNet hierarchy.

## Usage

Install the [repository requirements](https://github.com/CompVis/set-learner/blob/main/requirements.txt), then load the complete pretrained model with PyTorch Hub:

```python
import torch

model = torch.hub.load("CompVis/set-learner", "vicis", pretrained=True)

# Inputs are normalized to [-1, 1].
# context: [batch, 3, 4, 256, 256]
# query:   [batch, 3, 256, 256]
samples, conditioning = model.generate(
    context,
    query,
    seed=10_000,
    num_samples=4,
    cfg_scale=3.0,
    sample_steps=50,
)
```

The Hub entry point downloads `model.pt` from this repository, strictly loads the fixed architecture, moves it to CUDA in BF16, and returns it in evaluation mode. See the [inference CLI](https://github.com/CompVis/set-learner/blob/main/scripts/inference.py) for image preprocessing, saving individual PNGs, and producing labeled grids.

## Model details

- Four context images and one query image per inference call.
- DINOv2 ViT-L/14-reg image encoder.
- 24-layer, width-1024 set transformer producing four 256-dimensional concept directions.
- Query projection and direction conditioning in FP32.
- 24-layer, width-1024 latent rectified-flow transformer.
- Tiny Autoencoder with `4 × 32 × 32` latents for 256 × 256 images.
- Classifier-free guidance with scale 3 and 50 integration steps by default.

## Intended use and limitations

VICIS is a research model for studying visual concepts specified by image sets. A context set must contain a coherent shared concept; arbitrary or contradictory examples do not define a reliable target. Outputs can inherit biases and failure modes from ImageNet, DINOv2, and the generative training data. Generated images may be implausible, may fail to preserve the intended concept, and should not be treated as factual evidence or used for high-stakes decisions.

## Checkpoint

The expected `model.pt` format is the raw 898-entry inference state dictionary. Training-resumption checkpoints contain optimizer state and are not suitable as the inference artifact.

## Citation

```bibtex
@inproceedings{stracke2026show,
  title     = {Show Me Examples: Inferring Visual Concepts from Image Sets},
  author    = {Stracke, Nick and Bauer, Kolja and Baumann, Stefan Andreas and Bautista, Miguel Angel and Susskind, Josh and Ommer, Björn},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```
