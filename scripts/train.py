"""Minimal single-node training entry point for the fixed VICIS model."""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from vicis.data import make_dataloader
from vicis.model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train"))
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=74_000)
    parser.add_argument("--checkpoint-freq", type=int, default=2_000)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    world_size, rank, local_rank = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0")), int(os.getenv("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    start_step = 0
    seed = 42 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = build_model(pretrained_backbones=not (args.resume or args.init_checkpoint))
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(resume_state["model"], strict=True)
        start_step = int(resume_state["step"])
    elif args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state), strict=True)
    model.to(device=device, dtype=torch.bfloat16).train()
    model.direction_in_proj.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / 300, 1.0))
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
    if distributed:
        model = DDP(model, device_ids=[local_rank], static_graph=True)
    if args.compile:
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead")

    loader = iter(make_dataloader(args.data_dir, smoke_test=args.smoke_test))
    final_step = min(args.max_steps, start_step + 1) if args.smoke_test else args.max_steps
    progress = tqdm(range(start_step, final_step), initial=start_step, total=final_step, disable=rank != 0)
    for step in progress:
        batch = next(loader)
        batch = {name: value.to(device=device, dtype=torch.bfloat16, non_blocking=True) for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress.set_postfix(loss=f"{loss.item():.5f}", grad=f"{grad_norm.item():.3f}")

        completed = step + 1
        if rank == 0 and (completed % args.checkpoint_freq == 0 or completed == final_step):
            raw_model = model.module if distributed else model
            checkpoint_dir = args.output_dir / "checkpoints" / f"step_{completed}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(raw_model.state_dict(), checkpoint_dir / "model.pt")
            torch.save(
                {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": completed},
                checkpoint_dir / "training.pt",
            )
            logging.info("Saved checkpoint %s", checkpoint_dir)
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
