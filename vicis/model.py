"""Fixed VICIS model used for the ECCV 2026 release.

The implementation is intentionally self-contained and has no dependency on the
original research repository. Architecture constants match arXiv:2607.02402.
"""

from __future__ import annotations

import math
import types
from functools import partial, reduce
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import AutoencoderTiny
from einops import rearrange, repeat
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def _rms_norm(x, scale, eps=1e-6):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    inv = torch.rsqrt(torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True) + eps)
    return x * (scale.to(dtype) * inv).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return _rms_norm(x, self.scale)


class AdaRMSNorm(nn.Module):
    def __init__(self, width, cond_width):
        super().__init__()
        self.linear = nn.Linear(cond_width, width, bias=False)

    def forward(self, x, cond):
        return _rms_norm(x, self.linear(cond)[:, None] + 1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.register_buffer("weight", torch.randn(out_features // 2, in_features))

    def forward(self, x):
        phase = 2 * math.pi * x @ self.weight.T
        return torch.cat((phase.cos(), phase.sin()), dim=-1)


class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features * 2, bias=False)
        self.out_features = out_features

    def forward(self, x):
        value, gate = F.linear(x, self.weight).chunk(2, dim=-1)
        return value * F.silu(gate)


class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features * 2, bias=False)
        self.out_features = out_features

    def forward(self, x):
        value, gate = F.linear(x, self.weight).chunk(2, dim=-1)
        return value * F.gelu(gate)


class FeedForwardBlock(nn.Module):
    def __init__(self, width=1024, cond_width=1024):
        super().__init__()
        self.norm = AdaRMSNorm(width, cond_width)
        self.up_proj = LinearSwiGLU(width, width * 3)
        self.dropout = nn.Dropout(0.0)
        self.down_proj = nn.Linear(width * 3, width, bias=False)

    def forward(self, x, cond_norm):
        return x + self.down_proj(self.dropout(self.up_proj(self.norm(x, cond_norm))))


class MappingFeedForwardBlock(nn.Module):
    def __init__(self, width=1024):
        super().__init__()
        self.norm = RMSNorm(width)
        self.up_proj = LinearGEGLU(width, width * 3)
        self.dropout = nn.Dropout(0.0)
        self.down_proj = nn.Linear(width * 3, width, bias=False)

    def forward(self, x):
        return x + self.down_proj(self.dropout(self.up_proj(self.norm(x))))


class MappingNetwork(nn.Module):
    def __init__(self, depth=2, width=1024):
        super().__init__()
        self.in_norm = RMSNorm(width)
        self.blocks = nn.ModuleList(MappingFeedForwardBlock(width) for _ in range(depth))
        self.out_norm = RMSNorm(width)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


def make_axial_pos_2d(height, width, *, device=None, dtype=None):
    def centers(n, lo=-1.0, hi=1.0):
        edges = torch.linspace(lo, hi, n + 1, device=device, dtype=dtype)
        return (edges[:-1] + edges[1:]) / 2

    y, x = centers(height), centers(width)
    return torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1).view(height * width, 2)


class AxialRoPE2D(nn.Module):
    def __init__(self, dim=32, n_heads=16):
        super().__init__()
        lo, hi = math.log(math.pi), math.log(10 * math.pi)
        freq = torch.stack([torch.linspace(lo, hi, n_heads * dim // 4 + 1)[:-1].exp()] * 2)
        self.freqs = nn.Parameter(freq.view(2, dim // 4, n_heads).mT.contiguous(), requires_grad=False)

    def forward(self, pos):
        y = pos[..., None, 0:1] * self.freqs[0].to(pos.dtype)
        x = pos[..., None, 1:2] * self.freqs[1].to(pos.dtype)
        return torch.cat((y, x), dim=-1)

    @staticmethod
    def apply_emb(x, theta):
        dtype, out_dtype = torch.float32, x.dtype
        dim = theta.shape[-1]
        x1, x2, rest = x[..., :dim], x[..., dim : 2 * dim], x[..., 2 * dim :]
        cos, sin = theta.to(dtype).cos(), theta.to(dtype).sin()
        y1 = x1.to(dtype) * cos - x2.to(dtype) * sin
        y2 = x2.to(dtype) * cos + x1.to(dtype) * sin
        return torch.cat((y1.to(out_dtype), y2.to(out_dtype), rest), dim=-1)


def _cosine_scale(q, k, scale):
    dtype = torch.float32
    qn = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    kn = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    root = torch.sqrt(scale.to(dtype))
    return q * (root * torch.rsqrt(qn + 1e-6)).to(q.dtype), k * (root * torch.rsqrt(kn + 1e-6)).to(k.dtype)


class AttentionBlock(nn.Module):
    def __init__(self, *, flex=False):
        super().__init__()
        self.d_head, self.n_heads, self.flex = 64, 16, flex
        self.norm = AdaRMSNorm(1024, 1024)
        self.qkv_proj = nn.Linear(1024, 3072, bias=False)
        self.scale = nn.Parameter(torch.full((16,), 10.0))
        self.pos_emb = AxialRoPE2D(32, 16)
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(1024, 1024, bias=False)

    def forward(self, x, pos, cond_norm, block_mask=None):
        skip = x
        qkv = self.qkv_proj(self.norm(x, cond_norm))
        theta = self.pos_emb(pos.to(qkv.dtype)).movedim(-2, -3)
        q, k, v = rearrange(qkv, "b l (t h d) -> t b h l d", t=3, h=16, d=64)
        q, k = _cosine_scale(q, k, self.scale[:, None, None])
        q, k = self.pos_emb.apply_emb(q, theta), self.pos_emb.apply_emb(k, theta)
        if self.flex:
            pad = (-q.shape[-2]) % 128
            if pad:
                zeros = q.new_zeros(q.shape[0], q.shape[1], pad, q.shape[3])
                q, k, v = (torch.cat((item, zeros), dim=-2) for item in (q, k, v))
            out = flex_attention(q, k, v, scale=1.0, block_mask=block_mask)
            if pad:
                out = out[:, :, :-pad]
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        out = rearrange(out, "b h l d -> b l (h d)")
        return skip + self.out_proj(self.dropout(out))


class TransformerLayer(nn.Module):
    def __init__(self, *, flex=False):
        super().__init__()
        self.self_attn = AttentionBlock(flex=flex)
        self.ff = FeedForwardBlock()

    def forward(self, x, pos, cond_norm, block_mask=None):
        return self.ff(self.self_attn(x, pos, cond_norm, block_mask), cond_norm)


class TokenMerge2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 1024, bias=False)

    def forward(self, x, pos):
        x = rearrange(x, "... (h ph) (w pw) c -> ... h w (ph pw c)", ph=2, pw=2)
        pos = rearrange(pos, "... (h ph) (w pw) c -> ... h w (ph pw) c", ph=2, pw=2).mean(-2)
        return self.proj(x), pos


class TokenSplitLast2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(1024)
        self.proj = nn.Linear(1024, 16, bias=False)

    def forward(self, x):
        return rearrange(self.proj(self.norm(x)), "... h w (ph pw c) -> ... (h ph) (w pw) c", ph=2, pw=2)


class SimpleOutProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(1024)
        self.proj = nn.Linear(1024, 1024, bias=False)

    def forward(self, x):
        return self.proj(self.norm(x))


class Transformer(nn.Module):
    def __init__(self, *, set_encoder=False):
        super().__init__()
        self.auto_reshape = not set_encoder
        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        self.merges, self.splits = nn.ModuleList(), nn.ModuleList()
        self.mid_level = nn.ModuleList(TransformerLayer(flex=set_encoder) for _ in range(24))
        self.mid_merge = nn.Identity() if set_encoder else TokenMerge2D()
        self.mid_split = SimpleOutProj() if set_encoder else TokenSplitLast2D()

    def forward(self, x, pos, cond_norm, block_mask=None):
        x, pos = rearrange(x, "b c ... -> b ... c"), rearrange(pos, "b c ... -> b ... c")
        skip = x
        if not isinstance(self.mid_merge, nn.Identity):
            x, pos = self.mid_merge(x, pos)
        batch, *dims, channels = x.shape
        if self.auto_reshape:
            x, pos = x.reshape(batch, -1, channels), pos.reshape(batch, -1, 2)
        for layer in self.mid_level:
            x = layer(x, pos, cond_norm, block_mask)
        if self.auto_reshape:
            x = x.reshape(batch, *dims, channels)
        x = self.mid_split(x) if self.auto_reshape else self.mid_split(x)
        return rearrange(x, "b ... c -> b c ...")


def _dino_attention(self, x):
    batch, tokens, channels = x.shape
    qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
    out = F.scaled_dot_product_attention(*qkv.unbind(0), dropout_p=0.0)
    return self.proj_drop(self.proj(out.transpose(1, 2).reshape(batch, tokens, channels)))


class DinoV2Reg(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=pretrained)
        for block in self.model.blocks:
            block.attn.forward = types.MethodType(_dino_attention, block.attn)
            block.compile()
        self.model.forward = partial(self.model.get_intermediate_layers, return_class_token=True, reshape=True)
        self.model.requires_grad_(False).eval()

    def train(self, mode=True):
        # The accepted model uses DINOv2 as a frozen feature extractor.
        return super().train(False)

    def forward(self, images):
        side = min(images.shape[-2:])
        images = TF.center_crop(images, [side, side])
        if side // 224 > 1:
            images = F.avg_pool2d(images, side // 224)
        images = F.interpolate(images, (224, 224), mode="bilinear")
        images = TF.normalize((images + 1) / 2, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return self.model(images)[0]


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = DinoV2Reg(pretrained=pretrained)

    def forward(self, x):
        return self.model(x)


class SetLearner(nn.Module):
    def __init__(self, pretrained_backbones=False):
        super().__init__()
        self.img_encoder = ImageEncoder(pretrained=pretrained_backbones)
        self.sequence_encoder = Transformer(set_encoder=True)
        self.img_query = nn.Parameter(torch.randn(1, 1, 1024))
        self.seq_query = nn.Parameter(torch.randn(1, 1024, 4))
        self.attr_def_enc = nn.Linear(1, 1024, bias=False)
        self.set_emb_out_proj = nn.Linear(1024, 256, bias=False)
        self.num_set_dirs, self.dynamic_set_size, self.min_set_size = 4, True, 2
        self.block_mask_cache = {}

    def forward(self, context_set, attr_def=None):
        batch, _, set_size, _, _ = context_set.shape
        bounds = [i + 2 if self.dynamic_set_size else set_size for i in range(batch)]
        keep = torch.cat([torch.arange(set_size) < n for n in bounds]).to(context_set.device)
        features, _ = self.img_encoder(rearrange(context_set, "b c n h w -> (b n) c h w")[keep])
        _, channels, height, width = features.shape
        pos = make_axial_pos_2d(height, width, device=features.device, dtype=features.dtype)
        pos = repeat(pos, "(h w) c -> n c h w", n=sum(bounds), h=height, w=width)
        document_id = torch.cat([torch.full((n * height * width,), i, device=features.device) for i, n in enumerate(bounds)])
        features = rearrange(features, "n c h w -> (n h w) c")
        pos = rearrange(pos, "n c h w -> (n h w) c")
        features = torch.cat((features, repeat(self.seq_query, "1 c d -> (b d) c", b=batch)))
        pos = torch.cat((pos, pos.new_zeros(batch * 4, 2)))
        document_id = torch.cat((document_id, torch.arange(batch, device=features.device).repeat_interleave(4)))
        length, padded = features.shape[0], math.ceil(features.shape[0] / 128) * 128
        document_id = F.pad(document_id, (0, padded - length), value=-1)

        def same_document(_b, _h, q, kv):
            return document_id[q] == document_id[kv]

        if padded not in self.block_mask_cache:
            self.block_mask_cache[padded] = create_block_mask(same_document, 1, 1, padded, padded, device=features.device)
        if attr_def is None:
            attr_def = ["commonality"] * batch
        cond = self.attr_def_enc(features.new_tensor([[1.0 if x == "commonality" else 0.0] for x in attr_def]))
        out = self.sequence_encoder(features[None].movedim(1, 2), pos[None].movedim(1, 2), cond, self.block_mask_cache[padded])
        out = rearrange(out[0, :, -4 * batch :], "c (b d) -> b d c", b=batch, d=4)
        return F.normalize(self.set_emb_out_proj(out).float(), dim=-1)


class TinyAutoencoderKL(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.ae = AutoencoderTiny.from_pretrained("madebyollin/taesd") if pretrained else AutoencoderTiny()
        self.ae.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, image):
        return self.ae.encode(image, return_dict=False)[0]

    @torch.no_grad()
    def decode(self, latent):
        return self.ae.decode(latent, return_dict=False)[0]


class VICIS(nn.Module):
    def __init__(self, pretrained_backbones=False):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, 1024))
        self.unet = Transformer(set_encoder=False)
        self.ae = TinyAutoencoderKL(pretrained=pretrained_backbones)
        self.time_emb = FourierFeatures(1, 1024)
        self.time_in_proj = nn.Linear(1024, 1024, bias=False)
        self.mapping = MappingNetwork()
        self.set_learner = SetLearner(pretrained_backbones=pretrained_backbones)
        self.img_proj = nn.Linear(1024, 1024, bias=False)
        self.norm = nn.LayerNorm(1024)
        self.sampling_mode_enc = nn.Linear(1, 1024, bias=False)
        self.query_emb_proj = nn.Linear(1024, 256, bias=False)
        self.proj_scalar_fourier_emb = FourierFeatures(1, 1024)
        self.scalar_in_proj = nn.Linear(1024, 1024, bias=False)
        self.proj_scalar_mapping = MappingNetwork()
        self.direction_in_proj = nn.Linear(256, 1024, bias=False)
        self.c_dropout, self.cfg_scale = 0.1, 3.0

    def project_query(self, directions, query):
        _, query_embedding = self.set_learner.img_encoder(query)
        return torch.einsum("bdc,bc->bd", directions.float(), self.query_emb_proj(query_embedding).float())

    def context_embedding(self, context_set, query):
        directions = self.set_learner(context_set)
        projections = self.project_query(directions, query)
        context = self.direction_in_proj(directions * projections[..., None]).mean(dim=1)
        return context.to(torch.bfloat16), directions, projections

    def _conditioning(self, time, context, *, dropout):
        time = self.time_in_proj(self.time_emb(time[..., None]))
        if dropout:
            keep = (torch.rand(context.shape[0], device=context.device) >= self.c_dropout).to(context.dtype)
            context = context * keep[:, None]
        return self.mapping(time + context)

    def _position(self, latent):
        batch, _, height, width = latent.shape
        return repeat(make_axial_pos_2d(height, width, device=latent.device), "(h w) c -> b c h w", b=batch, h=height, w=width)

    def training_loss(self, context_set, query, target):
        batch, _, queries, _, _ = query.shape
        directions = self.set_learner(context_set)
        directions = repeat(directions, "b d c -> (b q) d c", q=queries)
        query = rearrange(query, "b c q h w -> (b q) c h w")
        target = rearrange(target, "b c q h w -> (b q) c h w")
        projections = self.project_query(directions, query)
        context = self.direction_in_proj(directions * projections[..., None]).mean(dim=1).to(torch.bfloat16)
        clean = self.ae.encode(target)
        time = torch.rand(clean.shape[0], device=clean.device, dtype=clean.dtype)
        noise = torch.randn_like(clean)
        time_expanded = time[:, None, None, None]
        noisy = (1 - time_expanded) * clean + time_expanded * noise
        cond = self._conditioning(time, context, dropout=True)
        velocity = self.unet(noisy, self._position(noisy), cond)
        return ((noise - clean - velocity) ** 2).flatten(1).mean(1).mean()

    def forward(self, context_set, query, target):
        return self.training_loss(context_set, query, target)

    @torch.no_grad()
    def generate(self, context_set, query, *, seed=0, num_samples=4, cfg_scale=3.0, sample_steps=50):
        self.eval()
        self.set_learner.dynamic_set_size = False
        context, directions, projections = self.context_embedding(context_set, query)
        context = context.expand(num_samples, -1)
        generator = torch.Generator(device=context.device).manual_seed(seed)
        latent = torch.randn(num_samples, 4, 32, 32, device=context.device, dtype=torch.bfloat16, generator=generator)
        dt = 1.0 / sample_steps
        for step in range(sample_steps, 0, -1):
            time = latent.new_full((num_samples,), step / sample_steps)
            pos = self._position(latent)
            conditional = self.unet(latent, pos, self._conditioning(time, context, dropout=False))
            unconditional = self.unet(latent, pos, self._conditioning(time, torch.zeros_like(context), dropout=False))
            latent = latent - dt * (unconditional + cfg_scale * (conditional - unconditional))
        return self.ae.decode(latent), {"directions": directions, "projections": projections, "context": context[:1]}


def build_model(*, pretrained_backbones=False) -> VICIS:
    return VICIS(pretrained_backbones=pretrained_backbones)


def load_model(checkpoint: str | Path, device: str | torch.device = "cuda", dtype=torch.bfloat16) -> VICIS:
    model = build_model()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=dtype)
    # Projection is deliberately performed in FP32 in the accepted model.
    model.direction_in_proj.float()
    return model.eval()
