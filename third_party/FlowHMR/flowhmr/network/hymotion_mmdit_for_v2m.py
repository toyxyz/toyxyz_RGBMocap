import math
import random
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor

from ..core.framework.loaders import load_object
from ..core.framework.type_converter import get_module_device
from .mmdit.attention import attention
from .mmdit.bricks import get_activation_layer, get_norm_layer
from .mmdit.encoders import MLP, MLPEncoder, TimestepEmbeddingEncoder
from .mmdit.modulate_layers import ModulateDiT, apply_gate, modulate
from .mmdit.positional_encoding import RotaryEmbedding
from .mmdit.query_batched_attention import (
    AttentionPlan,
    build_local_attention_plan,
    query_batched_attention,
    resolve_query_attention_config,
)
from .token_refiner import SingleTokenRefiner


class MMDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mlp_act_type: str,
        qk_norm_type: Optional[str] = None,
        qkv_bias: bool = False,
        positional_encoding_cfg: dict = {
            "max_seq_len": 5000,
            "use_real": True,
        },
        apply_rope_to_single_branch: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        assert self.feat_dim % num_heads == 0, f"feat_dim {self.feat_dim} must be divisible by num_heads {num_heads}"
        self.head_dim = self.feat_dim // num_heads

        self.mlp_hidden_dim = int(self.feat_dim * mlp_ratio)

        self._positional_encoding_cfg = positional_encoding_cfg.copy()
        self.rotary_emb = RotaryEmbedding(num_feats=self.head_dim, **self._positional_encoding_cfg)
        self.apply_rope_to_single_branch = apply_rope_to_single_branch

        self.motion_mod = ModulateDiT(
            self.feat_dim,
            factor=6,
            act_type="silu",
        )
        self.motion_norm1 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)
        self.motion_qkv = nn.Linear(self.feat_dim, self.feat_dim * 3, bias=qkv_bias)
        self.motion_q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.motion_k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.motion_out_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=qkv_bias)
        self.motion_norm2 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)
        self.motion_mlp = MLP(
            self.feat_dim,
            self.mlp_hidden_dim,
            act_type=mlp_act_type,
            bias=True,
        )

        self.text_mod = ModulateDiT(
            self.feat_dim,
            factor=6,
            act_type="silu",
        )
        self.text_norm1 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)
        self.text_qkv = nn.Linear(self.feat_dim, self.feat_dim * 3, bias=qkv_bias)
        self.text_q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.text_k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.text_out_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=qkv_bias)
        self.text_norm2 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)
        self.text_mlp = MLP(
            self.feat_dim,
            self.mlp_hidden_dim,
            act_type=mlp_act_type,
            bias=True,
        )

    def forward(
        self,
        motion_feat: Tensor,
        text_feat: Tensor,
        adapter: Tensor,
        attn_mask: Optional[Tensor] = None,
        return_attn: bool = False,
        attn_collector: Optional[List[Tensor]] = None,
        attention_plan: Optional[AttentionPlan] = None,
    ) -> Tuple[Tensor, Tensor]:
        if attention_plan is not None and (self.training or return_attn):
            raise ValueError("An attention plan requires evaluation without attention-map output")
        (
            motion_shift_msa,
            motion_scale_msa,
            motion_gate_msa,
            motion_shift_mlp,
            motion_scale_mlp,
            motion_gate_mlp,
        ) = self.motion_mod(adapter).chunk(6, dim=-1)
        (
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = self.text_mod(
            adapter
        ).chunk(6, dim=-1)

        motion_modulated = self.motion_norm1(motion_feat)
        motion_modulated = modulate(motion_modulated, shift=motion_shift_msa, scale=motion_scale_msa)
        motion_qkv = self.motion_qkv(motion_modulated)
        motion_q, motion_k, motion_v = rearrange(motion_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        motion_q = self.motion_q_norm(motion_q).to(motion_v)
        motion_k = self.motion_k_norm(motion_k).to(motion_v)

        if self.apply_rope_to_single_branch:
            # NOTE: we don't apply RoPE to text_branch_two here
            motion_q, motion_k = self.rotary_emb.apply_rotary_emb(motion_q, motion_k)

        text_modulated = self.text_norm1(text_feat)
        text_modulated = modulate(text_modulated, shift=text_shift_msa, scale=text_scale_msa)
        text_qkv = self.text_qkv(text_modulated)
        text_q, text_k, text_v = rearrange(
            text_qkv,
            "B L (K H D) -> K B L H D",
            K=3,
            H=self.num_heads,
        )
        text_q = self.text_q_norm(text_q).to(text_v)
        text_k = self.text_k_norm(text_k).to(text_v)
        if self.apply_rope_to_single_branch:
            text_q, text_k = self.rotary_emb.apply_rotary_emb(text_q, text_k)

        q = torch.cat((motion_q, text_q), dim=1)
        k = torch.cat((motion_k, text_k), dim=1)
        v = torch.cat((motion_v, text_v), dim=1)
        if not self.apply_rope_to_single_branch:
            q, k = self.rotary_emb.apply_rotary_emb(q, k)

        bsz, total_len, _, _ = q.shape
        motion_len = motion_feat.shape[1]
        text_len = text_feat.shape[1]
        dropout_p = 0.0 if not self.training else self.dropout

        ret = query_batched_attention(q, k, v, attention_plan) if attention_plan is not None else attention(
            q,
            k,
            v,
            mode="torch",  # TODO: support flash mode latter
            drop_rate=dropout_p,
            attn_mask=attn_mask,
            causal=False,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            max_seqlen_q=None,
            max_seqlen_kv=None,
            batch_size=bsz,
            training=self.training,
            return_attn=return_attn,
        )
        if isinstance(ret, tuple):
            attn_output, attn_w = ret
            if attn_collector is not None:
                attn_collector.append(attn_w.detach())
        else:
            attn_output = ret

        motion_attn_output, text_attn_output = (
            attn_output[:, :motion_len, ...],
            attn_output[:, motion_len:, ...],
        )

        motion_feat = motion_feat + apply_gate(self.motion_out_proj(motion_attn_output), gate=motion_gate_msa)
        motion_feat = motion_feat + apply_gate(
            self.motion_mlp(
                modulate(
                    self.motion_norm2(motion_feat),
                    shift=motion_shift_mlp,
                    scale=motion_scale_mlp,
                )
            ),
            gate=motion_gate_mlp,
        )

        text_feat = text_feat + apply_gate(self.text_out_proj(text_attn_output), gate=text_gate_msa)
        text_feat = text_feat + apply_gate(
            self.text_mlp(
                modulate(
                    self.text_norm2(text_feat),
                    shift=text_shift_mlp,
                    scale=text_scale_mlp,
                )
            ),
            gate=text_gate_mlp,
        )

        return motion_feat, text_feat


class MMSingleStreamBlock(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mlp_act_type: str,
        qk_norm_type: Optional[str] = None,
        qkv_bias: bool = False,
        positional_encoding_cfg: dict = {
            "max_seq_len": 5000,
            "use_real": True,
        },
        apply_rope_to_single_branch: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        assert self.feat_dim % num_heads == 0, f"feat_dim {self.feat_dim} must be divisible by num_heads {num_heads}"
        self.head_dim = self.feat_dim // num_heads

        self.mlp_hidden_dim = int(self.feat_dim * mlp_ratio)

        self._positional_encoding_cfg = positional_encoding_cfg.copy()
        self.rotary_emb = RotaryEmbedding(num_feats=self.head_dim, **self._positional_encoding_cfg)
        self.apply_rope_to_single_branch = apply_rope_to_single_branch

        self.modulation = ModulateDiT(self.feat_dim, factor=3, act_type="silu")
        self.norm = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)

        # qkv and mlp_in
        self.linear1 = nn.Linear(self.feat_dim, self.feat_dim * 3 + self.mlp_hidden_dim, bias=qkv_bias)
        # proj and mlp_out
        self.linear2 = nn.Linear(self.feat_dim + self.mlp_hidden_dim, self.feat_dim, bias=qkv_bias)

        self.q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)

        self.mlp_act = get_activation_layer(mlp_act_type)()

    def forward(
        self,
        x: Tensor,
        split_len: int,
        adapter: Tensor,
        attn_mask: Optional[Tensor] = None,
        return_attn: bool = False,
        attn_collector: Optional[List[Tensor]] = None,
        attention_plan: Optional[AttentionPlan] = None,
    ) -> Tensor:
        if attention_plan is not None and (self.training or return_attn):
            raise ValueError("An attention plan requires evaluation without attention-map output")
        (
            shift_msa,
            scale_msa,
            gate_msa,
        ) = self.modulation(
            adapter
        ).chunk(3, dim=-1)
        x_modulated = modulate(self.norm(x), shift_msa, scale_msa)

        qkv, mlp_hidden = torch.split(self.linear1(x_modulated), [3 * self.feat_dim, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        q1, q2 = q[:, :split_len, ...], q[:, split_len:, ...]
        k1, k2 = k[:, :split_len, ...], k[:, split_len:, ...]
        # apply rotary position embedding
        if self.apply_rope_to_single_branch:
            q1, k1 = self.rotary_emb.apply_rotary_emb(q1, k1)
            q2, k2 = self.rotary_emb.apply_rotary_emb(q2, k2)

        q = torch.cat((q1, q2), dim=1)
        k = torch.cat((k1, k2), dim=1)
        if not self.apply_rope_to_single_branch:
            q, k = self.rotary_emb.apply_rotary_emb(q, k)

        bsz, total_len = x_modulated.shape[:2]
        dropout_p = 0.0 if not self.training else self.dropout

        ret = query_batched_attention(q, k, v, attention_plan) if attention_plan is not None else attention(
            q,
            k,
            v,
            mode="torch",  # TODO: support flash mode latter
            drop_rate=dropout_p,
            attn_mask=attn_mask,
            causal=False,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            max_seqlen_q=None,
            max_seqlen_kv=None,
            batch_size=bsz,
            training=self.training,
            return_attn=return_attn,
        )
        if isinstance(ret, tuple):
            attn_output, attn_w = ret
            if attn_collector is not None:
                attn_collector.append(attn_w.detach())
        else:
            attn_output = ret
        output = self.linear2(torch.cat((attn_output, self.mlp_act(mlp_hidden)), 2))

        return x + apply_gate(output, gate=gate_msa)


class HunyuanMotionMMDiT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feat_dim: int,
        output_dim: Optional[int] = None,
        ctxt_input_dim: int = 4096,
        vtxt_input_dim: int = 256,
        text_refiner_module: str = "",
        text_refiner_cfg: dict = {
            "num_layers": 2,
        },
        num_layers: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        norm_type: str = "layer",
        qk_norm_type: str = "rms",
        qkv_bias: bool = True,
        dropout: float = 0.0,
        final_layer_module: str = "flowhmr/network/mmdit/encoders.FinalLayer",
        final_layer_cfg: dict = {
            "act_type": "silu",
        },
        mask_mode: Optional[str] = None,
        apply_rope_to_single_branch: bool = True,
        insert_start_token: bool = False,
        with_long_skip_connection: bool = False,
        attention_backend: Optional[str] = None,
        attention_query_chunk_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.attention_backend, self.attention_query_chunk_size = resolve_query_attention_config(
            attention_backend, attention_query_chunk_size
        )
        self.motion_input_dim = input_dim
        self.ctxt_input_dim = ctxt_input_dim
        self.vtxt_input_dim = vtxt_input_dim
        self.feat_dim = feat_dim
        self.output_dim = output_dim or input_dim
        self.mask_mode = mask_mode
        self.insert_start_token = insert_start_token
        if self.insert_start_token:
            self.start_token = nn.Parameter(torch.randn(1, feat_dim))
        self.with_long_skip_connection = with_long_skip_connection
        if self.with_long_skip_connection:
            from .mmdit.encoders import FinalLayer

            self.long_skip_net = FinalLayer(feat_dim=feat_dim, out_dim=feat_dim, act_type="silu")

        self.input_encoder = nn.Linear(in_features=input_dim, out_features=feat_dim)
        if isinstance(ctxt_input_dim, dict):
            self.ctxt_encoder = nn.ModuleDict(
                {
                    key: nn.Linear(in_features=ctxt_input_dim[key], out_features=feat_dim)
                    for key in ctxt_input_dim.keys()
                }
            )
        elif isinstance(ctxt_input_dim, int):
            self.ctxt_encoder = nn.Linear(in_features=ctxt_input_dim, out_features=feat_dim)
        else:
            raise ValueError(f"Invalid ctxt_input_dim: {ctxt_input_dim}")
        self.vtxt_encoder = MLPEncoder(in_dim=vtxt_input_dim, feat_dim=feat_dim, num_layers=2, act_type="silu")
        self.timestep_encoder = TimestepEmbeddingEncoder(
            embedding_dim=feat_dim,
            feat_dim=feat_dim,
            time_factor=kwargs.get("time_factor", 1.0),
        )

        if text_refiner_module != "" and text_refiner_module is not None:
            text_refiner_cfg.update(input_dim=feat_dim, feat_dim=feat_dim, num_heads=num_heads)
            self._text_refiner_cfg = text_refiner_cfg.copy()
            self.text_refiner = load_object(text_refiner_module, text_refiner_cfg)

        self.num_layers = num_layers
        assert num_layers % 3 == 0, f"num_layers must be divisible by 3, but got {num_layers}"
        self.mm_double_blocks_layers = int(num_layers // 3)
        self.mm_single_blocks_layers = int(num_layers - num_layers // 3)

        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    feat_dim=feat_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    mlp_act_type=mlp_act_type,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    apply_rope_to_single_branch=apply_rope_to_single_branch,
                )
                for _ in range(self.mm_double_blocks_layers)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    feat_dim=feat_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    mlp_act_type=mlp_act_type,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    apply_rope_to_single_branch=apply_rope_to_single_branch,
                )
                for _ in range(self.mm_single_blocks_layers)
            ]
        )

        final_layer_cfg.update(feat_dim=feat_dim, out_dim=self.output_dim)
        self._final_layer_cfg = final_layer_cfg.copy()
        self.final_layer = load_object(final_layer_module, final_layer_cfg)

        self.gradient_checkpointing = bool(kwargs.get("gradient_checkpointing", False))

        self.narrowband_length = kwargs.get("narrowband_length", 2.0) * 30.0
        self.narrowband_v2m_length = kwargs.get("narrowband_v2m_length", -1.0)
        self.narrowband_tm_length = kwargs.get("narrowband_tm_length", 0.0)  # T→M: 0=disabled, -1=full, >0=window(frames)
        self.narrowband_tt_length = kwargs.get("narrowband_tt_length", None)  # T→T: None=follow v2m_length

    def forward(
        self,
        x: Tensor,
        ctxt_input: Union[Tensor, Dict[str, Tensor]],
        vtxt_input: Tensor,
        timesteps: Tensor,
        x_mask_temporal: Tensor,
        ctxt_mask_temporal: Tensor,
        cond_mask_prob: float = 0.0,
        **kwargs,
    ) -> Tensor:
        device = get_module_device(self)

        origin_x = x
        motion_feat = self.input_encoder(x)
        origin_feat = motion_feat
        if self.insert_start_token:
            start_token = self.start_token[None].repeat(motion_feat.shape[0], 1, 1)
            motion_feat = torch.cat((start_token, motion_feat), dim=1)
            x_mask_temporal = torch.cat(
                [
                    torch.ones_like(x_mask_temporal[:, :1], dtype=torch.bool),
                    x_mask_temporal,
                ],
                dim=1,
            )

        timestep_feat = self.timestep_encoder(timesteps)
        if vtxt_input is None:
            vtxt_input = torch.zeros((motion_feat.shape[0], 1, self.vtxt_input_dim), device=device)
        vtxt_feat = self.vtxt_encoder(vtxt_input.float())
        adapter = timestep_feat + vtxt_feat

        if isinstance(self.ctxt_encoder, nn.ModuleDict):
            assert isinstance(ctxt_input, dict), f"ctxt_input must be a dict, got {type(ctxt_input)}"
            ctxt_feat = 0
            for key in self.ctxt_encoder.keys():
                try:
                    ctxt_feat += self.ctxt_encoder[key](ctxt_input[key])
                except:
                    raise ValueError(f"Invalid ctxt_input: {key}:{ctxt_input[key].shape}")
        else:
            assert isinstance(ctxt_input, Tensor), f"ctxt_input must be a Tensor, got {type(ctxt_input)}"
            seq_text = ctxt_input.shape[1]
            ctxt_feat = self.ctxt_encoder(ctxt_input.float())
        assert isinstance(ctxt_feat, Tensor), f"ctxt_feat must be a Tensor, got {type(ctxt_feat)}"

        motion_key_padding_mask = self._canonical_mask(x_mask_temporal).to(device)
        ctxt_key_padding_mask = self._canonical_mask(ctxt_mask_temporal).to(device)
        seq_key_padding_mask = torch.cat((motion_key_padding_mask, ctxt_key_padding_mask), dim=1)

        use_query_batched = (
            not self.training and self.attention_backend == "query_batched" and self.mask_mode == "narrowband_v2m"
        )
        if hasattr(self, "text_refiner"):
            refiner_kwargs = {}
            if use_query_batched and isinstance(self.text_refiner, SingleTokenRefiner):
                refiner_kwargs["attention_query_chunk_size"] = self.attention_query_chunk_size
            ctxt_feat = self.text_refiner(
                x=ctxt_feat, t=timesteps, mask=(ctxt_key_padding_mask == 0).to(device), **refiner_kwargs
            )

        attention_plan = None
        if use_query_batched:
            tt_length = self.narrowband_tt_length if self.narrowband_tt_length is not None else self.narrowband_v2m_length
            attention_plan = build_local_attention_plan(
                motion_feat.shape[1], ctxt_feat.shape[1], chunk_size=self.attention_query_chunk_size,
                mm_window=int(round(self.narrowband_length)),
                mc_window=int(round(self.narrowband_v2m_length)) if self.narrowband_v2m_length > 0 else None,
                cm_window=(-1 if self.narrowband_tm_length == 0 else
                           int(round(self.narrowband_tm_length)) if self.narrowband_tm_length > 0 else None),
                cc_window=int(round(tt_length)) if tt_length > 0 else None,
                key_padding_mask=torch.isneginf(seq_key_padding_mask), device=device,
            )
            seq_mask = None
        elif self.mask_mode is None:
            seq_mask = None
        elif self.mask_mode == "causal":
            motion_len = motion_feat.shape[1]
            seq_mask = torch.triu(
                torch.full((motion_len, motion_len), float("-inf"), device=device),
                diagonal=1,
            )
        elif self.mask_mode == "narrowband":
            window = int(round(self.narrowband_length))
            motion_len = motion_feat.shape[1]
            idx = torch.arange(motion_len, device=device)
            dist = (idx[None, :] - idx[:, None]).abs()
            band = dist <= window
            seq_mask = torch.full((motion_len, motion_len), float("-inf"), device=device)
            seq_mask = seq_mask.masked_fill(band, 0.0)
        elif self.mask_mode == "narrowband_v2m":
            # motion-motion narrowband mask
            mm_window = int(round(self.narrowband_length))
            motion_len = motion_feat.shape[1]
            mm_idx = torch.arange(motion_len, device=device)
            mm_dist = (mm_idx[None, :] - mm_idx[:, None]).abs()
            mm_band = mm_dist <= mm_window
            mm_mask = torch.full((motion_len, motion_len), float("-inf"), device=device)
            mm_mask = mm_mask.masked_fill(mm_band, 0.0)

            # motion-video narrowband mask
            condition_len = ctxt_feat.shape[1]
            mc_window = (
                int(round(self.narrowband_v2m_length))
                if self.narrowband_v2m_length > 0
                else max(motion_len, condition_len)
            )
            condition_idx = torch.arange(condition_len, device=device)
            mc_dist = (mm_idx[:, None] - condition_idx[None, :]).abs()
            mc_band = mc_dist <= mc_window
            mc_mask = torch.full((motion_len, condition_len), float("-inf"), device=device)
            mc_mask = mc_mask.masked_fill(mc_band, 0.0)

            # T→M mask: 0=disabled, -1=full window, >0=window size in frames
            if self.narrowband_tm_length == 0:
                cm_mask = torch.full((condition_len, motion_len), float("-inf"), device=device)
            else:
                tm_window = int(round(self.narrowband_tm_length)) if self.narrowband_tm_length > 0 else max(motion_len, condition_len)
                tm_dist = (condition_idx[:, None] - mm_idx[None, :]).abs()
                tm_band = tm_dist <= tm_window
                cm_mask = torch.full((condition_len, motion_len), float("-inf"), device=device)
                cm_mask = cm_mask.masked_fill(tm_band, 0.0)

            # T→T narrowband mask (independent or follows narrowband_v2m_length)
            tt_length = self.narrowband_tt_length if self.narrowband_tt_length is not None else self.narrowband_v2m_length
            cc_window = int(round(tt_length)) if tt_length > 0 else condition_len
            cc_dist = (condition_idx[None, :] - condition_idx[:, None]).abs()
            cc_band = cc_dist <= cc_window
            cc_mask = torch.full((condition_len, condition_len), float("-inf"), device=device)
            cc_mask = cc_mask.masked_fill(cc_band, 0.0)

            # merge to a complete attention mask
            seq_mask = torch.cat([mm_mask, mc_mask], dim=1)
            seq_mask = torch.cat([seq_mask, torch.cat([cm_mask, cc_mask], dim=1)], dim=0)
        else:
            raise ValueError(f"Unsupported mask mode: {self.mask_mode}")

        # precompute shared attention masks (broadcastable over heads)
        bsz = x.shape[0]
        motion_len = motion_feat.shape[1]
        text_len = ctxt_feat.shape[1]
        total_len = motion_len + text_len
        mask_dtype = motion_feat.dtype
        # narrowband_v2m mode handles T→M in seq_mask; other modes need _build to disable it
        tm_in_seq_mask = (self.mask_mode == "narrowband_v2m")
        attn_mask_double = None if attention_plan is not None else self._build_dmm_attn_mask_shared(
            bsz=bsz,
            motion_len=motion_len,
            text_len=text_len,
            dtype=mask_dtype,
            key_padding_mask=seq_key_padding_mask,
            attn_mask=seq_mask,
            device=device,
            disable_tm=not tm_in_seq_mask,
        )
        for i_layer, mod in enumerate(self.double_blocks):
            if self.gradient_checkpointing and self.training:
                motion_feat, ctxt_feat = torch.utils.checkpoint.checkpoint(
                    lambda m, t, a, am, _mod=mod: _mod(
                        motion_feat=m, text_feat=t, adapter=a, attn_mask=am
                    ),
                    motion_feat, ctxt_feat, adapter, attn_mask_double,
                    use_reentrant=False,
                )
            else:
                motion_feat, ctxt_feat = mod(
                    motion_feat=motion_feat,
                    text_feat=ctxt_feat,
                    adapter=adapter,
                    attn_mask=attn_mask_double,
                    attention_plan=attention_plan,
                )

        # precompute shared attention masks for single stream blocks too
        split_len = motion_feat.shape[1]
        x = torch.cat((motion_feat, ctxt_feat), 1)
        attn_mask_single = None if attention_plan is not None else self._build_smm_attn_mask_shared(
            bsz=bsz,
            split_len=split_len,
            total_len=total_len,
            dtype=mask_dtype,
            key_padding_mask=seq_key_padding_mask,
            attn_mask=seq_mask,
            device=device,
            disable_tm=not tm_in_seq_mask,
        )
        for i_layer, mod in enumerate(self.single_blocks):
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    lambda inp, a, am, _mod=mod: _mod(
                        x=inp, split_len=split_len, adapter=a, attn_mask=am
                    ),
                    x, adapter, attn_mask_single,
                    use_reentrant=False,
                )
            else:
                x = mod(
                    x=x,
                    split_len=split_len,
                    adapter=adapter,
                    attn_mask=attn_mask_single,
                    attention_plan=attention_plan,
                )

        x = x[:, :split_len, ...]
        if self.insert_start_token:
            x = x[:, 1:, ...]

        if self.with_long_skip_connection:
            x = self.long_skip_net(origin_feat, timestep_feat) + x

        predicted_res = self.final_layer(x, adapter)
        return predicted_res

    def forward_with_attn(
        self,
        x: Tensor,
        ctxt_input: Tensor,
        vtxt_input: Tensor,
        timesteps: Tensor,
        x_mask_temporal: Tensor,
        ctxt_mask_temporal: Tensor,
    ):
        device = get_module_device(self)
        motion_feat = self.input_encoder(x)
        timestep_feat = self.timestep_encoder(timesteps)
        vtxt_feat = self.vtxt_encoder(vtxt_input.float())
        adapter = timestep_feat + vtxt_feat

        motion_key_padding_mask = self._canonical_mask(x_mask_temporal).to(device)
        ctxt_key_padding_mask = self._canonical_mask(ctxt_mask_temporal).to(device)
        seq_key_padding_mask = torch.cat((motion_key_padding_mask, ctxt_key_padding_mask), dim=1)

        if self.mask_mode is None:
            seq_mask = None
        elif self.mask_mode == "causal":
            motion_len = motion_feat.shape[1]
            seq_mask = torch.triu(
                torch.full((motion_len, motion_len), float("-inf"), device=device),
                diagonal=1,
            )
        elif self.mask_mode == "narrowband":
            window = int(round(self.narrowband_length))
            motion_len = motion_feat.shape[1]
            idx = torch.arange(motion_len, device=device)
            dist = (idx[None, :] - idx[:, None]).abs()
            band = dist <= window
            seq_mask = torch.full((motion_len, motion_len), float("-inf"), device=device)
            seq_mask = seq_mask.masked_fill(band, 0.0)
        else:
            raise ValueError(f"Unsupported mask mode: {self.mask_mode}")

        ctxt_feat = self.ctxt_encoder(ctxt_input.float())
        if hasattr(self, "text_refiner"):
            ctxt_feat = self.text_refiner(x=ctxt_feat, t=timesteps, mask=(ctxt_key_padding_mask == 0).to(device))

        bsz = x.shape[0]
        motion_len = motion_feat.shape[1]
        text_len = ctxt_feat.shape[1]
        total_len = motion_len + text_len
        mask_dtype = motion_feat.dtype

        attn_collector = []

        # Double blocks
        attn_mask_double = self._build_dmm_attn_mask_shared(
            bsz=bsz,
            motion_len=motion_len,
            text_len=text_len,
            dtype=mask_dtype,
            key_padding_mask=seq_key_padding_mask,
            attn_mask=seq_mask,
            device=device,
        )
        for mod in self.double_blocks:
            motion_feat, ctxt_feat = mod(
                motion_feat=motion_feat,
                text_feat=ctxt_feat,
                adapter=adapter,
                attn_mask=attn_mask_double,
                return_attn=True,
                attn_collector=attn_collector,
            )

        # Single blocks
        split_len = motion_feat.shape[1]
        x_all = torch.cat((motion_feat, ctxt_feat), 1)
        attn_mask_single = self._build_smm_attn_mask_shared(
            bsz=bsz,
            split_len=split_len,
            total_len=total_len,
            dtype=mask_dtype,
            key_padding_mask=seq_key_padding_mask,
            attn_mask=seq_mask,
            device=device,
        )
        for mod in self.single_blocks:
            x_all = mod(
                x=x_all,
                split_len=split_len,
                adapter=adapter,
                attn_mask=attn_mask_single,
                return_attn=True,
                attn_collector=attn_collector,
            )

        x_out = x_all[:, :split_len, ...]
        predicted_res = self.final_layer(x_out, adapter)
        return predicted_res, attn_collector, motion_len, text_len

    @staticmethod
    def _canonical_mask(input_mask: Tensor) -> Tensor:
        if input_mask.ndim == 1:
            input_mask = input_mask.unsqueeze(1)
        key_padding_mask = torch.where(
            input_mask,
            torch.zeros_like(input_mask, dtype=torch.float),
            torch.full_like(input_mask, float("-inf"), dtype=torch.float),
        )
        return key_padding_mask

    def _build_dmm_attn_mask_shared(
        self,
        bsz: int,
        motion_len: int,
        text_len: int,
        dtype: torch.dtype,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
        device: torch.device,
        disable_tm: bool = True,
    ) -> Tensor:
        total_len = motion_len + text_len
        base = torch.zeros((bsz, 1, total_len, total_len), dtype=dtype, device=device)
        if attn_mask is not None:
            if attn_mask.dim() != 2:
                raise RuntimeError(f"attn_mask should be 2D, got {attn_mask.shape}")
            if attn_mask.shape == (motion_len, motion_len):
                base[:, :, :motion_len, :motion_len] += attn_mask.view(1, 1, motion_len, motion_len)
            elif attn_mask.shape == (total_len, total_len):
                base += attn_mask.view(1, 1, total_len, total_len)
            elif attn_mask.shape == (motion_len, total_len):
                base[:, :, :motion_len, :] += attn_mask.view(1, 1, motion_len, total_len)
            elif attn_mask.shape == (total_len, motion_len):
                base[:, :, :, :motion_len] += attn_mask.view(1, 1, total_len, motion_len)
            else:
                raise RuntimeError(
                    f"attn_mask should be 2D with one of the following valid shapes: {(motion_len, motion_len)} or {(total_len, total_len)} or {(motion_len, total_len)} or {(total_len, motion_len)}, got {attn_mask.shape}"
                )
        if key_padding_mask is not None:
            mask_total_len = key_padding_mask.shape[1]
            if mask_total_len == motion_len:
                pad = torch.zeros((bsz, text_len), dtype=key_padding_mask.dtype, device=device)
                key_padding_mask = torch.cat((key_padding_mask, pad), dim=-1)
            base = base + key_padding_mask.view(bsz, 1, 1, total_len)
        # disable T→M only when not already handled by seq_mask
        if disable_tm:
            base[:, :, motion_len:, :motion_len] = float("-inf")
        return base

    def _build_smm_attn_mask_shared(
        self,
        bsz: int,
        split_len: int,
        total_len: int,
        dtype: torch.dtype,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
        device: torch.device,
        disable_tm: bool = True,
    ) -> Tensor:
        base = torch.zeros((bsz, 1, total_len, total_len), dtype=dtype, device=device)
        if attn_mask is not None:
            if attn_mask.dim() != 2:
                raise RuntimeError(f"attn_mask should be 2D, got {attn_mask.shape}")
            if attn_mask.shape == (split_len, split_len):
                base[:, :, :split_len, :split_len] += attn_mask.view(1, 1, split_len, split_len)
            elif attn_mask.shape == (total_len, total_len):
                base += attn_mask.view(1, 1, total_len, total_len)
            elif attn_mask.shape == (split_len, total_len):
                base[:, :, :split_len, :] += attn_mask.view(1, 1, split_len, total_len)
            elif attn_mask.shape == (total_len, split_len):
                base[:, :, :, :split_len] += attn_mask.view(1, 1, total_len, split_len)
            else:
                raise RuntimeError(
                    f"attn_mask should be 2D with one of the following valid shapes: {(split_len, split_len)} or {(total_len, total_len)} or {(split_len, total_len)} or {(total_len, split_len)}, got {attn_mask.shape}"
                )
        if key_padding_mask is not None:
            mask_total_len = key_padding_mask.shape[1]
            if mask_total_len == split_len:
                pad = torch.zeros(
                    (bsz, total_len - split_len),
                    dtype=key_padding_mask.dtype,
                    device=device,
                )
                key_padding_mask = torch.cat((key_padding_mask, pad), dim=-1)
            base = base + key_padding_mask.view(bsz, 1, 1, total_len)
        # disable T→M only when not already handled by seq_mask
        if disable_tm:
            base[:, :, split_len:, :split_len] = float("-inf")
        return base

    def params_count(self):
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            counts = {
                "refiner": (
                    sum(p.numel() for p in self.text_refiner.parameters()) if hasattr(self, "text_refiner") else 0
                ),
                "double": sum(
                    [
                        sum(p.numel() for p in block.motion_qkv.parameters())
                        + sum(p.numel() for p in block.motion_out_proj.parameters())
                        + sum(p.numel() for p in block.motion_mlp.parameters())
                        + sum(p.numel() for p in block.text_qkv.parameters())
                        + sum(p.numel() for p in block.text_out_proj.parameters())
                        + sum(p.numel() for p in block.text_mlp.parameters())
                        for block in self.double_blocks
                    ]
                ),
                "single": sum(
                    [
                        sum(p.numel() for p in block.linear1.parameters())
                        + sum(p.numel() for p in block.linear2.parameters())
                        for block in self.single_blocks
                    ]
                ),
                "final": sum(p.numel() for p in self.final_layer.parameters()),
                "total": sum(p.numel() for p in self.parameters()),
            }
            extra_mod = sum(
                [
                    sum(p.numel() for p in block.motion_mod.parameters())
                    + sum(p.numel() for p in block.text_mod.parameters())
                    for block in self.double_blocks
                ]
            ) + sum([sum(p.numel() for p in block.modulation.parameters()) for block in self.single_blocks])
            extra_enc = (
                sum(p.numel() for p in self.input_encoder.parameters())
                + sum(p.numel() for p in self.ctxt_encoder.parameters())
                + sum(p.numel() for p in self.vtxt_encoder.parameters())
                + sum(p.numel() for p in self.timestep_encoder.parameters())
            )
            counts["attn+mlp"] = counts["double"] + counts["single"] + counts["refiner"]
            print(f"Extra encoders parameters: {extra_enc/1e9:.2f}B")
            print(f"Attn+mlp parameters: {counts['attn+mlp'] / 1e9:.2f}B")
            print(f"Modulations parameters: {extra_mod/1e9:.2f}B")
            print(f"Final layer parameters: {counts['final'] / 1e9:.2f}B")
            print(f"Total parameters: {counts['total'] / 1e9:.2f}B")
            return counts

    def flops_count(self, motion_seq_len: int = 200, text_seq_len: int = 256, batch_size: int = 1):
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            flops = {}

            # input_encoder: motion_seq_len * input_dim * feat_dim
            input_encoder_flops = motion_seq_len * self.motion_input_dim * self.feat_dim
            # ctxt_encoder: text_seq_len * ctxt_input_dim * feat_dim
            ctxt_encoder_flops = text_seq_len * self.ctxt_input_dim * self.feat_dim

            vtxt_encoder_flops = self.vtxt_input_dim * self.feat_dim + self.feat_dim * self.feat_dim

            # timestep_encoder: embedding + MLP
            timestep_encoder_flops = self.feat_dim * self.feat_dim

            encoder_flops = input_encoder_flops + ctxt_encoder_flops + vtxt_encoder_flops + timestep_encoder_flops
            flops["encoders"] = encoder_flops

            # Text Refiner FLOPs
            if hasattr(self, "text_refiner"):
                refiner_flops = 0
                refiner_layers = getattr(self, "_text_refiner_cfg", {}).get("num_layers", 0)
                for _ in range(refiner_layers):
                    qkv_flops = text_seq_len * self.feat_dim * (self.feat_dim * 3)
                    attn_flops = text_seq_len * text_seq_len * self.feat_dim
                    proj_flops = text_seq_len * self.feat_dim * self.feat_dim
                    mlp_flops = (
                        text_seq_len * self.feat_dim * (self.feat_dim * 4)
                        + text_seq_len * (self.feat_dim * 4) * self.feat_dim
                    )
                    refiner_flops += qkv_flops + attn_flops + proj_flops + mlp_flops
                flops["refiner"] = refiner_flops
            else:
                flops["refiner"] = 0

            # Double Stream Blocks FLOPs
            double_flops = 0
            total_seq_len = motion_seq_len + text_seq_len

            for _ in range(self.mm_double_blocks_layers):
                # Motion branch
                motion_qkv_flops = motion_seq_len * self.feat_dim * (self.feat_dim * 3)
                motion_attn_flops = motion_seq_len * total_seq_len * self.feat_dim
                motion_proj_flops = motion_seq_len * self.feat_dim * self.feat_dim
                motion_mlp_flops = (
                    motion_seq_len * self.feat_dim * int(self.feat_dim * 4.0)
                    + motion_seq_len * int(self.feat_dim * 4.0) * self.feat_dim
                )

                # Text branch
                text_qkv_flops = text_seq_len * self.feat_dim * (self.feat_dim * 3)
                text_attn_flops = text_seq_len * total_seq_len * self.feat_dim
                text_proj_flops = text_seq_len * self.feat_dim * self.feat_dim
                text_mlp_flops = (
                    text_seq_len * self.feat_dim * int(self.feat_dim * 4.0)
                    + text_seq_len * int(self.feat_dim * 4.0) * self.feat_dim
                )

                # Modulation FLOPs
                motion_mod_flops = self.feat_dim * (self.feat_dim * 6) # ModulateDiT
                text_mod_flops = self.feat_dim * (self.feat_dim * 6)

                block_flops = (
                    motion_qkv_flops
                    + motion_attn_flops
                    + motion_proj_flops
                    + motion_mlp_flops
                    + text_qkv_flops
                    + text_attn_flops
                    + text_proj_flops
                    + text_mlp_flops
                    + motion_mod_flops
                    + text_mod_flops
                )
                double_flops += block_flops

            flops["double"] = double_flops

            # Single Stream Blocks FLOPs
            single_flops = 0
            for _ in range(self.mm_single_blocks_layers):
                linear1_flops = total_seq_len * self.feat_dim * (self.feat_dim * 3 + int(self.feat_dim * 4.0))

                attn_flops = total_seq_len * total_seq_len * self.feat_dim

                # Output projection + MLP output
                linear2_flops = total_seq_len * (self.feat_dim + int(self.feat_dim * 4.0)) * self.feat_dim

                # Modulation
                mod_flops = self.feat_dim * (self.feat_dim * 3)

                block_flops = linear1_flops + attn_flops + linear2_flops + mod_flops
                single_flops += block_flops

            flops["single"] = single_flops

            # Final Layer FLOPs
            final_flops = motion_seq_len * self.feat_dim * self.output_dim
            flops["final"] = final_flops

            flops["total"] = sum(flops.values())
            for key in flops:
                flops[key] *= batch_size

            gflops = {k: v / 1e9 for k, v in flops.items()}
            print(f"Encoders GFLOPs: {gflops['encoders']:.2f}G")
            print(f"Text Refiner GFLOPs: {gflops['refiner']:.2f}G")
            print(f"Double Blocks GFLOPs: {gflops['double']:.2f}G")
            print(f"Single Blocks GFLOPs: {gflops['single']:.2f}G")
            print(f"Final Layer GFLOPs: {gflops['final']:.2f}G")
            print(f"Total GFLOPs: {gflops['total']:.2f}G")

            return gflops


def visualize_mask(mask, motion_len, title=""):
    import matplotlib.pyplot as plt
    import os

    if len(mask.shape) == 2:
        m = mask.detach().float()
    elif len(mask.shape) == 3:
        m = mask[0].detach().float()
    elif len(mask.shape) == 4:
        m = mask[0, 0].detach().float()

    m = torch.where(torch.isinf(m), torch.full_like(m, float("nan")), m)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(m.cpu(), cmap="viridis", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axvline(motion_len - 0.5, color="w", lw=1)
    plt.axhline(motion_len - 0.5, color="w", lw=1)
    plt.xlabel("key index")
    plt.ylabel("query index")
    plt.title(title)
    plt.tight_layout()

    os.makedirs("output/test", exist_ok=True)
    img_name = title if title else "test"
    plt.savefig(f"output/test/{img_name}.png")


if __name__ == "__main__":

    from configs._base_.model_network_base import MOTION_MODEL_CONFIG # pyright: ignore

    network_module_cfg = MOTION_MODEL_CONFIG["1.04B"]["network_module_args"]
    network_module_cfg = dict(network_module_cfg)

    bsz, seq_len, input_dim = 1, 450, 272
    network_module_cfg["input_dim"] = input_dim
    MMDiT = HunyuanMotionMMDiT(**network_module_cfg)
    print("=== params ===")
    MMDiT.params_count()
    print("\n=== FLOPs ===")
    MMDiT.flops_count(motion_seq_len=seq_len, text_seq_len=256, batch_size=bsz)

    x = torch.randn(bsz, seq_len, input_dim)
    ctxt_condition = torch.randn(bsz, 256, 4096)
    vtxt_condition = torch.randn(bsz, 1, 768)
    timesteps = torch.randint(0, 1000, (bsz,))
    length = torch.arange(seq_len).unsqueeze(0).repeat(bsz, 1)
    ctxt_length = torch.arange(256).unsqueeze(0).repeat(bsz, 1)
    x_mask_temporal = length < 100
    ctxt_mask_temporal = ctxt_length < 50
    x = MMDiT(
        x=x,
        ctxt_input=ctxt_condition,
        vtxt_input=vtxt_condition,
        timesteps=timesteps,
        x_mask_temporal=x_mask_temporal,
        ctxt_mask_temporal=ctxt_mask_temporal,
    )
    assert x.shape == (
        bsz,
        seq_len,
        input_dim,
    ), f"unexpected output shape: {x.shape}, which should be ({bsz}, {seq_len}, {input_dim})"
    print(x.shape)
