import dataclasses
import json
import math
from pathlib import Path

import triton
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from triton_kernels import rmsnorm_kernel
from triton_kernels import rotary_position_embedding_kernel
from triton_kernels import flash_attention_kernel
from triton_kernels import mlp_kernel_step1
from triton_kernels import mlp_kernel_step2
from triton_kernels import fused_add_rmsnorm_kernel

@dataclasses.dataclass
class ModelConfig:
    head_dim: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    torch_dtype: str
    vocab_size: int

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, use_triton=False):
        if use_triton:
            orig_shape = x.shape
            x_2d = x.view(-1, orig_shape[-1])
            M, N = x_2d.shape
            out = torch.empty_like(x_2d)
            
            BLOCK_SIZE = triton.next_power_of_2(N)
            num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
            
            rmsnorm_kernel[(M,)](
                x_2d, self.weight, out, x_2d.stride(0), N, self.eps,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
            )
            return out.view(*orig_shape)
        else:
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_weight = nn.Parameter(torch.empty((2 * intermediate_size, hidden_size)))
        self.down_proj_weight = nn.Parameter(torch.empty((hidden_size, intermediate_size)))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        gate_key = prefix + "gate_proj.weight"
        up_key = prefix + "up_proj.weight"
        down_key = prefix + "down_proj.weight"

        if gate_key in state_dict and up_key in state_dict:
            gate_w = state_dict.pop(gate_key)
            up_w = state_dict.pop(up_key)
            state_dict[prefix + "gate_up_weight"] = torch.cat([gate_w, up_w], dim=0)
            
        if down_key in state_dict:
            state_dict[prefix + "down_proj_weight"] = state_dict.pop(down_key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x, use_triton=False):
        orig_shape = x.shape
        x_2d = x.view(-1, self.hidden_size)
        
        if use_triton:
            M, K = x_2d.shape
            N = self.intermediate_size

            step1_out = torch.empty((M, N), device=x.device, dtype=x.dtype)
            grid1 = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
            
            mlp_kernel_step1[grid1](
                x_2d, self.gate_up_weight, self.gate_up_weight, step1_out,
                M, N, K,
                x_2d.stride(0), x_2d.stride(1),
                self.gate_up_weight.stride(1), self.gate_up_weight.stride(0), 
                step1_out.stride(0), step1_out.stride(1),
                BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2
            )
            
            final_out_fp32 = torch.zeros((M, K), device=x.device, dtype=torch.float32)
            split_k = 4 if M <= 16 else 1 
            grid2 = (triton.cdiv(M, 64), triton.cdiv(K, 64), split_k)

            mlp_kernel_step2[grid2](
                step1_out, self.down_proj_weight, final_out_fp32,
                M, K, N, 
                step1_out.stride(0), step1_out.stride(1),
                self.down_proj_weight.stride(1), self.down_proj_weight.stride(0),
                final_out_fp32.stride(0), final_out_fp32.stride(1),
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, SPLIT_K=split_k
            )
            final_out = final_out_fp32.to(x.dtype)
            return final_out.view(*orig_shape)
        else:
            gate_up = F.linear(x_2d, self.gate_up_weight)
            gate, up = gate_up.chunk(2, dim=-1)
            res = F.linear(F.silu(gate) * up, self.down_proj_weight)
            return res.view_as(x)

def apply_rotary_position_embedding(input, sin_table, cos_table, use_triton=False):
    if use_triton:
        batch_size, seq_len, n_heads, head_dim = input.shape
        output = torch.empty_like(input)

        BLOCK_SIZE = triton.next_power_of_2(head_dim * n_heads)
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        BLOCK_N_HEADS = triton.next_power_of_2(n_heads)
        BLOCK_D_HALF = triton.next_power_of_2(head_dim // 2)

        rotary_position_embedding_kernel[(batch_size, seq_len)](
            input, output, sin_table, cos_table,
            seq_len * n_heads * head_dim, n_heads * head_dim, head_dim,
            BLOCK_N_HEADS, BLOCK_D_HALF, num_warps=num_warps
        )
        return output
    else:
        sin_table = sin_table[None, :, None, :]
        cos_table = cos_table[None, :, None, :]
        input_0 = input[..., : input.shape[-1] // 2]
        input_1 = input[..., input.shape[-1] // 2 :]
        input_0_rotated = input_0 * cos_table - input_1 * sin_table
        input_1_rotated = input_0 * sin_table + input_1 * cos_table
        return torch.cat((input_0_rotated, input_1_rotated), dim=-1)

def apply_scaled_dot_product_attention(query, key, value, use_triton=False):
    if use_triton:
        batch, q_heads, M, d = query.shape
        _, k_heads, N, _ = key.shape
        output = torch.empty_like(query)

        block_m = 32
        block_n = 32
        block_d = triton.next_power_of_2(d)
        num_warps = min(max(block_m * block_d // 256, 1), 8)
        grid = (batch, q_heads, triton.cdiv(M, block_m))

        flash_attention_kernel[grid](
            Q_ptr=query, K_ptr=key, V_ptr=value, OUT_ptr=output,
            Q_HEAD_NUMS=q_heads, K_HEAD_NUMS=k_heads, M=M, N=N, d=d,
            Q_stride_B=query.stride(0), Q_stride_H=query.stride(1), Q_stride_M=query.stride(2), Q_stride_d=query.stride(3),
            K_stride_B=key.stride(0), K_stride_H=key.stride(1), K_stride_N=key.stride(2), K_stride_d=key.stride(3),
            V_stride_N=value.stride(2), V_stride_d=value.stride(3),
            OUT_stride_M=output.stride(2), OUT_stride_d=output.stride(3),
            BLOCKSIZE_M=block_m, BLOCKSIZE_N=block_n, BLOCKSIZE_d=block_d,
            num_warps=num_warps, num_stages=3 
        )
        return output
    else:
        _, num_heads_q, seq_len_q, emb_dim = query.shape
        _, num_heads_k, seq_len_k, _ = key.shape
        _, num_heads_v, _, _ = value.shape

        key = key.repeat_interleave(num_heads_q // num_heads_k, 1)
        value = value.repeat_interleave(num_heads_q // num_heads_v, 1)

        return F.scaled_dot_product_attention(query, key, value, is_causal=True)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, sin_table, cos_table, use_triton=False):
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape).permute(0, 2, 1, 3)

        query_states = apply_rotary_position_embedding(query_states, sin_table, cos_table, use_triton).permute(0, 2, 1, 3)
        key_states = apply_rotary_position_embedding(key_states, sin_table, cos_table, use_triton).permute(0, 2, 1, 3)

        attn_output = apply_scaled_dot_product_attention(query_states, key_states, value_states, use_triton)

        return self.o_proj(attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1))

def fused_add_rmsnorm(hidden_states, residual, weight, eps):
    orig_shape = hidden_states.shape
    x_2d = hidden_states.view(-1, orig_shape[-1])
    res_2d = residual.view(-1, orig_shape[-1])
    M, N = x_2d.shape
    out_norm = torch.empty_like(x_2d)
    grid = (M,)
    BLOCK_SIZE = triton.next_power_of_2(N)

    fused_add_rmsnorm_kernel[grid](
        x_2d, res_2d, weight, out_norm, x_2d,
        x_2d.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
    )
    return hidden_states, out_norm.view(*orig_shape)

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.rms_norm_eps = config.rms_norm_eps

    def forward(self, hidden_states, sin_table, cos_table, use_triton=False):
        attn_in = self.input_layernorm(hidden_states, use_triton)
        attn_out = self.self_attn(attn_in, sin_table, cos_table, use_triton)

        if use_triton:
            hidden_states, mlp_in = fused_add_rmsnorm(
                hidden_states, attn_out, self.post_attention_layernorm.weight, self.rms_norm_eps
            )
        else:
            hidden_states = hidden_states + attn_out
            mlp_in = self.post_attention_layernorm(hidden_states, use_triton)

        mlp_out = self.mlp(mlp_in, use_triton)
        hidden_states = hidden_states + mlp_out

        return hidden_states

@torch.compiler.disable
def generate_sin_and_cos_tables(seq_len, emb_dim, base, dtype, device):
    theta = base ** (-2 * (torch.arange(emb_dim // 2, dtype=torch.float32, device=device) / emb_dim))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    sin_table = torch.sin(positions * theta).to(dtype)
    cos_table = torch.cos(positions * theta).to(dtype)
    
    return sin_table, cos_table

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.torch_dtype = config.torch_dtype
        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(self.num_hidden_layers)])
        self.norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

    def forward(self, input_ids, use_triton=False):
        hidden_states = self.embed_tokens(input_ids)
        seq_len = hidden_states.shape[1]

        sin_table, cos_table = generate_sin_and_cos_tables(
            seq_len, self.head_dim, base=self.rope_theta,
            dtype=getattr(torch, self.torch_dtype), device=input_ids.device,
        )

        for i in range(self.num_hidden_layers):
            hidden_states = self.layers[i](hidden_states, sin_table, cos_table, use_triton)

        return self.norm(hidden_states, use_triton)

class ModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.is_compiled = False

    def compile(self):
        """让 torch.compile 介入，只编译原生的 PyTorch 算子"""
        if not self.is_compiled:
            print("⏳ 正在启动 torch.compile (首次编译可能需要 1-3 分钟)...")
        
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            
            self.model = torch.compile(self.model, dynamic=True)
            
            self.is_compiled = True
        return self

    def generate(self, input_ids, max_new_tokens=20, use_triton=False):
        _use_triton = use_triton if not self.is_compiled else False
        
        for _ in range(max_new_tokens):
            hidden_states = self.model(input_ids, use_triton=_use_triton)
            logits = self.lm_head(hidden_states[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
        return input_ids

    @staticmethod
    def from_pretrained(model_path):
        model_path = Path(model_path)
        with open(model_path / "config.json") as f:
            config = json.load(f)

        if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

        config = ModelConfig(**{k: v for k, v in config.items() if k in ModelConfig.__annotations__})
        model = ModelForCausalLM(config).to(getattr(torch, config.torch_dtype))
        state_dict = load_file(model_path / "model.safetensors")

        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        model.load_state_dict(state_dict)
        return model