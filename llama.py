import dataclasses
import json
import math
from pathlib import Path

import triton
import torch
import torch.nn as nn
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


# class RMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps):
#         super().__init__()

#         self.weight = nn.Parameter(torch.ones(hidden_size))

#         self.eps = eps

#     def forward(self, input):
#         return (
#             input
#             * torch.rsqrt(input.pow(2).mean(dim=-1, keepdim=True) + self.eps)
#             * self.weight
#         )


# triton实现版本
class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # 张量压缩为二维
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        M, N = x_2d.shape
        
        # 设置输出
        out = torch.empty_like(x_2d)
        
        # 配置kernel参数
        BLOCK_SIZE = triton.next_power_of_2(N)
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        
        # 启动参数
        rmsnorm_kernel[(M,)](
            x_2d,          # input_ptr
            self.weight,   # gamma_ptr
            out,           # output_ptr
            x_2d.stride(0),# stride_row 
            N,             # N cols
            self.eps,      # eps
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        
        # 还原形状并输出
        return out.view(*orig_shape)


# class MLP(nn.Module):
#     def __init__(self, hidden_size, intermediate_size):
#         super().__init__()

#         self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

#         self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

#         self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

#         self.silu = nn.SiLU()

#     def forward(self, input):
#         return self.down_proj(self.silu(self.gate_proj(input)) * self.up_proj(input))

# triton实现
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # 定义融合后的权重容器
        # gate_up_weight 存储 [2 * intermediate_size, hidden_size]
        self.gate_up_weight = nn.Parameter(torch.empty((2 * intermediate_size, hidden_size)))
        # down_proj_weight 存储 [hidden_size, intermediate_size]
        self.down_proj_weight = nn.Parameter(torch.empty((hidden_size, intermediate_size)))

    # 自动拦截并转换官方权重
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        gate_key = prefix + "gate_proj.weight"
        up_key = prefix + "up_proj.weight"
        down_key = prefix + "down_proj.weight"

        # 处理 Gate 和 Up 的合并
        if gate_key in state_dict and up_key in state_dict:
            gate_w = state_dict.pop(gate_key)
            up_w = state_dict.pop(up_key)
            # 垂直拼接
            combined_w = torch.cat([gate_w, up_w], dim=0)
            state_dict[prefix + "gate_up_weight"] = combined_w
            
        # 处理 Down Proj 的改名
        if down_key in state_dict:
            state_dict[prefix + "down_proj_weight"] = state_dict.pop(down_key)

        # 调用原生的加载逻辑
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # 兼容 3D 输入 [Batch, Seq, Dim] -> [M, K]
        orig_shape = x.shape
        x_2d = x.view(-1, self.hidden_size)
        M, K = x_2d.shape
        N = self.intermediate_size

        # 融合 Gate/Up & SiLU 
        # 输出为 [M, N]
        step1_out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid1 = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
        
        # 固定triton核函数的配置
        mlp_kernel_step1[grid1](
            x_2d, self.gate_up_weight, self.gate_up_weight, step1_out,
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            self.gate_up_weight.stride(1), self.gate_up_weight.stride(0), # 逻辑转置
            step1_out.stride(0), step1_out.stride(1),
            BLOCK_M=32, BLOCK_N=64, BLOCK_K=32,
            num_warps = 4,
            num_stages = 2
        )
        # 自动调优的配置
        # mlp_kernel_step1[grid1](
        #     x_2d, self.gate_up_weight, self.gate_up_weight, step1_out,
        #     M, N, K,
        #     x_2d.stride(0), x_2d.stride(1),
        #     self.gate_up_weight.stride(1), self.gate_up_weight.stride(0), # 逻辑转置
        #     step1_out.stride(0), step1_out.stride(1)
        # )
        # print(f"MLP Step 1 Best Config: {mlp_kernel_step1.best_config}")
        # Down Projection (Split-K 并行优化) 
        # 输出为 [M, K]
        final_out = torch.zeros((M, K), device=x.device, dtype=x.dtype)
        
        # 针对不同 M 动态设置 Split-K，提升 Decode 阶段（M=1）的并行度
        split_k = 4 if M <= 16 else 1 
        grid2 = (triton.cdiv(M, 64), triton.cdiv(K, 64), split_k)

        # 第二个核函数
        mlp_kernel_step2[grid2](
            step1_out, self.down_proj_weight, final_out,
            M, K, N, 
            step1_out.stride(0), step1_out.stride(1),
            self.down_proj_weight.stride(1), self.down_proj_weight.stride(0),
            final_out.stride(0), final_out.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            SPLIT_K=split_k
        )

        return final_out.view(*orig_shape)

# 这个是rope
# def apply_rotary_position_embedding(input, sin_table, cos_table):
#     sin_table = sin_table[None, :, None, :]
#     cos_table = cos_table[None, :, None, :]

#     input_0 = input[..., : input.shape[-1] // 2]
#     input_1 = input[..., input.shape[-1] // 2 :]
#     input_0_rotated = input_0 * cos_table - input_1 * sin_table
#     input_1_rotated = input_0 * sin_table + input_1 * cos_table

#     return torch.cat((input_0_rotated, input_1_rotated), dim=-1)

# 核函数重构版本
def apply_rotary_position_embedding(input, sin_table, cos_table):
    batch_size, seq_len, n_heads, head_dim = input.shape

    output = torch.empty_like(input)

    # 配置kernel参数
    BLOCK_SIZE = triton.next_power_of_2(head_dim * n_heads)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    BLOCK_N_HEADS = triton.next_power_of_2(n_heads)
    BLOCK_D_HALF = triton.next_power_of_2(head_dim // 2)

    # 核函数启动
    rotary_position_embedding_kernel[(batch_size, seq_len)](
        input,                     # input_ptr
        output,                    # output_ptr
        sin_table,                 # cos_table
        cos_table,                 # sin_table
        seq_len * n_heads * head_dim,  # stride_batch
        n_heads * head_dim,            # stride_seq
        head_dim,                      # stride_head
        BLOCK_N_HEADS,
        BLOCK_D_HALF,
        num_warps=num_warps
    )

    return output
    

# gqa
# def apply_scaled_dot_product_attention(query, key, value):
#     _, num_heads_q, seq_len_q, emb_dim = query.shape
#     _, num_heads_k, seq_len_k, _ = key.shape
#     _, num_heads_v, _, _ = value.shape

#     key = key.repeat_interleave(num_heads_q // num_heads_k, 1)
#     value = value.repeat_interleave(num_heads_q // num_heads_v, 1)

#     scale = 1 / math.sqrt(emb_dim)
#     attn_mask = torch.tril(
#         torch.full((seq_len_q, seq_len_k), True, device=query.device)
#     )

#     attn_output = torch.matmul(query, key.permute(0, 1, 3, 2)) * scale
#     attn_output = torch.where(attn_mask, attn_output, float("-inf"))
#     attn_output = torch.softmax(attn_output, dim=-1)
#     attn_output = torch.matmul(attn_output, value)

#     return attn_output

# triton实现的softmax attention
def apply_scaled_dot_product_attention(query, key, value):
    # 确保输入是连续的，并获取形状
    batch, q_heads, M, d = query.shape
    _, k_heads, N, _ = key.shape
    
    # 校验 head_dim 是否匹配
    assert d == query.shape[-1] == key.shape[-1] == value.shape[-1]
    
    # 预先分配输出矩阵
    output = torch.empty_like(query)

    # 计算 3D Grid
    # grid[0]: Batch 维度
    # grid[1]: Head 维度 (Q 的头数)
    # grid[2]: Sequence 维度的分块 (M 方向)
    block_m = 32
    block_n = 32
    block_d = triton.next_power_of_2(d) # 确保是2的幂次以优化对齐
    num_warps = min(max(block_m * block_d // 256, 1), 8)
    grid = (batch, q_heads, triton.cdiv(M, block_m))

    # 调用核函数
    flash_attention_kernel[grid](
        Q_ptr=query, K_ptr=key, V_ptr=value, OUT_ptr=output,
        Q_HEAD_NUMS=q_heads, K_HEAD_NUMS=k_heads,
        M=M, N=N, d=d,
        # 传入步长 (Strides)
        Q_stride_B=query.stride(0), Q_stride_H=query.stride(1), Q_stride_M=query.stride(2), Q_stride_d=query.stride(3),
        K_stride_B=key.stride(0), K_stride_H=key.stride(1), K_stride_N=key.stride(2),   K_stride_d=key.stride(3),
        V_stride_N=value.stride(2), V_stride_d=value.stride(3),
        OUT_stride_M=output.stride(2), OUT_stride_d=output.stride(3),
        BLOCKSIZE_M=block_m,
        BLOCKSIZE_N=block_n,
        BLOCKSIZE_d=block_d,
        num_warps=num_warps,
        num_stages=3 
    )
    return output


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )

        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, hidden_states, sin_table, cos_table):
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape).permute(0, 2, 1, 3)

        query_states = apply_rotary_position_embedding(
            query_states, sin_table, cos_table
        ).permute(0, 2, 1, 3)
        key_states = apply_rotary_position_embedding(
            key_states, sin_table, cos_table
        ).permute(0, 2, 1, 3)

        attn_output = apply_scaled_dot_product_attention(
            query_states, key_states, value_states
        )

        return self.o_proj(
            attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        )

# 单个计算单元
# class DecoderLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

#         self.self_attn = Attention(config)

#         self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

#         self.mlp = MLP(config.hidden_size, config.intermediate_size)

#     def forward(self, hidden_states, sin_table, cos_table):
#         hidden_states += self.self_attn(
#             self.input_layernorm(hidden_states), sin_table, cos_table
#         )

#         hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))

#         return hidden_states

# triton算子融合优化
# 提供一个方法实现triton包装
def fused_add_rmsnorm(hidden_states, residual, weight, eps):
    # 将 [Batch, Seq, Hidden] 转为 [M, N]
    orig_shape = hidden_states.shape
    x_2d = hidden_states.view(-1, orig_shape[-1])
    res_2d = residual.view(-1, orig_shape[-1])
    M, N = x_2d.shape
    
    # 准备输出：Norm 后的结果
    out_norm = torch.empty_like(x_2d)
    
    # 设置 Grid：一行一个 Program
    grid = (M,)
    
    # 设置分块：对齐到 2 的幂次
    BLOCK_SIZE = triton.next_power_of_2(N)

    # 把 x_2d 同时也传给 out_add_ptr，实现原地更新 hidden_states
    fused_add_rmsnorm_kernel[grid](
        x_2d, res_2d, weight, out_norm, x_2d,
        x_2d.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8
    )
    
    # 返回原地更新后的 hidden_states 和给 MLP 用的输入
    return hidden_states, out_norm.view(*orig_shape)

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.rms_norm_eps = config.rms_norm_eps

    def forward(self, hidden_states, sin_table, cos_table):
        # 照常算第一个 Norm 和 Attention
        attn_in = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(attn_in, sin_table, cos_table)

        # 把上一动的 += 和下一动的 Norm 融合
        # 代替了原先的:
        # hidden_states += attn_out
        # mlp_in = self.post_attention_layernorm(hidden_states)
        hidden_states, mlp_in = fused_add_rmsnorm(
            hidden_states, 
            attn_out, 
            self.post_attention_layernorm.weight, 
            self.rms_norm_eps
        )

        # 算 MLP
        mlp_out = self.mlp(mlp_in)

        # 最后一次残差加法
        hidden_states += mlp_out

        return hidden_states

def generate_sin_and_cos_tables(seq_len, emb_dim, base, dtype, device):
    theta = base ** (
        -2 * (torch.arange(emb_dim // 2, dtype=dtype, device=device) / emb_dim)
    )

    positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    sin_table = torch.sin(positions * theta)
    cos_table = torch.cos(positions * theta)

    return sin_table, cos_table

# 模型主干
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

        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(self.num_hidden_layers)
        )

        self.norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        seq_len = hidden_states.shape[1]

        sin_table, cos_table = generate_sin_and_cos_tables(
            seq_len,
            self.head_dim,
            base=self.rope_theta,
            dtype=getattr(torch, self.torch_dtype),
            device=input_ids.device,
        )

        for i in range(self.num_hidden_layers):
            hidden_states = self.layers[i](hidden_states, sin_table, cos_table)

        return self.norm(hidden_states)

# 最上层
class ModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def generate(self, input_ids, max_new_tokens=20):
        for _ in range(max_new_tokens):
            hidden_states = self.model(input_ids)

            logits = self.lm_head(hidden_states[:, -1, :])

            next = torch.argmax(logits, dim=-1).unsqueeze(-1)

            input_ids = torch.cat((input_ids, next), dim=-1)

        return input_ids

    @staticmethod
    def from_pretrained(model_path):
        model_path = Path(model_path)

        with open(model_path / "config.json") as f:
            config = json.load(f)

        if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

        config = ModelConfig(
            **{
                key: value
                for key, value in config.items()
                if key in ModelConfig.__annotations__
            }
        )

        model = ModelForCausalLM(config).to(getattr(torch, config.torch_dtype))

        state_dict = load_file(model_path / "model.safetensors")

        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        model.load_state_dict(state_dict)

        return model
