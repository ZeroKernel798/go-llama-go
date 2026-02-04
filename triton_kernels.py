import triton
import triton.language as tl

# 该文件是各个算子的triton实现和具体思路分析
# RMSNorm
@triton.jit
def rmsnorm_kernel(
    input_ptr, gamma_ptr, output_ptr,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr
):
    # 一个线程块负责一行 一行有N列
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride
    out_start_ptr = output_ptr + row_idx * stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # 显示转换为fp32
    x = tl.load(row_start_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # fp32下计算平方和、均值
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    
    # 计算归一化系数
    inv_rms = tl.rsqrt(sum_sq / N + eps)
    
    # 权重转fp32
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # fp32下完成缩放
    output_data = x * inv_rms * gamma
    
    # 结果变成bf16输出
    tl.store(out_start_ptr + cols, output_data.to(tl.bfloat16), mask=mask)


# triton和add的融合算子 能减少显存读写
@triton.jit
def fused_add_rmsnorm_kernel(
    input_ptr,       # 原始 hidden_states [M, N]
    residual_ptr,    # 刚才算出来的 attn_out 或 mlp_out [M, N]
    gamma_ptr,       # RMSNorm 权重 [N]
    out_norm_ptr,    # 算完 Norm 的结果 [M, N]
    out_add_ptr,     # 加完后的新 hidden_states [M, N] (用于下一次残差)
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr
):
    # 定位行
    row_idx = tl.program_id(0)
    offset = row_idx * stride + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N

    # 同时加载输入和残差，直接在寄存器里加
    # 这步省去了原生的 hidden_states += attn_out 独立 Kernel
    x = tl.load(input_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    
    x_sum = x + res

    # 把加完后的结果写回
    tl.store(out_add_ptr + offset, x_sum.to(tl.bfloat16), mask=mask)

    # 直接复用之前的 RMSNorm 逻辑，拿寄存器里的 x_sum 算
    x_sq = x_sum * x_sum
    sum_sq = tl.sum(x_sq, axis=0)
    inv_rms = tl.rsqrt(sum_sq / N + eps)
    
    gamma = tl.load(gamma_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)
    output_data = x_sum * inv_rms * gamma
    
    # 写回 Norm 后的结果
    tl.store(out_norm_ptr + offset, output_data.to(tl.bfloat16), mask=mask)


# 这是rope的triton实现
@triton.jit
def rotary_position_embedding_kernel(
    input_ptr, output_ptr, sin_table, cos_table,
    stride_batch, stride_seq, stride_head,
    N_HEADS: tl.constexpr, 
    D_HALF: tl.constexpr):
    # 依旧是一个block处理一个seq位置的所有head
    # 为了提高性能 减少切片和cat这些操作

    # 定位 Token 
    bid = tl.program_id(0)
    sid = tl.program_id(1)
    
    # 计算 2D 索引 (Heads x Half_Dim) 注意必须为编译期常量
    head_idx = tl.arange(0, N_HEADS) 
    dim_idx = tl.arange(0, D_HALF)  # 对应 D_HALF
    
    # 计算 x (前半截) 和 y (后半截) 的物理偏移
    row_base = input_ptr + bid * stride_batch + sid * stride_seq
    
    # offs_x 是每个头前 32 位的地址，offs_y 是后 32 位的地址
    offs_x = head_idx[:, None] * stride_head + dim_idx[None, :]
    offs_y = offs_x + D_HALF
    
    # 计算掩码
    mask = (head_idx[:, None] < N_HEADS) & (dim_idx[None, :] < D_HALF)
    
    # 获取数据 (直接加载 x 和 y，跳过切片)
    x = tl.load(row_base + offs_x, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(row_base + offs_y, mask=mask, other=0.0).to(tl.float32)

    # 获取 sin/cos 并广播 
    s = tl.load(sin_table + sid * D_HALF + dim_idx, mask=dim_idx < D_HALF).to(tl.float32)
    c = tl.load(cos_table + sid * D_HALF + dim_idx, mask=dim_idx < D_HALF).to(tl.float32)
    
    # 把 [32] 变成 [1, 32]
    s = s[None, :]
    c = c[None, :]

    # x_new = x * cos - y * sin
    # y_new = x * sin + y * cos
    x_new = x * c - y * s
    y_new = x * s + y * c

    # 写回数据 (分两次写，避开 tl.cat)
    out_base = output_ptr + bid * stride_batch + sid * stride_seq
    tl.store(out_base + offs_x, x_new.to(tl.bfloat16), mask=mask)
    tl.store(out_base + offs_y, y_new.to(tl.bfloat16), mask=mask)


# softmax attention 的triton实现 采用flash attention思路
@triton.jit
def flash_attention_kernel(Q_ptr, K_ptr, V_ptr, OUT_ptr,
                      Q_HEAD_NUMS, K_HEAD_NUMS, M, N, d,
                      Q_stride_B, Q_stride_H, Q_stride_M, Q_stride_d,
                      K_stride_B, K_stride_H, K_stride_N, K_stride_d,
                      V_stride_N, V_stride_d,
                      OUT_stride_M, OUT_stride_d,
                      BLOCKSIZE_M: tl.constexpr,
                      BLOCKSIZE_N: tl.constexpr,
                      BLOCKSIZE_d: tl.constexpr):
    
    # 注意维度的分布是batch head seq dim 也是 batch head m/n dim
    # 采用三维网格形式 最外层对应batch 第二层对应head 第三层对应seq 
    # 线程块采用二维线程块 
    pid0 = tl.program_id(0) # batch
    pid1 = tl.program_id(1) # heaad
    pid2 = tl.program_id(2) # seq 

    # 当前head所在的起始位置
    offset_base = pid0 * Q_stride_B + pid1 * Q_stride_H
    # 这个是Q矩阵M维度的偏移
    offset_M = pid2 * BLOCKSIZE_M + tl.arange(0, BLOCKSIZE_M)
    # 这是大家D维度的偏移 D维度是个常数 测试模型为64
    offset_d = tl.arange(0, BLOCKSIZE_d)
    # N维度需要从头滑动　所以不给初始值
    offset_N = tl.arange(0, BLOCKSIZE_N)

    # 当前块 只负责这个Q_offset 
    Q_offset = offset_M[:, None] * Q_stride_M + offset_d[None, :] * Q_stride_d
    Q_mask = (offset_M[:, None] < M) & (offset_d[None, :] < d)
    Q_data = tl.load(Q_ptr + Q_offset + offset_base, mask=Q_mask)

    # 分块矩阵应该是blk_m * d   d * blk_n  blk_n * d最后得到的是blk_m * d
    accumulator = tl.zeros((BLOCKSIZE_M, BLOCKSIZE_d), dtype=tl.float32)
    # 这个是每一行分母指数和 每一行一个 恰好跟列的维度对应
    softmax_running_sum = tl.zeros([BLOCKSIZE_M], dtype=tl.float32)
    # 这个是每一行的局部最大值 会动态更新 每一行一个 恰好跟列维度对应
    softmax_current_max = tl.full([BLOCKSIZE_M], float("-inf"), dtype=tl.float32)
    # 这个是d的平方根 用于缩放
    rs_attention_logits_scale = tl.rsqrt(d + 0.0)

    # 对kv矩阵进行滑动分块处理
    # 要先计算一下 Q对应的是哪个KV的头
    HEAD_IDX = pid1 // (Q_HEAD_NUMS // K_HEAD_NUMS)
    for current_index in range(0, N, BLOCKSIZE_N):
        current_base_offset = pid0 * K_stride_B + HEAD_IDX * K_stride_H
        current_k_offset = current_index + offset_N
        current_v_offset = current_k_offset

        K_offset = current_k_offset[:, None] * K_stride_N + offset_d[None, :] * K_stride_d
        V_offset = current_v_offset[:, None] * V_stride_N + offset_d[None, :] * V_stride_d

        K_mask = (current_k_offset[:, None] < N) & (offset_d[None, :] < d)
        V_mask = (current_v_offset[:, None] < N) & (offset_d[None, :] < d)

        K_data = tl.load(K_ptr + K_offset + current_base_offset, mask=K_mask)
        V_data = tl.load(V_ptr + V_offset + current_base_offset, mask=V_mask)

        # 为了利用tensor core
        attention_logits = tl.dot(Q_data, tl.trans(K_data)) * rs_attention_logits_scale

        # 因果掩码逻辑
        causal_mask = offset_M[:, None] >= current_k_offset[None, :]
        boundary_mask = (offset_M[:, None] < M) & (current_k_offset[None, :] < N)
        attention_logits = tl.where(causal_mask & boundary_mask, attention_logits, float("-inf"))

        # 找每一列的最大值
        current_block_max = tl.max(attention_logits, axis=-1)
        # 跟之前的最大值比较 然后更新
        max_value = tl.maximum(current_block_max, softmax_current_max)
        # 修正偏差
        alpha = tl.exp(softmax_current_max - max_value)
        softmax_current_max = max_value

        # 做安全softmax
        attention_logits_shift = attention_logits - max_value[:, None]

        # 当前这组值 
        softmax_nom = tl.exp(attention_logits_shift)
        # 当前的局部分母指数和
        softmax_denom = tl.sum(softmax_nom, axis=1)
        # 补差价再加上末尾的值
        softmax_running_sum = tl.fma(softmax_running_sum, alpha, softmax_denom)
        # 同上
        accumulator = tl.fma(accumulator, alpha[:, None], tl.dot(softmax_nom.to(V_data.dtype), V_data))

    accumulator /= softmax_running_sum[:, None]
    
    # 写回结果
    OUT_offset = offset_M[:, None] * OUT_stride_M + offset_d[None, :] * OUT_stride_d
    OUT_mask = (offset_M[:, None] < M) & (offset_d[None, :] < d)

    tl.store(OUT_ptr + OUT_offset + offset_base, accumulator.to(tl.float16), mask=OUT_mask)


# mlp的triton实现。通过矩阵拼接等操作 减少全局显存的读取 提高效率
# 加入配置调优的triton核函数
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=5, num_warps=4),
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
#         triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
#     ],
#     key=['M', 'N', 'K'],
# )
@triton.jit
def mlp_kernel_step1(
    input_ptr, gate_ptr, up_ptr, output_ptr, 
    M, N, K,
    stride_inm, stride_ink, 
    stride_gk, stride_gn, 
    stride_outm, stride_outn, 
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # 这是个访存密集型算子 可以通过矩阵拼接、合并计算过程来减少显存操作
    # 将gate和up权重进行拼接 然后一次读取和计算
    # 为了习惯 将input矩阵维度定义为mk 其他矩阵维度为kn 结果矩阵维度为mn

    # 先找到行列的数据块索引
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # 设置访问逻辑偏移量
    m_offset = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = pid1 * BLOCK_N + tl.arange(0, BLOCK_N) # gate
    u_offset = n_offset + N
    
    # 下面是分块操作
    # 先定义每一轮分块操作的结果
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offset = k + tl.arange(0, BLOCK_K)

        # 地址偏移量
        in_offset = m_offset[:, None] * stride_inm + k_offset[None, :] * stride_ink
        gate_offset = k_offset[:, None] * stride_gk + n_offset[None, :] * stride_gn
        up_offset = k_offset[:, None] * stride_gk + u_offset[None, :] * stride_gn

        # 设置掩码
        in_mask = (m_offset[:, None] < M) & (k_offset[None, :] < K)
        gate_mask = (k_offset[:, None] < K) & (n_offset[None, :] < N)
        up_mask = (k_offset[:, None] < K) & (n_offset[None, :] < N)

        # data获取
        in_data = tl.load(input_ptr + in_offset, mask=in_mask).to(tl.float32)
        gate_data = tl.load(gate_ptr + gate_offset, mask=gate_mask).to(tl.float32)
        up_data = tl.load(up_ptr + up_offset, mask=up_mask).to(tl.float32)


        # 计算局部分块的结果
        acc_gate += tl.dot(in_data, gate_data)
        acc_up += tl.dot(in_data, up_data)
    
    # 循环结束 表示gate up 两种权重下的输出矩阵里面的block_m block_n大小的块计算完毕了
    # 求silu 以及 做逐元素乘法
    activated_gate = acc_gate * tl.sigmoid(acc_gate)
    final_result = activated_gate * acc_up

    # 最后存回显存
    out_offset = m_offset[:, None] * stride_outm + n_offset[None, :] * stride_outn
    out_mask = (m_offset < M)[:, None] & (n_offset < N)[None, :]
    tl.store(output_ptr + out_offset, final_result, mask=out_mask)

@triton.jit
def mlp_kernel_step2(
    x_ptr,           # 第一步的结果 [M, K]
    weight_ptr,      # Down 权重 [K, N]
    output_ptr,      # 最终输出 [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr  # 把 K 轴切成几份并行
):
    # 获取三个维度的 Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)  # 这里就是 Split-K 的分片索引

    # 计算当前分片负责的 K 范围
    k_per_split = tl.cdiv(K, SPLIT_K)
    start_k = pid_k * k_per_split
    end_k = tl.minimum(start_k + k_per_split, K)

    # 设置行列偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 初始化局部累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 在自己负责的 K 范围内进行分块累加
    for k in range(start_k, end_k, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # 掩码处理
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # 加载数据
        x = tl.load(x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk), mask=mask_x, other=0.0)
        w = tl.load(weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn), mask=mask_w, other=0.0)

        # 矩阵乘加
        acc += tl.dot(x, w)

    # 原子加法 
    # 因为不同的 pid_k 都在算同一个 (pid_m, pid_n) 的输出块
    out_offset = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # 必须用原子加，把局部结果合并到全局显存
    tl.atomic_add(output_ptr + out_offset, acc, mask=out_mask)
