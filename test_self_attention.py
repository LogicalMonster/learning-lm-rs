import torch
import torch.nn.functional as F
from torch import Tensor

def self_attention(
    q: Tensor, 
    full_k: Tensor, 
    full_v: Tensor, 
    n_kv_h: int, 
    n_groups: int, 
    seq_len: int, 
    total_seq_len: int, 
    dqkv: int
) -> Tensor:
    """
    带有因果遮罩的自注意力机制。

    参数:
        q (Tensor): 查询张量，形状为 [seq_len, n_kv_h * n_groups * dqkv]。
        full_k (Tensor): 键张量，形状为 [total_seq_len, n_kv_h * dqkv]。
        full_v (Tensor): 值张量，形状为 [total_seq_len, n_kv_h * dqkv]。
        n_kv_h (int): 键/值头的数量。
        n_groups (int): 每个键/值头的查询组数。
        seq_len (int): 输入的序列长度。
        total_seq_len (int): 总序列长度（包括上下文）。
        dqkv (int): 每个键、值和查询头的维度。

    返回:
        Tensor: 输出张量，形状为 [seq_len, n_kv_h * n_groups * dqkv]。
    """
    # 重塑查询、键和值张量
    q.transpose_(0, 1)
    q = q.view(n_kv_h, n_groups, dqkv, seq_len)
    q.transpose_(-2, -1)
    print('q:', q)

    full_k.transpose_(0, 1)
    full_k = full_k.view(n_kv_h, 1, dqkv, total_seq_len)
    print('full_k:', full_k)

    # 使用缩放点积计算注意力分数
    att_scores = torch.matmul(q, full_k) / (dqkv ** 0.5)
    print('att_scores:', att_scores)

    # 应用因果遮罩
    causal_mask = torch.triu(torch.ones(seq_len, total_seq_len, dtype=torch.bool), diagonal=seq_len + 1)
    att_scores = att_scores.masked_fill(causal_mask, float('-inf'))

    # 对注意力分数应用softmax以获得注意力概率
    att_probs = F.softmax(att_scores, dim=-1)
    print('att_probs:', att_probs)

    full_v.transpose_(0, 1)
    full_v = full_v.view(n_kv_h, 1, dqkv, total_seq_len)
    full_v.transpose_(-2, -1)
    print('full_v:', full_v)
    # 计算注意力输出
    # n_kv_h, n_groups, seq_len, dqkv
    att_output = torch.matmul(att_probs, full_v)
    print('att_output:', att_output)

    # 将输出重塑为原始隐藏状态的形状
    att_output.transpose_(-2, -1)
    att_output = att_output.reshape(n_kv_h * n_groups * dqkv, seq_len)
    hidden_states = att_output.transpose(0, 1)
    print('hidden_states:', hidden_states)

    return hidden_states

# 测试代码
if __name__ == "__main__":
    n_q_h = 2
    n_kv_h = 1  # 键/值头的数量
    n_groups = n_q_h // n_kv_h  # 每个键/值头的查询组数
    seq_len = 2  # 序列长度
    total_seq_len = 4  # 总序列长度（包括上下文）
    hidden_size = 4
    dqkv = hidden_size // n_q_h  # 每个头的维度

    # 创建测试输入
    # q = torch.randn([seq_len, n_kv_h * n_groups * dqkv], dtype=torch.float32)
    q = torch.arange((seq_len * n_kv_h * n_groups * dqkv), dtype=torch.float32)
    q = (q + 1) / 10
    q = q.reshape(seq_len, n_kv_h * n_groups * dqkv)
    # full_k = torch.randn([total_seq_len, n_kv_h * dqkv], dtype=torch.float32)
    full_k = torch.arange(total_seq_len * n_kv_h * dqkv, dtype=torch.float32)
    full_k = (full_k + 1) / 10
    full_k = full_k.reshape(total_seq_len, n_kv_h * dqkv)
    # full_v = torch.randn([total_seq_len, n_kv_h * dqkv], dtype=torch.float32)
    full_v = torch.arange(total_seq_len * n_kv_h * dqkv, dtype=torch.float32)
    full_v = (full_v + 1) / 10
    full_v = full_v.reshape(total_seq_len, n_kv_h * dqkv)

    print(q)
    print(full_k)
    print(full_v)

    # 运行函数
    output = self_attention(
        q, full_k, full_v,
        n_kv_h, n_groups, seq_len, total_seq_len, dqkv
    )

    # 验证输出形状
    assert output.shape == (seq_len, n_kv_h * n_groups * dqkv)
    print("输出张量形状:", output.shape)
    print(output)
