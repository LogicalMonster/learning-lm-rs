import torch
from transformers import LlamaConfig, LlamaModel, AutoConfig
import numpy as np
import os

def debug_attention():
    # 从实际的config.json加载配置
    model_dir = "models/story"
    config = AutoConfig.from_pretrained(model_dir)
    
    # 从safetensors加载模型
    model = LlamaModel.from_pretrained(model_dir, config=config, attn_implementation="eager")
    model.eval()
    
    # 创建示例输入
    batch_size = 1
    seq_len = 4
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # 运行模型并获取中间结果
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        
    # 提取第一层的attention权重
    attention = outputs.attentions[0]  # shape: [batch, num_heads, seq_len, seq_len]
    
    # 打印关键形状和值
    print(f"Attention shape: {attention.shape}")
    print("\nAttention weights sample:")
    print(attention[0, 0, :, :])  # 打印第一个头的attention矩阵
    
    # 获取q,k,v值用于对比
    qkv = model.layers[0].self_attn.q_proj(outputs.last_hidden_state)
    q = qkv.view(batch_size, seq_len, config.num_attention_heads, -1)
    print(f"\nQ tensor shape: {q.shape}")
    print("Q values sample:")
    print(q[0, 0, 0, :10])  # 打印第一个位置,第一个头的前10个值

def print_attention_math_example():
    """打印一个简单的attention计算示例"""
    q = torch.randn(1, 4, 2)  # [batch, seq_len, head_dim]
    k = torch.randn(1, 4, 2)  # [batch, seq_len, head_dim]
    
    # 计算attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(2)
    print("\nAttention score calculation example:")
    print(f"Q shape: {q.shape}")
    print(f"K shape: {k.shape}")
    print(f"Scores shape: {scores.shape}")
    print("\nScores:")
    print(scores[0])
    
    # 应用softmax
    attn_weights = torch.softmax(scores, dim=-1)
    print("\nAfter softmax:")
    print(attn_weights[0])

if __name__ == "__main__":
    debug_attention()
    print_attention_math_example() 
