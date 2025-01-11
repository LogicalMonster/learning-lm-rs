import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaModel

# 指定模型目录
# model_path = "raincandy-u/TinyStories-656K"
model_path = "models/story"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# print('tokenizer:', tokenizer)

# 加载模型并移动到 GPU（如果可用）
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = LlamaModel.from_pretrained(model_path, attn_implementation="eager")
# .to(device)
model.eval()

print('model:', model)

# 测试模型和分词器
text = "<|start_story|>Once upon a time, "

inputs = tokenizer(text, return_tensors="pt")
# .to(device)

print('inputs:', inputs)

input_ids = inputs["input_ids"]

# print('input_ids:', input_ids)

with torch.no_grad():
    outputs = model(input_ids, output_attentions=True)

attention = outputs.attentions[1]

# 打印关键形状和值
print(f"Attention shape: {attention.shape}")
print("\nAttention weights sample:")
# print(attention[0, 0, :, :])  # 打印第一个头的attention矩阵
# print(attention)

q = model.layers[0].self_attn.q_proj(outputs.last_hidden_state)
q = q.view(1, 6, 4, 2, -1)
print(f"\nQ tensor shape: {q.shape}")

v = model.layers[0].self_attn.v_proj(outputs.last_hidden_state)
v = v.view(1, 6, 4, 1, -1)
print(f"\nV tensor shape: {v.shape}")

# outputs = model.generate(
#     **inputs,
#     pad_token_id=tokenizer.eos_token_id,
#     max_new_tokens=256,
#     do_sample=True,
#     top_k=40,
#     top_p=0.9,
#     temperature=0.6,
#     # output_attentions=True
# )

# print('outputs.shape:', outputs.shape)

# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print('generated_text:', generated_text)

# 请将上述模型的推理步骤细化，给出从text经过tokenizer.encode、input embedding、Decoder(RMS Norm、masked multi-head self-attention、RMS Norm、MLP)、RMS Norm、Output embedding、tokenizer.decode的完整计算过程


