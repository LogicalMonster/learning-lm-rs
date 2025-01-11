import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def detailed_inference(text, model, tokenizer, device):
    """
    详细展示模型推理的每个步骤
    """
    print("输入文本:", text)
    
    # 1. Tokenization
    # 使用tokenizer将文本转换为input_ids和attention_mask
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print("\n1. Tokenization 结果:")
    print(f"input_ids shape: {input_ids.shape}")
    print(f"tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
    
    # 2. Input Embedding
    # 获取模型的词嵌入层并计算嵌入
    inputs_embeds = model.get_input_embeddings()(input_ids)
    print(f"\n2. Input Embedding shape: {inputs_embeds.shape}")
    
    # 3. 创建因果注意力掩码（casual mask）
    seq_length = input_ids.size(1)
    casual_mask = torch.tril(torch.ones((seq_length, seq_length), device=device))
    casual_mask = casual_mask.view(1, 1, seq_length, seq_length)
    print(f"\n3. Casual Mask shape: {casual_mask.shape}")
    
    # 4. Decoder layers处理
    # 使用model.transformer.h或model.gpt_neox.layers获取decoder层
    hidden_states = inputs_embeds
    
    # 获取decoder层（不同模型可能有不同的属性名）
    decoder_blocks = (
        model.transformer.h if hasattr(model, 'transformer') 
        else model.gpt_neox.layers if hasattr(model, 'gpt_neox') 
        else None
    )
    
    if decoder_blocks is None:
        raise ValueError("Unsupported model architecture")
    
    print("\n4. Decoder Layers 处理:")
    for layer_idx, decoder_layer in enumerate(decoder_blocks):
        # 4.1 Pre-attention RMS Norm
        if hasattr(decoder_layer, 'ln_1'):
            hidden_states = decoder_layer.ln_1(hidden_states)
            print(f"\nLayer {layer_idx} - Pre-attention RMS Norm shape: {hidden_states.shape}")
        
        # 4.2 Self Attention
        # 获取attention输出
        attention_output = decoder_layer.attn(
            hidden_states,
            attention_mask=casual_mask,
            use_cache=False,
            output_attentions=True
        )
        attn_output = attention_output[0]
        attn_weights = attention_output[1]
        
        print(f"Layer {layer_idx} - Attention output shape: {attn_output.shape}")
        print(f"Layer {layer_idx} - Attention weights shape: {attn_weights.shape}")
        
        # 残差连接
        hidden_states = attn_output + hidden_states
        
        # 4.3 Post-attention RMS Norm
        if hasattr(decoder_layer, 'ln_2'):
            hidden_states = decoder_layer.ln_2(hidden_states)
            print(f"Layer {layer_idx} - Post-attention RMS Norm shape: {hidden_states.shape}")
        
        # 4.4 MLP
        mlp_output = decoder_layer.mlp(hidden_states)
        print(f"Layer {layer_idx} - MLP output shape: {mlp_output.shape}")
        
        # 残差连接
        hidden_states = mlp_output + hidden_states
    
    # 5. Final Layer Norm
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        hidden_states = model.transformer.ln_f(hidden_states)
    print(f"\n5. Final Layer Norm output shape: {hidden_states.shape}")
    
    # 6. Language Model Head
    lm_logits = model.lm_head(hidden_states)
    print(f"\n6. Language Model Head output shape: {lm_logits.shape}")
    
    # 7. 获取下一个token的预测
    next_token_logits = lm_logits[:, -1, :]
    
    # 应用采样参数
    temperature = 0.6
    scaled_logits = next_token_logits / temperature
    
    # Top-k sampling
    top_k = 40
    top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k)
    probs = torch.softmax(top_k_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    next_token = top_k_indices.gather(-1, next_token)
    
    print(f"\n7. 生成的下一个token: {tokenizer.decode(next_token[0])}")
    
    return next_token

# 文本生成函数
def generate_with_details(
    prompt,
    model,
    tokenizer,
    device,
    max_new_tokens=50,
    temperature=0.6,
    top_k=40,
    top_p=0.9
):
    """
    带有详细推理过程的文本生成函数
    """
    print("开始生成文本...")
    print(f"初始提示词: {prompt}\n")
    
    # 初始化input_ids
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # 逐个生成token
    for i in range(max_new_tokens):
        next_token = detailed_inference(
            tokenizer.decode(input_ids[0]),
            model,
            tokenizer,
            device
        )
        
        # 将新token添加到输入序列
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        # 检查是否生成了结束符号
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # 解码完整序列
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"\n最终生成的文本: {generated_text}")
    
    return generated_text

# 使用示例
"""
# 设置模型和设备
model_path = "raincandy-u/TinyStories-656K"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# 生成文本
prompt = "<|start_story|>Once upon a time, "
generated_text = generate_with_details(
    prompt,
    model,
    tokenizer,
    device,
    max_new_tokens=50
)
"""