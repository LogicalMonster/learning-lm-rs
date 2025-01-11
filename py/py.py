import torch
import torch.nn.functional as F

def inference_process(text, model, tokenizer, device):
    # 1. Tokenization
    # 将输入文本转换为token ids
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    print(f"1. Tokenized input shape: {input_ids.shape}")
    print(f"Token IDs: {input_ids[0].tolist()}")
    
    # 2. Input Embedding
    # 获取模型的词嵌入层
    token_embeddings = model.get_input_embeddings()(input_ids)
    print(f"2. Input embeddings shape: {token_embeddings.shape}")
    
    # 3. Decoder Layers
    hidden_states = token_embeddings
    
    for layer_idx, decoder_layer in enumerate(model.transformer.h):
        # 3.1 Pre-attention RMS Norm
        if hasattr(decoder_layer, 'ln_1'):
            normalized_hidden_states = decoder_layer.ln_1(hidden_states)
            print(f"3.1 Layer {layer_idx} - Pre-attention RMS Norm shape: {normalized_hidden_states.shape}")
        
        # 3.2 Masked Multi-head Self-attention
        # 创建attention mask
        attention_mask = torch.ones_like(input_ids)
        causal_mask = torch.triu(torch.ones(input_ids.shape[1], input_ids.shape[1]), diagonal=1).bool()
        causal_mask = causal_mask.to(device)
        
        # 执行self-attention
        attn_output = decoder_layer.attn(
            normalized_hidden_states,
            attention_mask=attention_mask,
            causal_mask=causal_mask
        )[0]
        print(f"3.2 Layer {layer_idx} - Self-attention output shape: {attn_output.shape}")
        
        # 残差连接
        hidden_states = hidden_states + attn_output
        
        # 3.3 Post-attention RMS Norm
        if hasattr(decoder_layer, 'ln_2'):
            normalized_hidden_states = decoder_layer.ln_2(hidden_states)
            print(f"3.3 Layer {layer_idx} - Post-attention RMS Norm shape: {normalized_hidden_states.shape}")
        
        # 3.4 MLP
        mlp_output = decoder_layer.mlp(normalized_hidden_states)
        print(f"3.4 Layer {layer_idx} - MLP output shape: {mlp_output.shape}")
        
        # 残差连接
        hidden_states = hidden_states + mlp_output
    
    # 4. Final RMS Norm
    if hasattr(model.transformer, 'ln_f'):
        hidden_states = model.transformer.ln_f(hidden_states)
    print(f"4. Final RMS Norm output shape: {hidden_states.shape}")
    
    # 5. Output Embedding (Language Modeling Head)
    # 使用输出嵌入矩阵将隐藏状态映射到词汇表大小的logits
    lm_logits = model.lm_head(hidden_states)
    print(f"5. Output logits shape: {lm_logits.shape}")
    
    # 6. Generate next token
    # 获取最后一个位置的logits
    next_token_logits = lm_logits[:, -1, :]
    
    # 应用temperature
    temperature = 0.6
    next_token_logits = next_token_logits / temperature
    
    # 应用top_k采样
    top_k = 40
    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
    
    # 计算softmax概率
    probs = F.softmax(top_k_logits, dim=-1)
    
    # 采样下一个token
    next_token = torch.multinomial(probs, num_samples=1)
    next_token = top_k_indices.gather(-1, next_token)
    
    # 7. Decode
    generated_token = tokenizer.decode(next_token[0])
    print(f"7. Generated next token: {generated_token}")
    
    return next_token

# 使用示例
def generate_text(prompt, model, tokenizer, device, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    for _ in range(max_length):
        next_token = inference_process(
            tokenizer.decode(input_ids[0]),
            model,
            tokenizer,
            device
        )
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# 调用示例
# text = "<|start_story|>Once upon a time, "
# generated_text = generate_text(text, model, tokenizer, device)
# print("Generated text:", generated_text)
