# Tokenizer: 文本 -> ids
# vocab_size: 32000

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./models/story"

tokenzier = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path)

text = "Once upon a time, "

ids = tokenzier.encode(text)

tokens = tokenzier.convert_ids_to_tokens(ids)

print('type(ids):', type(ids))
print('ids:', ids)
print('type(tokens):', type(tokens))
print('tokens:', tokens)

# Input Embedding
# hidden_size(embedding_dim): 4096

# RMS Norm
# hidden_size: 4096
# rms_norm: 1e-05


# MLP
# hidden_size: 4096
# intermediate_size: 11008
# hidden_act: silu

# Self-Attention
# hidden_size: 4096
# num_attention_heads: 32
# num_key_value_heads: 32

# 在Decoder中，每个token只对它前面的token求注意力，使用masked softmax
# Encoder使用普通的softmax
# Llama是Decoder-only模型

# Q: (L, hidden_size) -> (L, num_attention_heads, hidden_size // num_attention_heads)
# K: (L, hidden_size) -> (L, num_key_value_heads, hidden_size // num_key_value_heads)
# V: (L, hidden_size) -> (L, num_key_value_heads, hidden_size // num_key_value_heads)

# Q: (L, num_attention_heads, hidden_size // num_attention_heads) -> (L, num_key_value_heads, num_attention_heads // num_key_value_heads, hidden_size // num_attention_heads)
# K: (L, num_key_value_heads, hidden_size // num_key_value_heads) -> (L, num_key_value_heads, 1, hidden_size // num_key_value_heads)
# V: (L, num_key_value_heads, hidden_size // num_key_value_heads) -> (L, num_key_value_heads, 1, hidden_size // num_key_value_heads)

# Decoder
# num_hidden_layers: 32

# Output Embedding: ? -> logits
# tie_word_embeddings: False

# Embedding
# num_embeddings: vocab_size
# embedding_dim: hidden_size

# transformer
# d_model: hidden_size
# nhead: num_attention_heads
# num_decoder_layers: num_hidden_layers
# dim_feedforward: intermediate_size
# activation: silu
# layer_norm_eps: rms_norm
# batch_first: False (seq, batch, feature) ?
# norm_first: True ?
# bias: True ?

# TransformerDecoderLayer
# d_model: hidden_size
# nhead: num_attention_heads
# dim_feedforward: intermediate_size
# activation: silu
# layer_norm_eps: rms_norm
# batch_first: False (seq, batch, feature) ?
# norm_first: True ?
# bias: True ?

# TransformerDecoder
# num_layers: num_hidden_layers
# norm: rms

# 小项目和transformer库的实现一致，大项目则不然
