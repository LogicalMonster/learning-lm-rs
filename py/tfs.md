```
input_ids: (batch_size, sequence_length)
inputs_embeds: (batch_size, sequence_length, hidden_size)
cache_position: (sequence_length)
position_ids: (batch_size, sequence_length)
causal_mask = None
hidden_states: (batch_size, sequence_length, hidden_size)
position_embeddings: ((batch_size, sequence_length, hidden_size // num_attention_heads), (batch_size, sequence_length, hidden_size // num_attention_heads))
attn_implementation="sdpa"
eager: LlamaAttention
flash_attention_2: LlamaFlashAttention2
sdpa: LlamaSdpaAttention
```
