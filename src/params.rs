use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        // 打印所有张量名称
        println!("Available tensors:");
        for name in safetensor.names() {
            println!("{}", name);
        }

        // 辅助函数：从 safetensors 中获取张量
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let shape = tensor.shape().to_vec();
            let data: Vec<f32> = tensor
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            Tensor::new(data, &shape)
        };

        // 初始化各层参数的向量
        let n_layers = config.num_hidden_layers;
        let mut rms_att_w = Vec::with_capacity(n_layers);
        let mut wq = Vec::with_capacity(n_layers);
        let mut wk = Vec::with_capacity(n_layers);
        let mut wv = Vec::with_capacity(n_layers);
        let mut wo = Vec::with_capacity(n_layers);
        let mut rms_ffn_w = Vec::with_capacity(n_layers);
        let mut w_up = Vec::with_capacity(n_layers);
        let mut w_gate = Vec::with_capacity(n_layers);
        let mut w_down = Vec::with_capacity(n_layers);

        // 加载每一层的参数
        for i in 0..n_layers {
            rms_att_w.push(get_tensor(&format!(
                "model.layers.{i}.input_layernorm.weight"
            )));
            wq.push(get_tensor(&format!(
                "model.layers.{i}.self_attn.q_proj.weight"
            )));
            wk.push(get_tensor(&format!(
                "model.layers.{i}.self_attn.k_proj.weight"
            )));
            wv.push(get_tensor(&format!(
                "model.layers.{i}.self_attn.v_proj.weight"
            )));
            wo.push(get_tensor(&format!(
                "model.layers.{i}.self_attn.o_proj.weight"
            )));
            rms_ffn_w.push(get_tensor(&format!(
                "model.layers.{i}.post_attention_layernorm.weight"
            )));
            w_up.push(get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight")));
            w_gate.push(get_tensor(&format!(
                "model.layers.{i}.mlp.gate_proj.weight"
            )));
            w_down.push(get_tensor(&format!(
                "model.layers.{i}.mlp.down_proj.weight"
            )));
        }

        // 构建并返回 LLamaParams 结构体
        LLamaParams {
            embedding_table: get_tensor(if config.tie_word_embeddings {
                "lm_head.weight"
            } else {
                "model.embed_tokens.weight"
            }), // 使用 lm_head 作为嵌入表
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
