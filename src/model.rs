use std::fs::File;
use std::process::exit;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use rand::seq;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0); // (seq, n_h * dqkv) = (seq, d) @ (d, n_h * dqkv)
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0); // (seq, n_kv_h * dqkv) = (seq, d) @ (d, n_kv_h * dqkv)
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0); // (seq, n_kv_h * dqkv) = (seq, d) @ (d, n_kv_h * dqkv)
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            // q.print();
            // full_k.print();
            // full_v.print();
            // exit(0);

            // todo!("self_attention(...)");
            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );
            // todo!("down_proj matmul and add residual");
            // out = attn_V @ O_weight.T;
            // residual = out + residual;
            OP::matmul_transb(
                &mut residual,
                1.0,
                &hidden_states,
                &self.params.wo[layer],
                1.0,
            );
            // todo!("mlp(...)");
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        // let mut result = Vec::<u32>::new();

        // todo!("实现文本生成");

        // result
        let mut result = Vec::<u32>::new(); // 初始化结果，用输入的 token_ids 开始

        let mut cache = self.new_cache();

        let mut input = Tensor::<u32>::new(token_ids.to_vec(), &vec![token_ids.len()]);

        while result.len() < max_len {
            // 调用模型进行推理，获取当前序列的 logits
            let logits = self.forward(&input, &mut cache); // 假设返回一个 Tensor<f32>

            // 使用 random_sample 函数进行采样
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);

            result.push(next_token);

            // 停止条件：如果生成了结束标志 token，则提前退出
            if next_token == self.eos_token_id {
                break;
            }
            input = Tensor::<u32>::new(vec![next_token], &vec![1]);
        }

        result
    }
}

fn self_attention(
    mut hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,        // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                     // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                     // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                     // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // q: [seq_len, n_kv_h * n_groups * dqkv] -> [n_kv_h, n_groups, seq_len, dqkv]
    let q = Tensor::new(q.data().to_vec(), &vec![seq_len, n_kv_h * n_groups * dqkv]);
    let mut q = OP::trans(&q);
    let q = q.reshape(&vec![n_kv_h, n_groups, dqkv, seq_len]);
    let q = OP::trans(q);

    // k: [total_seq_len, n_kv_h * dqkv] -> [n_kv_h, 1, total_seq_len, dqkv]
    let k = Tensor::new(k.data().to_vec(), &vec![total_seq_len, n_kv_h * dqkv]);
    let mut k = OP::trans(&k);
    let k = k.reshape(&vec![n_kv_h, 1, dqkv, total_seq_len]);
    let k = OP::trans(k);

    // att_scores: [n_kv_h, n_groups, seq_len, total_seq_len]
    OP::matmul_transb(att_scores, 0., &q, &k, 1. / (dqkv as f32).sqrt());
    OP::masked_softmax(att_scores);

    // temp: [n_kv_h, n_groups, seq_len, dqkv]
    let mut temp = Tensor::new(
        hidden_states.data().to_vec(),
        &vec![n_kv_h, n_groups, seq_len, dqkv],
    );

    // v: [total_seq_len, n_kv_h * dqkv] -> [n_kv_h, 1, dqkv, total_seq_len]
    let v = Tensor::new(v.data().to_vec(), &vec![total_seq_len, n_kv_h * dqkv]);
    let mut v = OP::trans(&v);
    let v = v.reshape(&vec![n_kv_h, 1, dqkv, total_seq_len]);

    // temp: [n_kv_h, n_groups, seq_len, dqkv]
    OP::matmul_transb(&mut temp, 0., &att_scores, &v, 1.);

    // temp: [n_kv_h, n_groups, seq_len, dqkv] -> [seq_len, n_kv_h * n_groups * dqkv]
    let mut temp = OP::trans(&temp);
    let temp = temp.reshape(&vec![n_kv_h * n_groups * dqkv, seq_len]);
    let temp = OP::trans(&temp);
    
    // hidden_states = temp
    let temp_data = temp.data();
    let hidden_states_data_mut = unsafe { hidden_states.data_mut() };
    let m = seq_len;
    let n = n_kv_h * n_groups * dqkv;
    for i in 0..m {
        for j in 0..n {
            hidden_states_data_mut[i * n + j] = temp_data[i * n + j];
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // todo!("Implement mlp");
    OP::rms_norm(hidden_states, residual, rms_w, eps);
    OP::matmul_transb(gate, 0., hidden_states, w_gate, 1.);
    OP::matmul_transb(up, 0., hidden_states, w_up, 1.);
    OP::swiglu(up, gate);
    OP::matmul_transb(residual, 1., up, w_down, 1.);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}

#[test]
pub fn test_self_attention() {
    let seq_len = 2;
    let total_seq_len = 4;
    let hidden_size = 4;
    let n_q_h = 2;
    let n_kv_h = 1;
    let n_groups = n_q_h / n_kv_h;
    let dqkv = hidden_size / n_q_h;
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, n_kv_h * n_groups * dqkv]); //
    let mut att_scores = Tensor::<f32>::default(&vec![n_kv_h, n_groups, seq_len, total_seq_len]);
    let q = Tensor::<f32>::new(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        &vec![seq_len, n_kv_h * n_groups * dqkv],
    );
    let k = Tensor::<f32>::new(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        &vec![total_seq_len, n_kv_h * dqkv],
    );
    let v = Tensor::<f32>::new(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        &vec![total_seq_len, n_kv_h * dqkv],
    );

    self_attention(
        &mut hidden_states,
        &mut att_scores,
        &q,
        &k,
        &v,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );
    hidden_states.print();

    assert!(hidden_states.close_to(
        &Tensor::<f32>::new(
            vec![
                0.30565512, 0.40565515, 0.31317782, 0.41317782, 
                0.43862665, 0.5386267, 0.45236826, 0.5523683
            ],
            &vec![seq_len, hidden_size]
        ),
        1e-3
    ))
}
