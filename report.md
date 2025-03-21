# 简单大模型推理系统项目报告

## 进度说明

添加self-attention测试用例, 并给出pytorch实现进行对照, 实现模型推理和 AI 对话功能。

## 一、作业阶段

### 1. 算子：SiLU函数（10分）

```rust
// operators.rs
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let y = unsafe { y.data_mut() };
    let x = x.data();

    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
    for i in 0..len {
        y[i] = (1. / (1. + (-x[i]).exp())) * x[i] * y[i];
    }
}
```

### 2. 算子：RMS Normalization（20分）

```rust
// operators.rs
pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试")
    let len = y.size();
    assert!(len == x.size());
    let n = w.size();
    assert!(len % n == 0);

    let y = unsafe { y.data_mut() };
    let x = x.data();
    let w = w.data();

    // 按分组大小分割数据
    for (x_chunk, y_chunk) in x.chunks(n).zip(y.chunks_mut(n)) {
        // 计算RMS
        let sum_squares: f32 = x_chunk.iter().map(|&e| e * e).sum();
        let rms = (sum_squares / n as f32 + epsilon).sqrt();

        // 应用权重和归一化
        for ((x_val, y_val), w_val) in x_chunk.iter().zip(y_chunk.iter_mut()).zip(w.iter()) {
            *y_val = w_val * x_val / rms;
        }
    }
}
```

### 3. 算子：矩阵乘(广播乘法)（30分）

```rust
// operators.rs
pub fn trans(b: &Tensor<f32>) -> Tensor<f32> {
    // 获取形状和尺寸
    let b_shape = b.shape();
    let (m, n) = (b_shape[b_shape.len() - 2], b_shape[b_shape.len() - 1]);

    // 更新后的形状：交换最后两个维度
    let mut shape = b_shape.to_vec();
    shape[b_shape.len() - 2] = n;
    shape[b_shape.len() - 1] = m;

    // 原始数据
    let b_data = b.data();

    // 新的数据向量
    let mut data = Vec::with_capacity(b_data.len());

    // 按块迭代，并进行转置
    for chunk in b_data.chunks(m * n) {
        for j in 0..n {
            for i in 0..m {
                // 使用直接索引访问元素
                data.push(chunk[i * n + j]);
            }
        }
    }

    // 构造新的 Tensor
    Tensor::<f32>::new(data, &shape)
}

pub fn mul(a: &mut Tensor<f32>, alpha: f32) {
    let len = a.size();

    // 原始数据
    let a = unsafe { a.data_mut() };

    // 按块迭代，并进行转置
    for i in 0..len {
        a[i] *= alpha;
    }
}

pub fn add(a: &mut Tensor<f32>, b: &Tensor<f32>) {
    let len = a.size();

    // 原始数据
    let a = unsafe { a.data_mut() };
    let b = b.data();

    // 按块迭代，并进行转置
    for i in 0..len {
        a[i] += b[i];
    }
}

// 辅助函数：计算广播形状
fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    let mut result_shape = vec![];
    let max_rank = shape1.len().max(shape2.len());
    for i in 0..max_rank {
        let dim1 = if i < shape1.len() {
            shape1[shape1.len() - 1 - i]
        } else {
            1
        };
        let dim2 = if i < shape2.len() {
            shape2[shape2.len() - 1 - i]
        } else {
            1
        };
        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            panic!("Shapes {:?} and {:?} are not broadcastable", shape1, shape2);
        }
        result_shape.push(dim1.max(dim2));
    }
    result_shape.reverse();
    result_shape
}

// 辅助函数：根据广播后的形状计算原始张量的索引
fn broadcast_index(
    original_shape: &[usize],
    broadcast_shape: &[usize],
    broadcast_idx: usize,
) -> usize {
    let mut original_idx = 0;
    let mut stride = 1;
    let mut broadcast_stride = 1;
    for (&orig_dim, &bcast_dim) in original_shape
        .iter()
        .rev()
        .zip(broadcast_shape.iter().rev())
    {
        let coord = (broadcast_idx / broadcast_stride % bcast_dim) % orig_dim;
        original_idx += coord * stride;
        stride *= orig_dim;
        broadcast_stride *= bcast_dim;
    }
    original_idx
}

pub fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    // 检查输入维度是否符合矩阵乘法要求
    if a.shape().len() < 2 || b.shape().len() < 2 {
        panic!("Both tensors must be at least 2D for matrix multiplication.");
    }
    if a.shape()[a.shape().len() - 1] != b.shape()[b.shape().len() - 2] {
        panic!(
            "Matrix multiplication not possible: {:?} and {:?}",
            a.shape(),
            b.shape()
        );
    }

    // 获取输入张量的非矩阵维度
    let batch_shape1 = &a.shape()[..a.shape().len() - 2];
    let batch_shape2 = &b.shape()[..b.shape().len() - 2];

    // 计算广播后的批量维度
    let broadcast_shape = broadcast_shape(batch_shape1, batch_shape2);

    // 获取矩阵乘法的维度
    let m = a.shape()[a.shape().len() - 2];
    let n = a.shape()[a.shape().len() - 1];
    let p = b.shape()[b.shape().len() - 1];

    // 最终结果的形状
    let mut result_shape = broadcast_shape.clone();
    result_shape.push(m);
    result_shape.push(p);

    // 数据准备
    let result_size: usize = result_shape.iter().product();
    let mut result_data = vec![0.0; result_size];

    // 迭代批量维度，计算每个矩阵的结果
    for batch_idx in 0..broadcast_shape.iter().product::<usize>() {
        // 计算广播后的索引
        let idx1 = broadcast_index(&batch_shape1, &broadcast_shape, batch_idx);
        let idx2 = broadcast_index(&batch_shape2, &broadcast_shape, batch_idx);

        // 提取具体的矩阵
        let a = &a.data()[idx1 * m * n..][..m * n];
        let b = &b.data()[idx2 * n * p..][..n * p];

        // 计算矩阵乘法结果
        let result_offset = batch_idx * m * p;
        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    result_data[result_offset + i * p + j] += a[i * n + k] * b[k * p + j];
                }
            }
        }
    }

    // 创建结果张量
    Tensor::new(result_data, &result_shape)
}

// 矩阵乘法（B矩阵转置）
// C = beta * C + alpha * A @ B^T
// 注意：不需要显式地转置B矩阵
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");
    mul(c, beta);
    let mut foo = matmul(a, &trans(b));
    mul(&mut foo, alpha);
    add(c, &foo);
}
```

### 4. 模型结构：Feed-Forward神经网络（20分）

```rust
// model.rs
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
```

### 5. Llama模型参数加载（20分）

```rust
// params.rs
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
```

## 二、项目阶段

### 1. 模型结构：Self-Attention

实现一：
```rust
// model.rs
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
```

实现二：
```rust
// model.rs
fn self_attention(
    hidden_states: &mut Tensor<f32>, // 输出张量，形状为 [seq_len, n_kv_h * n_groups * dqkv]
    att_scores: &mut Tensor<f32>,    // 临时张量，形状为 [n_kv_h, n_groups, seq_len, total_seq_len]
    q: &Tensor<f32>,                 // 查询张量，形状为 [seq_len, n_kv_h * n_groups * dqkv]
    k: &Tensor<f32>,                 // 键张量，形状为 [total_seq_len, n_kv_h * dqkv]
    v: &Tensor<f32>,                 // 值张量，形状为 [total_seq_len, n_kv_h * dqkv]
    n_kv_h: usize,   // 针对 k 和 v 的 head 数
    n_groups: usize, // 对于 q 来说：n_q_heads = n_kv_h * n_groups
    seq_len: usize,  // 当前序列长度（查询数目）
    total_seq_len: usize, // 过去与当前所有 token 数（键和值的行数）
    dqkv: usize,     // 每个 q、k、v 的向量长度
) {
    // 缩放因子 1/sqrt(dqkv)
    let scale = 1.0 / (dqkv as f32).sqrt();

    // 取 q, k, v 的底层数据（只读）
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();

    // --- 1. 计算注意力得分 ---
    // 将 q 视为逻辑形状 [seq_len, n_kv_h, n_groups, dqkv]，即：
    //   q[t, head, group, d] = q[ t * (n_kv_h*n_groups*dqkv) + (head*n_groups+group)*dqkv + d ]
    //
    // k 视为 [total_seq_len, n_kv_h, dqkv]：
    //   k[s, head, d] = k[ s * (n_kv_h*dqkv) + head*dqkv + d ]
    //
    // att_scores 的逻辑形状为 [n_kv_h, n_groups, seq_len, total_seq_len]，
    // 下标计算公式：
    //   att_scores[head, group, t, s] =
    //       att_data[ ((head * n_groups + group) * seq_len + t) * total_seq_len + s ]
    //
    // 此处将所有对 att_scores 的写操作限定在一个代码块中，
    // 保证结束后 mutable borrow 结束，从而后续可再次借用 att_scores。
    {
        let att_data = unsafe { att_scores.data_mut() };
        for head in 0..n_kv_h {
            for group in 0..n_groups {
                for t in 0..seq_len {
                    // 计算 att_scores 在底层存储中的起始下标：
                    let att_base = ((head * n_groups + group) * seq_len + t) * total_seq_len;
                    // q[t, head, group, :] 在 q 中的起始下标
                    let q_base = t * (n_kv_h * n_groups * dqkv) + (head * n_groups + group) * dqkv;
                    for s in 0..total_seq_len {
                        let mut dot = 0.0;
                        // k[s, head, :] 在 k 中的起始下标
                        let k_base = s * (n_kv_h * dqkv) + head * dqkv;
                        for d in 0..dqkv {
                            dot += q_data[q_base + d] * k_data[k_base + d];
                        }
                        att_data[att_base + s] = dot * scale;
                    }
                }
            }
        }
    } // 此处 mutable borrow 对 att_scores 结束

    // --- 2. 应用 masked softmax ---
    // 调用已有的实现，对 att_scores 进行归一化处理
    OP::masked_softmax(att_scores);

    // --- 3. 利用注意力权重对 v 加权求和，得到输出 hidden_states ---
    // 输出 hidden_states 的逻辑形状为 [seq_len, n_kv_h, n_groups, dqkv]，
    // 下标计算为：
    //   hidden_states[t, head, group, d] =
    //     sum_{s=0}^{total_seq_len-1} ( att_scores[head, group, t, s] * v[s, head, d] )
    let hidden_data = unsafe { hidden_states.data_mut() };
    {
        // 此处只需要对 att_scores 进行只读借用，因而使用 data() 即可
        let att_data = att_scores.data();
        for head in 0..n_kv_h {
            for group in 0..n_groups {
                for t in 0..seq_len {
                    let out_base = t * (n_kv_h * n_groups * dqkv) + (head * n_groups + group) * dqkv;
                    let att_base = ((head * n_groups + group) * seq_len + t) * total_seq_len;
                    for d in 0..dqkv {
                        let mut sum_val = 0.0;
                        for s in 0..total_seq_len {
                            // v[s, head, d] 在 v 中的下标：
                            let v_index = s * (n_kv_h * dqkv) + head * dqkv + d;
                            sum_val += att_data[att_base + s] * v_data[v_index];
                        }
                        hidden_data[out_base + d] = sum_val;
                    }
                }
            }
        }
    }
}
```

### 2. 功能：文本生成

```rust
// model.rs
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
```

### 3. （可选）功能：AI对话

代码见分支chat

```rust
// model.rs
    /// chat_generate 与 generate 类似，但接收外部 kvcache 用于多轮对话
    pub fn chat_generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        cache: &mut KVCache<f32>,  // 假定 kvcache 模块中定义了 Cache 类型
    ) -> Vec<u32> {
        let mut result = Vec::<u32>::new();
        let mut input = Tensor::<u32>::new(token_ids.to_vec(), &vec![token_ids.len()]);
        while result.len() < max_len {
            // 前向推理，传入当前 token 及持久化 cache
            let logits = self.forward(&input, cache);
            // 根据 logits 使用 top-p、top-k 及温度采样生成下一个 token
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result.push(next_token);
            // 如果生成结束符则退出
            if next_token == self.eos_token_id {
                break;
            }
            // 下次生成时只输入刚生成的 token（同时 cache 中已保存了之前的信息）
            input = Tensor::<u32>::new(vec![next_token], &vec![1]);
        }
        result
    }
```

```rust
// main.rs
/// 实现一个支持多轮对话的 chat 函数
fn chat(llama: &model::Llama<f32>, tokenizer: &Tokenizer) {
    // 保存多轮对话的消息记录
    let mut messages: Vec<Message> = Vec::new();

    // 设置系统 prompt
    messages.push(Message {
        role: "system".to_string(),
        content: "You are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful, helpful, and focused on delivering the most accurate information to the user.".to_string(),
    });

    // 初始化 kvcache，一次对话过程中不断复用
    let mut kvcache = llama.new_cache();

    println!("开始对话，输入 exit 或 quit 退出。");

    loop {
        // 读取用户输入
        print!("User: ");
        io::stdout().flush().unwrap();
        let mut user_input = String::new();
        if io::stdin().read_line(&mut user_input).is_err() {
            eprintln!("读取输入出错！");
            continue;
        }
        let user_input = user_input.trim();
        if user_input.is_empty() {
            continue;
        }
        if user_input.eq_ignore_ascii_case("exit") || user_input.eq_ignore_ascii_case("quit") {
            break;
        }
        // 将用户消息加入对话记录
        messages.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        // 按照 Jinja2 模板构造对话 prompt：
        //
        //   对每条消息，输出："<|im_start|>{role}\n{content}<|im_end|>\n"
        //   最后如果需要生成回答，则附加生成提示："<|im_start|>assistant\n"
        //
        let mut prompt = String::new();
        for message in &messages {
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                message.role, message.content
            ));
        }
        // 添加生成提示
        prompt.push_str("<|im_start|>assistant\n");

        // 对 prompt 进行 token 化
        let encoding = tokenizer.encode(prompt.clone(), true).unwrap();
        let input_ids = encoding.get_ids();

        // 调用 chat_generate 接口生成回答，传入之前复用的 kvcache
        // let output_ids = llama.chat_generate(
        //     input_ids,
        //     500,    // 最大生成 token 数
        //     0.8,    // top_p
        //     30,     // top_k
        //     1.0,    // temperature
        //     &mut kvcache, // 使用对话过程中保存的 kvcache
        // );
        let output_ids = llama.chat_generate(
            input_ids,
            500,    // 最大生成 token 数
            0.55,    // top_p
            35,     // top_k
            0.65,    // temperature
            &mut kvcache, // 使用对话过程中保存的 kvcache
        );

        // 解码生成的 token 得到文本回答
        let response = tokenizer.decode(&output_ids, true).unwrap();
        println!("Assistant: {}", response);

        // 将助手回答加入对话记录
        messages.push(Message {
            role: "assistant".to_string(),
            content: response,
        });
    }
}
```
