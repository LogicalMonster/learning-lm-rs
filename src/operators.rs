// src/operators.rs
use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

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

// y = silu(x) * y
// hint: this is an element-wise operation
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

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
