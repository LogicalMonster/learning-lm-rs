// mod config;
// mod kvcache;
// mod model;
// mod operators;
// mod params;
// mod tensor;

// use std::path::PathBuf;
// use tokenizers::Tokenizer;

// fn main() {
//     let project_dir = env!("CARGO_MANIFEST_DIR");
//     // let model_dir = PathBuf::from(project_dir).join("models").join("story");
//     let model_dir = PathBuf::from(project_dir).join("models").join("chat");
//     let llama = model::Llama::<f32>::from_safetensors(&model_dir);
//     let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
//     // let input = "Once upon a time";
//     let input = "<|im_start|>system
// You are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful, helpful, and focused on delivering the most accurate information to the user.<|im_end|>
// <|im_start|>user
// Hey! Got a question for you!<|im_end|>
// <|im_start|>assistant";
//     let binding = tokenizer.encode(input, true).unwrap();
//     let input_ids = binding.get_ids();
//     print!("\n{}", input);
//     let output_ids = llama.generate(
//         input_ids,
//         500,
//         0.8,
//         30,
//         1.,
//     );
//     println!("{}", tokenizer.decode(&output_ids, true).unwrap());
// }


mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// 定义对话中的消息结构
struct Message {
    role: String,    // 角色，如 "user" 或 "assistant"
    content: String, // 消息内容
}

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

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    // 选择模型目录（此处可以选择不同模型，比如 "story" 或 "chat"）
    // let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");

    // 加载模型（假设模型类型为 Llama<f32>，并提供 from_safetensors 接口）
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    // 加载 tokenizer（假设使用 HuggingFace Tokenizers 格式）
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    // 进入对话
    chat(&llama, &tokenizer);
}
