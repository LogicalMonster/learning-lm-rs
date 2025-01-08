from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型目录
model_path = "./models/story"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path)

# 测试模型和分词器
text = "<|start_story|>Once upon a time, "
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(outputs)
