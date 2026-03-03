from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs/checkpoint-60", # 這是剛剛訓練完生成的權重資料夾
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

FastLanguageModel.for_inference(model) # 啟用加速推論

# 2. 定義跟訓練資料「一模一樣」的 System Prompt
system_prompt = """You control a network system. You are a classification task agent. Respond in the specified JSON format. 
If a command is potentially disruptive, your default action is to enter the discussion state to ask for confirmation. 
However, if the user's prompt includes a confirmation flag like --confirm or --force, you are permitted to skip the discussion and proceed directly to the answer. 
If the user wants to Install, Modify, or Delete flows, ALWAYS use the RequestUIForm task."""

# 3. 準備測試訊息
messages = [
    {"from": "system", "value": system_prompt},
    {"from": "human", "value": "Modify flow entry on s3 to drop packets"}
]

# 4. 格式化輸入
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

# 5. 生成
print("=== Generating Response ===")
outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 512,
    use_cache = True,
    temperature = 0.1, # JSON 生成建議用低溫
)

# 6. 解碼結果
response = tokenizer.batch_decode(outputs)
print(response[0].split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip())

# 7. 儲存完整模型 (這是你原本的目的)
# 這樣下次你就不用讀取 checkpoint-60，直接讀取 lora_model 即可
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

print("模型已儲存至 lora_model 資料夾！")