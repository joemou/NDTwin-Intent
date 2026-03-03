from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

# 1. 載入模型 (Llama 3.1 8B 4bit)
max_seq_length = 4096
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

print("正在下載/載入 Llama 3.1 8B 模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# 2. 設定 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. 處理數據格式
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = load_dataset("json", data_files="train.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 4. 訓練參數
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2, 
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 100, 
        learning_rate = 3e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 5. 開始訓練
print("開始訓練...")
trainer_stats = trainer.train()

# 6. 測試推論
print("\n=== 測試剛訓練好的模型 ===")
FastLanguageModel.for_inference(model)

messages = [
    {"from": "system", "value": "You control a network system. Respond in specified JSON format."},
    {"from": "human", "value": "Modify flow entry on s3 to drop packets"}
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
print(tokenizer.batch_decode(outputs)[0].split("<|start_header_id|>assistant<|end_header_id|>")[-1])


# 儲存 LoRA 配接器 (Adapter) 到 'lora_model' 資料夾
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

print("模型已儲存至 lora_model 資料夾！")
