from unsloth import FastLanguageModel

# 載入你的 LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # 指向你的資料夾
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)

# 轉存成 GGUF (q4_k_m 是最推薦的量化格式)
model.save_pretrained_gguf("my_network_agent", tokenizer, quantization_method = "q4_k_m")