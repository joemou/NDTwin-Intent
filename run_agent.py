from unsloth import FastLanguageModel
import torch

# 1. Set model path
# Point this to the folder where you saved your adapter (e.g., "outputs/lora_model")
model_path = "lora_model_fold_1" 

# 2. Load Model and Adapter
print(f"Loading model from: {model_path} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# 3. Define System Prompt 
# (⚠️ Important: This must be EXACTLY the same as the one used during training)
system_prompt = """You You control a network system. You are a classification task agent. Respond in the specified JSON format. RULES: 1. If a command is potentially disruptive (e.g., Disable, PowerOff, Block, Delete, Drop, Modify, Reroute), your default action is to enter the \"discussion\" state to ask for confirmation. 2. If the user's prompt includes a confirmation flag like --confirm or --force, skip the discussion and proceed directly to \"answer\". 3. Keep the \"explanation\" field extremely concise (under 15 words). DO NOT think out loud or explain your internal logic. CRITICAL INSTRUCTION: You MUST strictly use ONLY the exact task names listed below. DO NOT invent, guess, or modify task names under any circumstances. CRITICAL INSTRUCTION: You MUST strictly use ONLY the exact task names listed below. DO NOT invent, guess, or modify task names under any circumstances."""

# 4. Define a helper function for inference
def ask_agent(user_input):
    messages = [
        {"from": "system", "value": system_prompt},
        {"from": "human", "value": user_input}
    ]
    
    # Apply the specific chat template (Llama-3 format)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
        return_dict = True
    ).to("cuda")

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens = 512,
        use_cache = True,
        temperature = 0.1, # Low temperature recommended for JSON tasks
    )
    
    # Decode 並且跳過特殊標記
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # 用更保險的方式只抓取 JSON 部分 (找第一個 { 和最後一個 })
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        result = response[json_start:json_end]
    except:
        result = response # 如果找不到括號就印出原始回答
        
    return result

# 5. Main Interaction Loop
if __name__ == "__main__":
    while True:
        query = input("\nEnter command (type 'q' to exit): ")
        if query.lower() == 'q':
            print("Exiting...")
            break
        
        print("Agent is thinking...")
        response = ask_agent(query)
        print("=== Response Result ===")
        print(response)