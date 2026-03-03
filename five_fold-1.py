import json
import torch
import gc
from tqdm import tqdm
from sklearn.model_selection import KFold
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# 0. Global Configurations
# ==========================================
ALL_DATA_PATH = "all_data.json" # ★ Please merge all your data into this file
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096

SYSTEM_PROMPT = """You control a network system. You are a classification task agent. Respond in the specified JSON format. RULES: 1. If a command is potentially disruptive (e.g., Disable, PowerOff, Block, Delete, Drop, Modify, Reroute), your default action is to enter the \"discussion\" state to ask for confirmation. 2. If the user's prompt includes a confirmation flag like --confirm or --force, skip the discussion and proceed directly to \"answer\". 3. Keep the \"explanation\" field extremely concise (under 15 words). DO NOT think out loud or explain your internal logic."""

# ==========================================
# 1. Helper Functions (JSON Extraction & Comparison)
# ==========================================
def extract_json(response_str):
    try:
        if response_str.startswith("```json"): response_str = response_str[7:]
        if response_str.startswith("```"): response_str = response_str[3:]
        if response_str.endswith("```"): response_str = response_str[:-3]
        
        start_idx = response_str.find('{')
        end_idx = response_str.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            return json.loads(response_str[start_idx:end_idx])
        return None
    except Exception:
        return None

def compare_outputs(expected, actual):
    if actual is None: return False, "Failed to parse into a valid JSON format"
    if expected.get("state") != actual.get("state"):
        return False, f"State mismatch (Expected: {expected.get('state')}, Actual: {actual.get('state')})"
        
    if expected.get("state") == "answer":
        if expected.get("valid") != actual.get("valid"):
            return False, "Valid flag mismatch"
        exp_tasks = expected.get("tasks", [])
        act_tasks = actual.get("tasks", [])
        if len(exp_tasks) != len(act_tasks):
            return False, "Task count mismatch"
            
        for et, at in zip(exp_tasks, act_tasks):
            if et.get("type") != at.get("type"):
                return False, "Task type mismatch"
            
            et_params, at_params = et.get("parameters", {}), at.get("parameters", {})
            if "device_name" in et_params and et_params.get("device_name") != at_params.get("device_name"):
                return False, "Device Name parsing error"
    return True, "Success"

# ==========================================
# 2. Training Function (Single Fold)
# ==========================================
def train_fold(train_data_list, fold_idx):
    print(f"\n[Fold {fold_idx}] Loading model and setting up LoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    # Convert List to HuggingFace Dataset
    hf_dataset = Dataset.from_list(train_data_list)
    hf_dataset = hf_dataset.map(formatting_prompts_func, batched = True)

    print(f"[Fold {fold_idx}] Starting training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = hf_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2, 
            gradient_accumulation_steps = 8,
            warmup_steps = 5,
            num_train_epochs = 4, # Note: If the dataset grows, consider adjusting num_train_epochs
            learning_rate = 3e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = f"outputs_fold_{fold_idx}",
        ),
    )
    
    trainer.train()
    
    # Save the model for each fold (Optional)
    model.save_pretrained(f"lora_model_fold_{fold_idx}")
    tokenizer.save_pretrained(f"lora_model_fold_{fold_idx}")
    
    return model, tokenizer, trainer

# ==========================================
# 3. Inference and Evaluation Function (Single Fold)
# ==========================================
def evaluate_fold(model, tokenizer, test_data_list, fold_idx):
    print(f"\n[Fold {fold_idx}] Starting inference and validation ({len(test_data_list)} test samples)...")
    FastLanguageModel.for_inference(model)
    
    correct_count = 0
    failed_reports = []

    for idx, item in enumerate(tqdm(test_data_list, desc=f"Evaluating Fold {fold_idx}")):
        conversations = item.get("conversations", [])
        human_input = next(c["value"] for c in conversations if c["from"] == "human")
        expected_str = next(c["value"] for c in conversations if c["from"] == "gpt")
        expected_json = json.loads(expected_str)
        
        # Prepare inputs
        messages = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": human_input}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        # Inference
        input_length = inputs.shape[1]
        outputs = model.generate(
            inputs, max_new_tokens=512, use_cache=True, temperature=0.1, pad_token_id=tokenizer.eos_token_id
        )
        
        response_str = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        actual_json = extract_json(response_str)
        
        # Compare outputs
        is_correct, reason = compare_outputs(expected_json, actual_json)
        if is_correct:
            correct_count += 1
        else:
            failed_reports.append({
                "input": human_input,
                "reason": reason,
                "expected": expected_json,
                "actual_parsed": actual_json
            })
            
    accuracy = correct_count / len(test_data_list)
    print(f"[Fold {fold_idx}] Accuracy: {accuracy*100:.1f}% ({correct_count}/{len(test_data_list)})")
    
    if failed_reports:
        with open(f"error_report_fold_{fold_idx}.json", 'w', encoding='utf-8') as f:
            json.dump(failed_reports, f, indent=2, ensure_ascii=False)
            
    return accuracy

# ==========================================
# 4. Main Program: 5-Fold Cross-Validation Pipeline
# ==========================================
def main():
    print(f"Loading all data from: {ALL_DATA_PATH}")
    with open(ALL_DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
        
    # Setup 5-Fold split
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data), start=1):
        print("\n" + "="*50)
        print(f"🚀 Starting Fold {fold}/5")
        print("="*50)
        
        # Split data
        train_data = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]
        
        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        # 1. Train
        model, tokenizer, trainer = train_fold(train_data, fold)
        
        # 2. Evaluate
        acc = evaluate_fold(model, tokenizer, test_data, fold)
        fold_accuracies.append(acc)
        
        # 3. Free up memory (Crucial to prevent OOM in subsequent folds)
        print(f"[Fold {fold}] Clearing VRAM memory...")
        del model
        del tokenizer
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # Output final report
    print("\n" + "="*50)
    print("📊 5-Fold Cross-Validation Final Report")
    print("="*50)
    for i, acc in enumerate(fold_accuracies, start=1):
        print(f"Fold {i} Accuracy : {acc*100:.2f}%")
        
    avg_acc = sum(fold_accuracies) / len(fold_accuracies)
    print(f"🏆 Average Accuracy: {avg_acc*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()