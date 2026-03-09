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

# 👉 引入受限解碼所需的套件
from typing import List, Optional, Literal, Dict, Union
from pydantic import BaseModel, Field, RootModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

# ==========================================
# 0. 全局設定
# ==========================================
ALL_DATA_PATH = "all_data.json" # ★ 請確認這裡面的解答都符合新 Schema
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096

SYSTEM_PROMPT = """You control a network system. You are a classification task agent. Respond in the specified JSON format. \nRULES: \n1. DISRUPTIVE ACTIONS: If a command is explicitly disruptive (e.g., Disable, PowerOff, Block, Delete, Drop, Modify, Reroute, Update Firmware), enter the \"discussion\" state. NOTE: \"Install\", \"Create\", and \"Add\" are NOT disruptive, proceed to \"answer\" directly.\n2. CONFIRMATION FLAG: If the prompt includes --confirm or --force, skip discussion and proceed to \"answer\". \n3. MISSING PARAMS (STRICT): \"real-time traffic\" queries MUST have port and duration parameters. If missing, you MUST set \"valid\": 0 and explain: \"Missing port ID and duration parameters for real-time traffic monitoring.\"\n4. ALLOWED API LIST & DEFAULTS: You must map intents strictly to these Task types: \n  - Packet loss queries -> GetPacketLossRate\n   - Log queries -> GetDeviceLogs (Always use line_count: 100 unless specified)\n   DO NOT invent new task types like \"GetRealTimeTraffic\" or \"GetDeviceRecentLogs\".\n5. Keep the \"explanation\" under 15 words. DO NOT explain your internal logic."""

# ==========================================
# 1. 嚴格定義 JSON 格式與合法的 Task 名稱 (Pydantic Polymorphism)
# ==========================================
class ActionSchema(BaseModel):
    type: Optional[str] = None
    port: Optional[int] = None

class BandSchema(BaseModel):
    type: Optional[str] = None
    rate: Optional[int] = None

class BucketSchema(BaseModel):
    actions: Optional[List[ActionSchema]] = None

class ParametersSchema(BaseModel):
    device_name: Optional[str] = None
    form_type: Optional[str] = None
    k: Optional[int] = None
    src: Optional[str] = None
    dst: Optional[str] = None
    host_id: Optional[str] = None
    port_id: Optional[int] = None
    status: Optional[str] = None
    match: Optional[Dict[str, str]] = None
    priority: Optional[int] = None
    actions: Optional[List[ActionSchema]] = None
    new_path: Optional[List[str]] = None
    state: Optional[str] = None
    nickname: Optional[str] = None
    group_type: Optional[str] = None
    group_id: Optional[int] = None
    buckets: Optional[List[BucketSchema]] = None
    meter_id: Optional[int] = None
    flags: Optional[List[str]] = None
    bands: Optional[List[BandSchema]] = None
    backup_path: Optional[str] = None
    restore_path: Optional[str] = None
    host: Optional[str] = None
    count: Optional[int] = None
    line_count: Optional[int] = None
    firmware_path: Optional[str] = None
    duration_seconds: Optional[int] = None

class TaskSchema(BaseModel):
    order: int
    type: Literal[
        "DisableSwitch", "EnableSwitch", "PowerOffSwitch", "PowerOnSwitch", 
        "InstallFlowEntry", "ModifyFlowEntry", "DeleteFlowEntry", "GetTopKFlows", 
        "GetSwitchCpuUtilization", "GetTotalPowerConsumption", "GetASwitchCpuUtilization", 
        "GetASwitchPowerConsumption", "GetALinkBandwidthUtilization", "GetTopKCongestedLinks", 
        "GetTopKBandwidthUsers", "GetPath", "GetActiveFlowCount", "GetFlowEntryCount", 
        "GetFlowEntries", "GetNetworkTopology", "GetAllHosts", "BlockHost", 
        "GetLinkLatency", "GetPacketLossRate", "GetSwitchPorts", "RerouteFlow", 
        "GetSwitchMemoryUtilization", "GetSwitchTemperature", "SetSwitchPowerState", 
        "GetPathSwitchCount", "SetDeviceNickname", "ToggleHistoricalLogging", 
        "GetSwitchCapabilities", "InstallGroupEntry", "InstallMeterEntry", 
        "GetDeviceUptime", "RestartDevice", "BackupConfiguration", "RestoreConfiguration", 
        "PingHost", "TracerouteHost", "GetArpTable", "GetMacTable", "SetPortStatus", 
        "GetPortStatistics", "GetDeviceLogs", "ClearDeviceLogs", "UpdateDeviceFirmware", 
        "GetDeviceHealth"
    ]
    parameters: ParametersSchema

# 🚀 殺招：拆分三種絕對互斥的情境，並避開 lmformatenforcer 處理 Literal Int 的崩潰 Bug
class DiscussionResponse(BaseModel):
    state: Literal["discussion"]
    prompt: str = Field(..., description="Ask the user for confirmation.")

class InvalidAnswerResponse(BaseModel):
    state: Literal["answer"]
    valid: int = Field(0, description="Must be 0 to indicate missing params.") # 🛠️ 避開 Bug 的關鍵：改用 int 與 Field(0)
    explanation: str = Field(..., description="Explain why it failed (e.g., missing params).")
    # 這裡刻意不放 tasks，斷絕它生成 API 的可能

class ValidAnswerResponse(BaseModel):
    state: Literal["answer"]
    valid: int = Field(1, description="Must be 1 to indicate valid tasks.")    # 🛠️ 避開 Bug 的關鍵：改用 int 與 Field(1)
    explanation: str
    tasks: List[TaskSchema] = Field(..., min_length=1) # 強制不准為空！不准提早結束！

# 🚀 使用 RootModel 將三者聯集，狀態機會自動根據前面的生成決定後面的路徑
class ResponseSchema(RootModel):
    root: Union[DiscussionResponse, InvalidAnswerResponse, ValidAnswerResponse]

# ==========================================
# 2. 輔助函數 (擷取 JSON 與比對)
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
    if actual is None: return False, "無法解析為有效的 JSON 格式"
    if expected.get("state") != actual.get("state"):
        return False, f"State 不符 (預期: {expected.get('state')}, 實際: {actual.get('state')})"
        
    if expected.get("state") == "answer":
        if expected.get("valid") != actual.get("valid"):
            return False, "Valid 標記不符"
        exp_tasks = expected.get("tasks", [])
        act_tasks = actual.get("tasks", [])
        if len(exp_tasks) != len(act_tasks):
            return False, "Tasks 數量不符"
            
        for et, at in zip(exp_tasks, act_tasks):
            if et.get("type") != at.get("type"):
                return False, "Task 類型不符"
            
            et_params, at_params = et.get("parameters", {}), at.get("parameters", {})
            if "device_name" in et_params and et_params.get("device_name") != at_params.get("device_name"):
                return False, "Device Name 解析錯誤"
    return True, "Success"

# ==========================================
# 3. 訓練函數 (單一 Fold)
# ==========================================
def train_fold(train_data_list, fold_idx):
    print(f"\n[Fold {fold_idx}] 載入模型並設定 LoRA...")
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

    hf_dataset = Dataset.from_list(train_data_list)
    hf_dataset = hf_dataset.map(formatting_prompts_func, batched = True)

    print(f"[Fold {fold_idx}] 開始訓練...")
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
            warmup_steps = 10,
            num_train_epochs = 4, 
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
    return model, tokenizer, trainer

# ==========================================
# 4. 測試推論函數 (單一 Fold) + 受限解碼
# ==========================================
def evaluate_fold(model, tokenizer, test_data_list, fold_idx):
    print(f"\n[Fold {fold_idx}] 開始推論與驗證 ({len(test_data_list)} 筆測試資料)...")
    FastLanguageModel.for_inference(model)
    
    print(f"[Fold {fold_idx}] Building JSON Schema Parser for constrained decoding...")
    schema_dict = ResponseSchema.model_json_schema()
    parser = JsonSchemaParser(schema_dict)
    prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    
    correct_count = 0
    failed_reports = []

    for idx, item in enumerate(tqdm(test_data_list, desc=f"Evaluating Fold {fold_idx}")):
        conversations = item.get("conversations", [])
        human_input = next(c["value"] for c in conversations if c["from"] == "human")
        expected_str = next(c["value"] for c in conversations if c["from"] == "gpt")
        expected_json = json.loads(expected_str)
        
        messages = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": human_input}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        input_length = inputs.shape[1]
        outputs = model.generate(
            inputs, 
            max_new_tokens=1024, 
            use_cache=True, 
            temperature=0.1, 
            pad_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_function
        )
        
        response_str = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        actual_json = extract_json(response_str)
        
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
    print(f"[Fold {fold_idx}] 準確率: {accuracy*100:.1f}% ({correct_count}/{len(test_data_list)})")
    
    if failed_reports:
        with open(f"error_report_fold_{fold_idx}.json", 'w', encoding='utf-8') as f:
            json.dump(failed_reports, f, indent=2, ensure_ascii=False)
            
    return accuracy

# ==========================================
# 5. 主程式：5-Fold Cross-Validation 流程
# ==========================================
def main():
    print(f"載入所有資料自: {ALL_DATA_PATH}")
    with open(ALL_DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data), start=1):
        print("\n" + "="*50)
        print(f"🚀 開始執行 Fold {fold}/5")
        print("="*50)
        
        train_data = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]
        
        model, tokenizer, trainer = train_fold(train_data, fold)
        acc = evaluate_fold(model, tokenizer, test_data, fold)
        fold_accuracies.append(acc)
        
        print(f"[Fold {fold}] 清理 VRAM 記憶體...")
        del model
        del tokenizer
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("📊 5-Fold Cross-Validation 最終報告 (Constrained Decoding Enabled)")
    print("="*50)
    for i, acc in enumerate(fold_accuracies, start=1):
        print(f"Fold {i} 準確率 : {acc*100:.2f}%")
        
    avg_acc = sum(fold_accuracies) / len(fold_accuracies)
    print(f"🏆 平均準確率 (Average Accuracy): {avg_acc*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()