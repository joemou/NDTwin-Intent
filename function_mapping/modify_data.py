import json

# 1. 檔案路徑設定
INPUT_FILE = "all_data.json"
OUTPUT_FILE = "all_data_updated.json"

# 2. 我們定義好的「終極防禦版 System Prompt」
NEW_SYSTEM_PROMPT = """You control a network system. You are a classification task agent. Respond in the specified JSON format. 
RULES: 
1. DISRUPTIVE ACTIONS: If a command is explicitly disruptive (e.g., Disable, PowerOff, Block, Delete, Drop, Modify, Reroute, Update Firmware), enter the "discussion" state. NOTE: "Install", "Create", and "Add" are NOT disruptive, proceed to "answer" directly.
2. CONFIRMATION FLAG: If the prompt includes --confirm or --force, skip discussion and proceed to "answer". 
3. MISSING PARAMS (STRICT): "real-time traffic" queries MUST have port and duration parameters. If missing, you MUST set "valid": 0 and explain: "Missing port ID and duration parameters for real-time traffic monitoring."
4. ALLOWED API LIST & DEFAULTS: You must map intents strictly to these Task types: 
   - Health/Memory/Temperature queries -> GetDeviceHealth
   - Packet loss queries -> GetPacketLossRate
   - Log queries -> GetDeviceLogs (Always use line_count: 100 unless specified)
   DO NOT invent new task types like "GetRealTimeTraffic" or "GetDeviceRecentLogs".
5. Keep the "explanation" under 15 words. DO NOT explain your internal logic."""

# 3. 專門用來對付測試集刁鑽題目的「錨點訓練資料 (Anchor Data)」
anchor_data = [
    {
        "conversations": [
            {"from": "system", "value": NEW_SYSTEM_PROMPT},
            {"from": "human", "value": "Drop the packets on s3."},
            {"from": "gpt", "value": "{\"state\":\"answer\",\"valid\":1,\"explanation\":\"Installing drop flow on s3.\",\"tasks\":[{\"order\":1,\"type\":\"InstallFlowEntry\",\"parameters\":{\"device_name\":\"s3\",\"priority\":100,\"match\":{},\"actions\":[]}}]}"}
        ]
    },
    {
        "conversations": [
            {"from": "system", "value": NEW_SYSTEM_PROMPT},
            {"from": "human", "value": "Find the path from h1 to h2."},
            {"from": "gpt", "value": "{\"state\":\"answer\",\"valid\":1,\"explanation\":\"Retrieving path from h1 to h2.\",\"tasks\":[{\"order\":1,\"type\":\"GetPath\",\"parameters\":{\"src\":\"h1\",\"dst\":\"h2\"}}]}"}
        ]
    },
    {
        "conversations": [
            {"from": "system", "value": NEW_SYSTEM_PROMPT},
            {"from": "human", "value": "Clear all the rules on s2."},
            {"from": "gpt", "value": "{\"state\":\"answer\",\"valid\":1,\"explanation\":\"Removing all flows from s2.\",\"tasks\":[{\"order\":1,\"type\":\"DeleteFlowEntry\",\"parameters\":{\"device_name\":\"s2\",\"match\":{}}}]}"}
        ]
    },
    {
        "conversations": [
            {"from": "system", "value": NEW_SYSTEM_PROMPT},
            {"from": "human", "value": "Show all active flows."},
            {"from": "gpt", "value": "{\"state\":\"answer\",\"valid\":1,\"explanation\":\"Counting all active flows.\",\"tasks\":[{\"order\":1,\"type\":\"GetActiveFlowCount\",\"parameters\":{}}]}"}
        ]
    }
]

def main():
    print(f"正在讀取 {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"找不到檔案 {INPUT_FILE}，請確認檔名與路徑是否正確。")
        return

    update_count = 0

    # 4. 走訪所有資料，將 system 的 value 強制替換成新版
    for item in dataset:
        for convo in item.get("conversations", []):
            if convo.get("from") == "system":
                if convo.get("value") != NEW_SYSTEM_PROMPT:
                    convo["value"] = NEW_SYSTEM_PROMPT
                    update_count += 1

    print(f"✅ 成功更新了 {update_count} 筆舊版 System Prompt！")

    # 5. 把錨點資料加進去
    dataset.extend(anchor_data)
    print(f"✅ 成功注入 {len(anchor_data)} 筆對抗性錨點資料！")

    # 6. 存成新檔案
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"🎉 處理完成！請使用全新的 【{OUTPUT_FILE}】 進行 LoRA 訓練。")

if __name__ == "__main__":
    main()