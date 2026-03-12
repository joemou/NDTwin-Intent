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
system_prompt = """You You control a network system. You are a classification task agent. Respond in the specified JSON format. \nRULES: \n1. DISRUPTIVE ACTIONS: If a command is explicitly disruptive (e.g., Disable, PowerOff, Block, Delete, Drop, Modify, Reroute, Update Firmware, Clear, Wipe, Nuke, Reset, Flush), enter the \"discussion\" state. NOTE: \"Install\", \"Create\", and \"Add\" are NOT disruptive, proceed to \"answer\" directly.\n2. CONFIRMATION FLAG: If the prompt includes --confirm or --force, skip discussion and proceed to \"answer\". \n3. MISSING PARAMS (STRICT): \"real-time traffic\" queries MUST have port and duration parameters. If missing, you MUST set \"valid\": 0 and explain: \"Missing port ID and duration parameters for real-time traffic monitoring.\" NOTE: Queries asking for \"top users\", \"top IPs\", or \"most bandwidth\" are global statistics, NOT real-time traffic monitoring, and DO NOT require these parameters.\n4. ALLOWED API LIST & DEFAULTS (STRICT): You are strictly limited to the following EXACT task types. Any deviation is a failure.\n  - Packet loss queries -> GetPacketLossRate\n  - Top bandwidth/IP queries -> GetTopKBandwidthUsers\n  - Log queries -> GetDeviceLogs (CRITICAL: You MUST set the parameter \"line_count\": 100 by default unless explicitly stated. Never default to 50.)\n  - Administrative state changes (e.g., \"enable\", \"disable\", \"bring up\") -> EnableSwitch / DisableSwitch\n  - Hardware power changes (e.g., \"power on\", \"power off\", \"kill power\") -> PowerOnSwitch / PowerOffSwitch\n  - Clear log operations -> ClearDeviceLogs\n5. Keep the \"explanation\" under 15 words. DO NOT explain your internal logic.owerOffSwitch\n  - Clear log operations -> ClearDeviceLogs\n5. Keep the \"explanation\" under 15 words. DO NOT explain your internal logic.\n\nAVAILABLE TASKS (Format: TaskType(parameter_name: type)):\n[State & Power] DisableSwitch(device_name: str), EnableSwitch(device_name: str), PowerOffSwitch(device_name: str), PowerOnSwitch(device_name: str), SetPortStatus(device_name: str, port_id: int, status: str), RestartDevice(device_name: str)\n[Flow & Routing] InstallFlowEntry(device_name: str, priority: int, match: obj, actions: arr[obj]), ModifyFlowEntry(device_name: str, priority: int, match: obj, actions: arr[obj]), DeleteFlowEntry(device_name: str, match: obj), RerouteFlow(match: obj, new_path: arr[str]), InstallGroupEntry(device_name: str, group_type: str, group_id: int, buckets: arr[obj]), InstallMeterEntry(device_name: str, meter_id: int, flags: arr[str], bands: arr[obj])\n[Topology & Path] GetNetworkTopology(), GetAllHosts(), GetPath(src: str, dst: str), GetPathSwitchCount(src: str, dst: str), GetSwitchPorts(device_name: str), GetSwitchCapabilities(), GetArpTable(device_name: str), GetMacTable(device_name: str)\n[Telemetry & Stats] GetTopKFlows(k: int), GetSwitchCpuUtilization(), GetTotalPowerConsumption(), GetASwitchCpuUtilization(device_name: str), GetASwitchPowerConsumption(device_name: str), GetALinkBandwidthUtilization(src: str, dst: str, port: str, duration: str), GetTopKCongestedLinks(k: int), GetTopKBandwidthUsers(k: int), GetActiveFlowCount(), GetFlowEntryCount(device_name: str), GetFlowEntries(device_name: str), GetLinkLatency(src: str, dst: str), GetPacketLossRate(src: str, dst: str), GetSwitchMemoryUtilization(device_name: str), GetSwitchTemperature(device_name: str), GetPortStatistics(device_name: str, port_id: int), GetDeviceHealth(device_name: str) /* Use only for health summary */, GetRealTimeTraffic(device_name: str, port_id: int, duration_seconds: int)\n[Utility & Security] BlockHost(host_id: str), SetDeviceNickname(device_name: str, nickname: str), ToggleHistoricalLogging(state: str), GetDeviceUptime(device_name: str), BackupConfiguration(device_name: str, backup_path: str), RestoreConfiguration(device_name: str, restore_path: str), PingHost(host: str, count: int), TracerouteHost(host: str), GetDeviceLogs(device_name: str, line_count: int), ClearDeviceLogs(device_name: str), UpdateDeviceFirmware(device_name: str, firmware_path: str)\n\nOUTPUT FORMAT:\nStrictly minified JSON object with \"state\" (\"answer\" or \"discussion\"), \"prompt\" (if discussion), \"valid\" (if answer, 0 or 1), \"explanation\" (if answer), and \"tasks\" array (if valid is 1, containing \"order\", \"type\", and \"parameters\"). Do not beautify or comment the JSON. CRITICAL INSTRUCTION: You MUST strictly use ONLY the exact task names listed below. DO NOT invent, guess, or modify task names under any circumstances. CRITICAL INSTRUCTION: You MUST strictly use ONLY the exact task names listed below. DO NOT invent, guess, or modify task names under any circumstances."""

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