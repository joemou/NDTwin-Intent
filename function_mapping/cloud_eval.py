import json
import time
import os
from tqdm import tqdm
from openai import OpenAI
import google.generativeai as genai

# ==========================================
# Section 1: Global Settings & Pricing Models
# ==========================================
TEST_FILES = ["easytest.json", "hardtest.json"]
OUTPUT_ERROR_FILE = "cloud_error_report.json"

# Fetch API keys from environment variables securely
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

TARGET_MODEL = "gpt-4o"  # Note: Adjusted to a standard model name, change as needed

SYSTEM_PROMPT = """
System Prompt
You control a network system. You are a classification task agent. Respond in the specified JSON format.

1. PARAMETER AUTO-FILL & DEFAULTS (CRITICAL)
Implicit Defaults: If the user provides a target device but omits technical parameters, you MUST auto-fill them using these values and set "valid": 1. Do NOT mark as invalid for missing technical details.

InstallFlowEntry: priority: 100, match: {}, actions: [{"type": "OUTPUT", "port": 1}].

InstallGroupEntry: group_type: "all", group_id: 1, buckets: [{"actions": []}].

InstallMeterEntry: meter_id: 1, flags: [], bands: [{"type": "drop", "rate": 1000}].

GetDeviceLogs: line_count: 100.

PingHost: count: 4.

BackupConfiguration: backup_path: "/backup/{device_name}_config.bak".

Strict Requirement: Only GetRealTimeTraffic requires mandatory port_id and duration_seconds. If these two are missing, set "valid": 0.

2. DISRUPTIVE ACTIONS (State Logic)
RULE: If a command is explicitly disruptive, enter the "discussion" state to ask for confirmation.

Disruptive List: Disable, PowerOff, Block, Delete, Drop, Modify, Reroute, Update Firmware, Clear, Wipe, Nuke, Reset, Flush, Reboot, Restore.

Non-Disruptive: "Install", "Create", "Add", and all "Get/Show/List/Ping/Traceroute" are NOT disruptive; proceed to "answer" directly.

CONFIRMATION FLAG: If the prompt includes --confirm or --force, skip discussion and proceed to "answer".

3. TASK DEFINITIONS
[State & Power]: DisableSwitch(device_name), EnableSwitch(device_name), PowerOffSwitch(device_name), PowerOnSwitch(device_name), SetPortStatus(device_name, port_id, status), RestartDevice(device_name)

[Flow & Routing]: InstallFlowEntry(device_name, priority, match, actions), ModifyFlowEntry(device_name, priority, match, actions), DeleteFlowEntry(device_name, match), RerouteFlow(match, new_path), InstallGroupEntry(device_name, group_type, group_id, buckets), InstallMeterEntry(device_name, meter_id, flags, bands)

[Topology & Path]: GetNetworkTopology(), GetAllHosts(), GetPath(src, dst), GetPathSwitchCount(src, dst), GetSwitchPorts(device_name), GetSwitchCapabilities(), GetArpTable(device_name), GetMacTable(device_name)

[Telemetry & Stats]: GetTopKFlows(k), GetSwitchCpuUtilization(), GetTotalPowerConsumption(), GetASwitchCpuUtilization(device_name), GetASwitchPowerConsumption(device_name), GetALinkBandwidthUtilization(src, dst, port, duration), GetTopKCongestedLinks(k), GetTopKBandwidthUsers(k), GetActiveFlowCount(), GetFlowEntryCount(device_name), GetFlowEntries(device_name), GetLinkLatency(src, dst), GetPacketLossRate(src, dst), GetSwitchMemoryUtilization(device_name), GetSwitchTemperature(device_name), GetPortStatistics(device_name, port_id), GetDeviceHealth(device_name), GetRealTimeTraffic(device_name, port_id, duration_seconds)

[Utility & Security]: BlockHost(host_id), SetDeviceNickname(device_name, nickname), ToggleHistoricalLogging(state), GetDeviceUptime(device_name), BackupConfiguration(device_name, backup_path), RestoreConfiguration(device_name, restore_path), PingHost(host, count), TracerouteHost(host), GetDeviceLogs(device_name, line_count), ClearDeviceLogs(device_name), UpdateDeviceFirmware(device_name, firmware_path)

4. OUTPUT CONSTRAINTS
Explanation: Keep under 15 words. Do not explain logic.

Format: Strictly minified JSON. No whitespace, no markdown blocks, no comments.

Field Logic:

Discussion: {"state":"discussion","prompt":"..."} (Use when action is disruptive and no flag present).

Answer: {"state":"answer","valid":1,"explanation":"...","tasks":[{"order":1,"type":"...","parameters":{}}]} (Use for safe actions or disruptive actions with confirmation flag).
"""

# Pricing per 1M tokens in USD
PRICING = {
    "gpt-4o": {"input": 5.00, "output": 15.00}, 
    "gpt-5.4": {"input": 2.50, "output": 15.00}, 
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50}
}

# ==========================================
# Section 2: Parsing & Logic
# ==========================================
def extract_json(response_str):
    try:
        response_str = response_str.replace("```json", "").replace("```", "").strip()
        start_idx = response_str.find('{')
        end_idx = response_str.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            return json.loads(response_str[start_idx:end_idx])
        return None
    except Exception:
        return None

def compare_outputs(expected, actual):
    if actual is None: 
        return False, "Failed to parse into a valid JSON format"
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
# Section 3: Core API Call Handling
# ==========================================
def call_api(prompt):
    latency = 0
    cost = 0
    content = ""
    tokens_in = 0
    tokens_out = 0
    
    start_time = time.perf_counter()
    
    if "gpt" in TARGET_MODEL:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
            
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = resp.choices[0].message.content
        tokens_in = resp.usage.prompt_tokens
        tokens_out = resp.usage.completion_tokens
        
    else:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
            
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name=TARGET_MODEL, system_instruction=SYSTEM_PROMPT)
        resp = model.generate_content(prompt, generation_config={"temperature": 0.0})
        content = resp.text
        tokens_in = resp.usage_metadata.prompt_token_count
        tokens_out = resp.usage_metadata.candidates_token_count
        
    cost = (tokens_in / 1e6 * PRICING.get(TARGET_MODEL, {"input": 0})["input"]) + \
           (tokens_out / 1e6 * PRICING.get(TARGET_MODEL, {"output": 0})["output"])
               
    latency = time.perf_counter() - start_time
    return content, latency, cost, tokens_in, tokens_out

# ==========================================
# Section 4: Execution & Evaluation (Detailed Token Tracking)
# ==========================================
def main():
    # Dictionaries to store results and errors per file
    file_based_stats = {}
    categorized_errors = {}

    for test_file in TEST_FILES:
        if not os.path.exists(test_file):
            print(f"Warning: File '{test_file}' not found. Skipping...")
            continue

        # Initialize metrics for this specific file including tokens
        stats = {
            "correct": 0, 
            "processed": 0, 
            "latency": 0.0, 
            "cost": 0.0, 
            "tokens_in": 0, 
            "tokens_out": 0
        }
        categorized_errors[test_file] = []

        with open(test_file, "r", encoding="utf-8") as f:
            datasets = json.load(f)

        print(f"\n🚀 Testing Model: {TARGET_MODEL} | File: {test_file} ({len(datasets)} samples)")

        for item in tqdm(datasets, desc=f"Processing {test_file}"):
            convs = item.get("conversations", [])
            user_input = next(c["value"] for c in convs if c["from"] == "human")
            expected_str = next(c["value"] for c in convs if c["from"] == "gpt")
            expected_json = json.loads(expected_str)

            try:
                raw_res, lat, cost, t_in, t_out = call_api(user_input)
                actual_json = extract_json(raw_res)
                
                is_correct, reason = compare_outputs(expected_json, actual_json)
                
                # Update specific file stats with token counts
                stats["processed"] += 1
                stats["latency"] += lat
                stats["cost"] += cost
                stats["tokens_in"] += t_in
                stats["tokens_out"] += t_out
                
                if is_correct:
                    stats["correct"] += 1
                else:
                    categorized_errors[test_file].append({
                        "input": user_input, 
                        "reason": reason,
                        "expected": expected_json, 
                        "actual": actual_json
                    })
                
                time.sleep(0.5) # Rate limit protection
                
            except Exception as e:
                print(f"Error processing: {e}")
                categorized_errors[test_file].append({"input": user_input, "error": str(e)})

        file_based_stats[test_file] = stats

    # --- Final Report Rendering ---
    print("\n" + "═"*60)
    print(f"📊 DETAILED EVALUATION REPORT: {TARGET_MODEL}")
    print("═"*60)

    total_metrics = {
        "correct": 0, 
        "processed": 0, 
        "cost": 0.0, 
        "tokens_in": 0, 
        "tokens_out": 0
    }

    for file_name, s in file_based_stats.items():
        if s["processed"] > 0:
            acc = (s["correct"] / s["processed"]) * 100
            avg_lat = s["latency"] / s["processed"]
            
            print(f"📂 File: {file_name}")
            print(f"   Accuracy        : {acc:>6.2f}% ({s['correct']}/{s['processed']})")
            print(f"   Avg Latency     : {avg_lat:.4f}s")
            print(f"   Input Tokens    : {s['tokens_in']:,}")
            print(f"   Output Tokens   : {s['tokens_out']:,}")
            print(f"   Estimated Cost  : ${s['cost']:.6f}")
            print("-" * 40)
            
            # Aggregate for overall summary
            total_metrics["correct"] += s["correct"]
            total_metrics["processed"] += s["processed"]
            total_metrics["cost"] += s["cost"]
            total_metrics["tokens_in"] += s["tokens_in"]
            total_metrics["tokens_out"] += s["tokens_out"]

    # --- Overall Summary ---
    if total_metrics["processed"] > 0:
        overall_acc = (total_metrics["correct"] / total_metrics["processed"]) * 100
        print(f"🌍 OVERALL PERFORMANCE SUMMARY")
        print(f"   Total Queries   : {total_metrics['processed']}")
        print(f"   Overall Acc     : {overall_acc:.2f}%")
        print(f"   Total In Tokens : {total_metrics['tokens_in']:,}")
        print(f"   Total Out Tokens: {total_metrics['tokens_out']:,}")
        print(f"   Total Cost      : ${total_metrics['cost']:.6f} USD")
    print("═"*60)

    # Save categorized errors to JSON
    with open(OUTPUT_ERROR_FILE, "w", encoding="utf-8") as f:
        json.dump(categorized_errors, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Classified error report saved to: {OUTPUT_ERROR_FILE}")

if __name__ == "__main__":
    main()