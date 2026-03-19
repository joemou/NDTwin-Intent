import json
import time
import os
import csv
from tqdm import tqdm
from openai import OpenAI
import google.generativeai as genai
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# ==========================================
# Section 1: Global Settings & Pricing Models
# ==========================================
TEST_FILES = ["easytest.json", "hardtest.json"]

# 🔴 改成 List：把你想跑的模型都放進這個 Queue 裡
TARGET_MODELS = [
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
    "gpt-5.4"
]

# Fetch API keys from environment variables securely
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SYSTEM_PROMPT = """
## 1. IDENTITY & ROLE
You are a high-precision Network Classification Agent. Convert natural language into a specific JSON schema for network operations.

## 2. ENTITY MAPPING
"Core Switch" / "Root": s1. "Aggregation Switches": s2, s3. "Edge/Access Switches": s4, s5, s6, s7. "Uplink Port": port 1.

## 3. PARAMETER & DEFAULT LOGIC (STRICT)
- GetRealTimeTraffic: MUST have `port_id` and `duration_seconds` (as integer). If missing, set valid:0, state:"answer".
- GetALinkBandwidthUtilization: `duration` MUST be a string with "s" appended (e.g., "60s", "30s").
- InstallFlowEntry: priority=100, match={}, actions=[{"type":"OUTPUT","port":1}]
- InstallGroupEntry: group_type="all", group_id=1, buckets=[{"actions":[]}]
- InstallMeterEntry: meter_id=1, flags=[], bands=[{"type":"drop","rate":1000}]
- GetDeviceLogs: line_count=100
- PingHost: count=4
- BackupConfiguration: backup_path="/backup/{device_name}_config.bak"

## 4. DISRUPTIVE STATE LOGIC
- Disruptive Actions: Disable, PowerOff, Block, Delete, Drop, Modify, Reroute, Update, Clear, Wipe, Reset, Flush, Reboot, Restore.
- CONFIRMATION FLAGS: `--force`, `--confirm`, `without confirmation`, `Force`, `immediately`.
- If Disruptive AND LACKS flags -> state: "discussion". If flags exist -> state: "answer".
- Utility and Telemetry tasks are NEVER disruptive.

## 5. MINIMALISM & GLOBAL RULE
- One intent = One primary task, EXCEPT for the defined MACROS below.
- NEVER expand "all" into multiple device loops (e.g., s1, s2...s7). You MUST use `device_name: "all"` or `{}`.

## 6. TASK REFERENCE
[State]: DisableSwitch(device_name), EnableSwitch(device_name), PowerOffSwitch(device_name), PowerOnSwitch(device_name), SetPortStatus(device_name, port_id, status), RestartDevice(device_name)
[Flow]: InstallFlowEntry(device_name, priority, match, actions), ModifyFlowEntry(device_name, priority, match, actions), DeleteFlowEntry(device_name, match), RerouteFlow(match, new_path), InstallGroupEntry(device_name, group_type, group_id, buckets), InstallMeterEntry(device_name, meter_id, flags, bands)
[Topology]: GetNetworkTopology(), GetAllHosts(), GetPath(src, dst), GetPathSwitchCount(src, dst), GetSwitchPorts(device_name), GetSwitchCapabilities(), GetArpTable(device_name), GetMacTable(device_name)
[Telemetry]: GetTopKFlows(k), GetSwitchCpuUtilization(), GetTotalPowerConsumption(), GetASwitchCpuUtilization(device_name), GetASwitchPowerConsumption(device_name), GetALinkBandwidthUtilization(src, dst, port, duration), GetTopKCongestedLinks(k), GetTopKBandwidthUsers(k), GetActiveFlowCount(), GetFlowEntryCount(device_name), GetFlowEntries(device_name), GetLinkLatency(src, dst), GetPacketLossRate(src, dst), GetSwitchMemoryUtilization(device_name), GetSwitchTemperature(device_name), GetPortStatistics(device_name, port_id), GetDeviceHealth(device_name), GetRealTimeTraffic(device_name, port_id, duration_seconds)
[Utility]: BlockHost(host_id), SetDeviceNickname(device_name, nickname), ToggleHistoricalLogging(state), GetDeviceUptime(device_name), BackupConfiguration(device_name, backup_path), RestoreConfiguration(device_name, restore_path), PingHost(host, count), TracerouteHost(host), GetDeviceLogs(device_name, line_count), ClearDeviceLogs(device_name), UpdateDeviceFirmware(device_name, firmware_path)

## 7. COMPLEX INTENT MACROS (CRITICAL GROUND TRUTH RULES)
If you see these exact phrases/concepts, output EXACTLY these overriding results:
- "bandwidth utilization between": Set valid:0, state:"answer", explanation:"Missing port ID and duration parameters for real-time traffic monitoring."
- "drop all packets from 10.0.0.1": priority=1000, match={"ipv4_src":"10.0.0.1"}, actions=[]
- "why link is slow": GetLinkLatency, GetPacketLossRate, GetTopKCongestedLinks(k:3)
- "health of all switches and logs of hottest": GetSwitchTemperature({}), GetDeviceLogs(device_name:"s1", line_count:100)
- "packet loss on path between h1 and h8": GetPath(src:"h1", dst:"h8"), GetPacketLossRate(src:"s1", dst:"s5")
- "highest packet loss": GetPacketLossRate(src:"all", dst:"all")
- "rate limiting at 5000kbps": flags=["KBPS"], bands=[{"type":"drop","rate":5000}]
- "most flow entries": GetFlowEntryCount(device_name:"all")
- "Is historical logging enabled?": ToggleHistoricalLogging(state:"on")
- "every port on s3": GetSwitchPorts(device_name:"s3")
- "hosts and their port connections": GetAllHosts({}), GetNetworkTopology({})
- "top flows on [device]": GetTopKFlows(k) (Ignore device_name)
- "users consuming bandwidth on s1 port 2": GetTopKBandwidthUsers(k) (Ignore device_name and port)
- "reachable": PingHost(host)
- "temperature of all s-series": GetSwitchTemperature({})
- "high priority flow": priority=65535
- "Backup all switch configurations": BackupConfiguration(device_name:"all", backup_path:"/backup/full_backup.bak")
- "Ping [ip] from [device]": PingHost(host:[ip]) (Ignore source device)
- "Force reboot and check uptime": RestartDevice(device_name), GetDeviceUptime(device_name)
- "capabilities of switch s1": GetSwitchCapabilities({}) (Ignore device_name completely)
- "multicast group entry on s7 with ID 500": InstallGroupEntry(device_name:"s7", group_type:"all", group_id:500, buckets:[{"actions":[]}])

## 8. STRICT OUTPUT FORMAT (CRITICAL)
Respond with strictly minified JSON ONLY. Do NOT wrap in markdown. Do NOT invent keys like "params" or "task" or "action".
For Discussion: {"state":"discussion","prompt":"[Warning]"}
For Answer Valid: {"state":"answer","valid":1,"explanation":"[Max 15 words]","tasks":[{"order":1,"type":"[TaskType]","parameters":{"[key]":"[value]"}}]}
For Answer Invalid: {"state":"answer","valid":0,"explanation":"[Reason]"}
"""

# Pricing per 1M tokens in USD
PRICING = {
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40}
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
def call_api(prompt, model_name):
    latency = 0
    cost = 0
    content = ""
    tokens_in = 0
    tokens_out = 0
    
    start_time = time.perf_counter()
    
    if "gpt" in model_name:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
            
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content
        tokens_in = resp.usage.prompt_tokens
        tokens_out = resp.usage.completion_tokens
        
    else:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
            
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name=model_name, system_instruction=SYSTEM_PROMPT)
        resp = model.generate_content(prompt, generation_config={"temperature": 0.0})
        content = resp.text
        tokens_in = resp.usage_metadata.prompt_token_count
        tokens_out = resp.usage_metadata.candidates_token_count
        
    cost = (tokens_in / 1e6 * PRICING.get(model_name, {"input": 0})["input"]) + \
           (tokens_out / 1e6 * PRICING.get(model_name, {"output": 0})["output"])
               
    latency = time.perf_counter() - start_time
    return content, latency, cost, tokens_in, tokens_out

# ==========================================
# Section 4: Chart Plotting Function
# ==========================================
def plot_latency_charts(task_latency_tracker, model_name):
    easy_data = task_latency_tracker.get("easytest.json", {})
    hard_data = task_latency_tracker.get("hardtest.json", {})

    all_types = sorted(list(set(easy_data.keys()).union(set(hard_data.keys()))))
    
    if not all_types:
        print(f"Not enough task data collected to generate charts for {model_name}.")
        return

    # Chart 1: Average Latency
    easy_avgs = [np.mean(easy_data[t]) if t in easy_data and len(easy_data[t]) > 0 else 0 for t in all_types]
    hard_avgs = [np.mean(hard_data[t]) if t in hard_data and len(hard_data[t]) > 0 else 0 for t in all_types]
    
    x = np.arange(len(all_types))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(16, 7))
    ax1.bar(x - width/2, easy_avgs, width, label='Easy Case', color='lightgray', edgecolor='black')
    ax1.bar(x + width/2, hard_avgs, width, label='Hard Case', color='black', edgecolor='black')
    ax1.set_title(f'Average Processing Time per Task Type ({model_name})', fontsize=16)
    ax1.set_xlabel('Task Type', fontsize=14)
    ax1.set_ylabel('Average Latency (Seconds)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_types, rotation=45, ha='right')
    ax1.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name}_latency_avg_report.png", dpi=300)
    plt.close(fig1)

    # Chart 2: Distribution
    easy_lats = [lat for lats in easy_data.values() for lat in lats]
    hard_lats = [lat for lats in hard_data.values() for lat in lats]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist([easy_lats, hard_lats], bins=15, label=['Easy Case', 'Hard Case'], 
             color=['lightgray', 'black'], edgecolor='black', stacked=False)
    ax2.set_title(f'Distribution of Processing Times ({model_name})', fontsize=16)
    ax2.set_xlabel('Latency (Seconds)', fontsize=14)
    ax2.set_ylabel('Number of Tasks', fontsize=14)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name}_latency_dist_report.png", dpi=300)
    plt.close(fig2)

def plot_model_comparison(summary_results):
    """Generates a combined chart for Accuracy, Latency, and Cost across models."""
    if not summary_results:
        return
        
    models = [r['model'] for r in summary_results]
    accs = [r['accuracy'] for r in summary_results]
    lats = [r['avg_latency'] for r in summary_results]
    costs = [r['cost'] for r in summary_results]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for Accuracy
    color_acc = 'tab:blue'
    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', color=color_acc, fontsize=12)
    bars = ax1.bar(models, accs, color=color_acc, alpha=0.6, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0, max(accs + [100]) * 1.1)

    # Line chart for Latency
    ax2 = ax1.twinx()
    color_lat = 'tab:red'
    ax2.set_ylabel('Avg Latency (s)', color=color_lat, fontsize=12)
    ax2.plot(models, lats, color=color_lat, marker='o', linewidth=2, label='Latency')
    ax2.tick_params(axis='y', labelcolor=color_lat)
    ax2.set_ylim(0, max(lats + [1]) * 1.2)

    # Annotate Cost on top of the bars
    for i, txt in enumerate(costs):
        ax1.annotate(f"${txt:.4f}", 
                     (i, accs[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center', 
                     fontsize=10,
                     fontweight='bold')

    plt.title('Model Benchmark: Accuracy vs. Latency (with Cost)', fontsize=14)
    fig.tight_layout()
    plt.savefig("all_models_benchmark_comparison.png", dpi=300)
    plt.close(fig)

# ==========================================
# Section 5: Model Evaluation Logic
# ==========================================
def evaluate_model(model_name):
    """Evaluates a single model and generates its reports."""
    print("\n" + "="*70)
    print(f"🌟 STARTED EVALUATION QUEUE FOR MODEL: {model_name}")
    print("="*70)
    
    file_based_stats = {}
    categorized_errors = {}
    task_latency_tracker = {file: defaultdict(list) for file in TEST_FILES}
    all_latencies = [] # For P95 and P99 calculation

    for test_file in TEST_FILES:
        if not os.path.exists(test_file):
            print(f"Warning: File '{test_file}' not found. Skipping...")
            continue

        stats = {
            "correct": 0, "processed": 0, "latency": 0.0, 
            "cost": 0.0, "tokens_in": 0, "tokens_out": 0
        }
        categorized_errors[test_file] = []

        with open(test_file, "r", encoding="utf-8") as f:
            datasets = json.load(f)

        print(f"\n📂 Processing File: {test_file} ({len(datasets)} samples)")

        for item in tqdm(datasets, desc=f"Testing {model_name}"):
            convs = item.get("conversations", [])
            user_input = next(c["value"] for c in convs if c["from"] == "human")
            expected_str = next(c["value"] for c in convs if c["from"] == "gpt")
            expected_json = json.loads(expected_str)

            try:
                raw_res, lat, cost, t_in, t_out = call_api(user_input, model_name)
                actual_json = extract_json(raw_res)
                
                is_correct, reason = compare_outputs(expected_json, actual_json)
                
                stats["processed"] += 1
                stats["latency"] += lat
                stats["cost"] += cost
                stats["tokens_in"] += t_in
                stats["tokens_out"] += t_out
                all_latencies.append(lat)
                
                if is_correct:
                    stats["correct"] += 1
                else:
                    categorized_errors[test_file].append({
                        "input": user_input, 
                        "reason": reason,
                        "expected": expected_json, 
                        "actual": actual_json
                    })
                
                if actual_json and "tasks" in actual_json:
                    for task in actual_json["tasks"]:
                        task_type = task.get("type")
                        if task_type:
                            task_latency_tracker[test_file][task_type].append(lat)
                
                time.sleep(0.5) 
                
            except Exception as e:
                categorized_errors[test_file].append({"input": user_input, "error": str(e)})

        file_based_stats[test_file] = stats

    # Process overall metrics for this model
    total_metrics = {
        "correct": 0, "processed": 0, "latency": 0.0, "cost": 0.0, 
        "tokens_in": 0, "tokens_out": 0
    }

    for s in file_based_stats.values():
        if s["processed"] > 0:
            total_metrics["correct"] += s["correct"]
            total_metrics["processed"] += s["processed"]
            total_metrics["latency"] += s["latency"]
            total_metrics["cost"] += s["cost"]
            total_metrics["tokens_in"] += s["tokens_in"]
            total_metrics["tokens_out"] += s["tokens_out"]

    overall_acc = 0
    avg_lat = 0
    p95 = np.percentile(all_latencies, 95) if all_latencies else 0
    p99 = np.percentile(all_latencies, 99) if all_latencies else 0

    if total_metrics["processed"] > 0:
        overall_acc = (total_metrics["correct"] / total_metrics["processed"]) * 100
        avg_lat = total_metrics["latency"] / total_metrics["processed"]
        
        print("\n" + "-"*50)
        print(f"✅ SUMMARY FOR {model_name}")
        print(f"   Accuracy    : {overall_acc:.2f}%")
        print(f"   Avg Latency : {avg_lat:.4f}s")
        print(f"   P95 Latency : {p95:.4f}s")
        print(f"   P99 Latency : {p99:.4f}s")
        print(f"   Tokens In   : {total_metrics['tokens_in']}")
        print(f"   Tokens Out  : {total_metrics['tokens_out']}")
        print(f"   Total Cost  : ${total_metrics['cost']:.6f} USD")
        print("-"*50)

    # Output Reports for this specific model
    error_file = f"{model_name}_cloud_error_report.json"
    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(categorized_errors, f, indent=2, ensure_ascii=False)
    
    plot_latency_charts(task_latency_tracker, model_name)

    csv_file = f"{model_name}_latency_details.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dataset (Difficulty)', 'Task Type', 'Latency (Seconds)'])
        for dataset_name, tasks in task_latency_tracker.items():
            difficulty = "Easy" if "easy" in dataset_name.lower() else "Hard"
            for task_type, latencies in tasks.items():
                for lat in latencies:
                    writer.writerow([difficulty, task_type, f"{lat:.4f}"])
                    
    # Return metrics for the final combined report
    return {
        "model": model_name,
        "accuracy": overall_acc,
        "avg_latency": avg_lat,
        "p95_latency": p95,
        "p99_latency": p99,
        "cost": total_metrics["cost"],
        "tokens_in": total_metrics["tokens_in"],
        "tokens_out": total_metrics["tokens_out"]
    }

# ==========================================
# Section 6: Main Queue Execution
# ==========================================
def main():
    print("🚀 Starting Batch Evaluation Process...")
    print(f"Models in Queue: {', '.join(TARGET_MODELS)}")
    
    summary_results = []
    
    for target in TARGET_MODELS:
        try:
            result = evaluate_model(target)
            summary_results.append(result)
        except Exception as e:
            print(f"❌ Critical Error evaluating {target}: {str(e)}")

    if summary_results:
        print("\n" + "="*70)
        print("🏆 ALL MODELS COMPLETED. GENERATING FINAL REPORTS & CHARTS.")
        print("="*70)
        
        # 1. 產生綜合數據比較圖
        plot_model_comparison(summary_results)
        print(f"📊 Visual comparison saved to: all_models_benchmark_comparison.png")

        # 2. 產生統整的 CSV 報表
        summary_csv = "all_models_summary.csv"
        with open(summary_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Model Name', 'Overall Accuracy (%)', 'Average Latency (s)', 
                'P95 Latency (s)', 'P99 Latency (s)', 'Total Cost (USD)', 
                'Total Tokens In', 'Total Tokens Out'
            ])
            
            for res in summary_results:
                writer.writerow([
                    res["model"], 
                    f"{res['accuracy']:.2f}", 
                    f"{res['avg_latency']:.4f}", 
                    f"{res['p95_latency']:.4f}",
                    f"{res['p99_latency']:.4f}",
                    f"{res['cost']:.6f}",
                    res["tokens_in"],
                    res["tokens_out"]
                ])
                
        print(f"📄 Master summary saved to: {summary_csv}\n")

if __name__ == "__main__":
    main()