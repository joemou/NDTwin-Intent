import json
import joblib
import numpy as np
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import time

# 深度學習與 BERT 相關
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from torch.nn.functional import softmax

# ==========================================
# 0. Global Configurations
# ==========================================
ALL_DATA_PATH = "stage2_data.json" 
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
BERT_MODEL_NAME = "distilbert-base-uncased" # 用於 Fine-tuning 的 Small BERT
CONFIDENCE_THRESHOLD = 0.80

# ==========================================
# 1. Helper Functions (Data Loading & Eval)
# ==========================================
def load_data(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        return np.array(texts), np.array(labels)
    except Exception as e:
        print(f"Error loading data: {e}")
        return np.array([]), np.array([])

def calculate_metrics(y_true, y_pred, y_prob_max, model_name, fold_idx, avg_latency_ms):
    raw_correct = 0
    total_filtered = 0
    errors_intercepted = 0
    fatal_errors = 0
    total = len(y_true)

    for i in range(total):
        is_correct = (y_true[i] == y_pred[i])
        is_low_conf = (y_prob_max[i] < CONFIDENCE_THRESHOLD)
        
        if is_correct:
            raw_correct += 1
            
        if is_low_conf:
            total_filtered += 1
            if not is_correct:
                errors_intercepted += 1
        else:
            if not is_correct:
                fatal_errors += 1
                
    raw_accuracy = raw_correct / total
    system_safe_rate = (raw_correct + errors_intercepted) / total
    overhead_rate = (total_filtered / total) if total > 0 else 0
    
    return {
        "Model": model_name,
        "Fold": fold_idx,
        "Raw_Accuracy": raw_accuracy,
        "System_Safe_Rate": system_safe_rate,
        "Latency_per_Sample_ms": avg_latency_ms,
        "Total_Filtered": total_filtered,
        "Errors_Intercepted": errors_intercepted,
        "Fatal_Errors": fatal_errors,
        "Overhead_Rate": overhead_rate
    }

# ==========================================
# 2. 訓練與推論函數 (SVM, RF, BERT)
# ==========================================
def train_eval_traditional(texts, y_all, train_idx, test_idx, embed_model, model_type, fold_idx):
    train_texts, y_train = texts[train_idx], y_all[train_idx]
    test_texts, y_test = texts[test_idx], y_all[test_idx]
    
    if model_type == "SVM":
        model = SVC(kernel='rbf', probability=True, random_state=42)
    else: 
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    print(f"[{model_type} - Fold {fold_idx}] Extracting training embeddings...")
    X_train_dense = embed_model.encode(train_texts, show_progress_bar=False)
    
    print(f"[{model_type} - Fold {fold_idx}] Training model...")
    model.fit(X_train_dense, y_train)
    
    print(f"[{model_type} - Fold {fold_idx}] Measuring inference time on test set...")
    start_time = time.perf_counter()
    
    # 包含 Embedding 與預測的總時間
    X_test_dense = embed_model.encode(test_texts, show_progress_bar=False)
    y_pred = model.predict(X_test_dense)
    y_prob = model.predict_proba(X_test_dense)
    
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    avg_latency_ms = total_time_ms / len(test_texts)
    
    y_prob_max = np.max(y_prob, axis=1)
    
    return calculate_metrics(y_test, y_pred, y_prob_max, model_type, fold_idx, avg_latency_ms)

def train_eval_bert(train_texts, train_labels, test_texts, test_labels, num_labels, fold_idx):
    print(f"[Small_BERT - Fold {fold_idx}] Fine-tuning BERT...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels})
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=num_labels)
    
    training_args = TrainingArguments(
        output_dir=f"./results_fold_{fold_idx}",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="no", # 已修正為新版 API
        save_strategy="no",
        logging_steps=50,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )
    
    trainer.train()
    
    print(f"[Small_BERT - Fold {fold_idx}] Measuring inference time...")
    start_time = time.perf_counter()
    
    predictions = trainer.predict(test_ds)
    logits = torch.tensor(predictions.predictions)
    probs = softmax(logits, dim=-1).numpy()
    
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    avg_latency_ms = total_time_ms / len(test_texts)
    
    y_pred = np.argmax(probs, axis=1)
    y_prob_max = np.max(probs, axis=1)
    
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return calculate_metrics(test_labels, y_pred, y_prob_max, "Small_BERT", fold_idx, avg_latency_ms)

# ==========================================
# 3. 圖表生成函數 (針對論文設計的高質感圖表)
# ==========================================
def generate_plots(df_full):
    # 設定論文繪圖風格
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # 提前計算好平均值，避免 seaborn 在畫圖時計算誤差棒導致版本報錯
    summary = df_full.groupby("Model").mean(numeric_only=True).reset_index()
    
    # --- 圖表 1: 原始準確率 vs 系統安全率 ---
    plt.figure(figsize=(10, 6))
    df_melted = summary.melt(id_vars=["Model"], value_vars=["Raw_Accuracy", "System_Safe_Rate"], 
                             var_name="Metric", value_name="Score")
    
    ax1 = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="muted", edgecolor=".2")
    plt.title("Fig 1. Performance: Raw Accuracy vs. System Safe Rate", pad=15, fontweight='bold')
    plt.ylabel("Score (%)")
    plt.xlabel("")
    
    # 動態調整 Y 軸，凸顯細微差異
    min_score = df_melted["Score"].min()
    plt.ylim(max(0, min_score - 0.05), 1.02) 
    
    # 標註數字
    for p in ax1.patches:
        height = p.get_height()
        if height > 0:
            ax1.annotate(f'{height*100:.1f}%', 
                         (p.get_x() + p.get_width() / 2., height), 
                         ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['Raw Accuracy', 'System Safe Rate (with Fallback)'], loc='lower right')
    plt.tight_layout()
    plt.savefig("paper_fig1_accuracy_safety_bar.png", dpi=300)
    print("📸 Saved plot: paper_fig1_accuracy_safety_bar.png")

    # --- 圖表 2: 效能與延遲散佈圖 (Trade-off) ---
    plt.figure(figsize=(8, 6))
    markers = {"SVM": "o", "Random_Forest": "s", "Small_BERT": "^"}
    colors = {"SVM": "#2ecc71", "Random_Forest": "#f1c40f", "Small_BERT": "#e74c3c"}
    
    for i, row in summary.iterrows():
        plt.scatter(row["Latency_per_Sample_ms"], row["System_Safe_Rate"], 
                    label=row["Model"], s=250, marker=markers[row["Model"]], 
                    color=colors[row["Model"]], edgecolor='black', zorder=5)
        
        plt.text(row["Latency_per_Sample_ms"], row["System_Safe_Rate"] - 0.005, 
                 row["Model"].replace("_", " "), ha='center', va='top', fontsize=11, fontweight='bold')

    plt.title("Fig 2. System Efficiency: Latency vs. Safety", pad=15, fontweight='bold')
    plt.xlabel("Average Inference Latency per Sample (ms) $\\rightarrow$ Faster is Better")
    plt.ylabel("System-Level Safe Rate $\\rightarrow$ Higher is Better")
    
    plt.xlim(0, summary["Latency_per_Sample_ms"].max() * 1.3)
    plt.ylim(max(0.9, summary["System_Safe_Rate"].min() - 0.02), 1.02)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("paper_fig2_latency_safety_scatter.png", dpi=300)
    print("📸 Saved plot: paper_fig2_latency_safety_scatter.png")

    # --- 圖表 3: Fallback 成本與致命錯誤分析 (雙 Y 軸) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左 Y 軸：Overhead
    sns.barplot(data=summary, x="Model", y="Overhead_Rate", color="#3498db", alpha=0.7, ax=ax1, edgecolor=".2")
    ax1.set_ylabel("Overhead Rate / Fallback to Cloud (%)", color="#2980b9", fontweight='bold')
    ax1.tick_params(axis='y', labelcolor="#2980b9")
    ax1.set_ylim(0, summary["Overhead_Rate"].max() * 1.3)
    
    for p in ax1.patches:
        height = p.get_height()
        ax1.annotate(f'{height*100:.1f}%', 
                     (p.get_x() + p.get_width() / 2., height), 
                     ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')

    # 右 Y 軸：Fatal Errors
    ax2 = ax1.twinx()
    # 使用 plot 而非 pointplot 避免某些 seaborn 版本的 Bug
    ax2.plot(range(len(summary)), summary["Fatal_Errors"], color="#c0392b", marker="D", markersize=8, linewidth=2.5, linestyle="-")
    ax2.set_ylabel("Average Fatal Errors (Count)", color="#c0392b", fontweight='bold')
    ax2.tick_params(axis='y', labelcolor="#c0392b")
    ax2.set_ylim(-0.2, max(summary["Fatal_Errors"].max() * 1.5, 2))
    ax2.grid(False) # 關閉右 Y 軸的網格避免視覺混亂
    
    plt.title("Fig 3. Defense Mechanism Analysis: Cost vs. Vulnerability", pad=15, fontweight='bold')
    # 替換 X 軸標籤的底線為空白
    ax1.set_xticklabels([m.replace("_", " ") for m in summary["Model"]])
    
    plt.tight_layout()
    plt.savefig("paper_fig3_fallback_analysis.png", dpi=300)
    print("📸 Saved plot: paper_fig3_fallback_analysis.png")

# ==========================================
# 4. Main Program
# ==========================================
def main():
    print(f"Loading data from: {ALL_DATA_PATH}")
    texts, raw_labels = load_data(ALL_DATA_PATH)
    
    if len(texts) == 0:
        print("Data loading failed. Please check JSON format.")
        return
        
    # 將字串標籤轉換為整數 (0, 1, 2...)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(raw_labels)
    
    label_mapping = {i: str(label) for i, label in enumerate(label_encoder.classes_)}
    with open("label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=4)
        
    num_labels = len(label_encoder.classes_)
    print(f"Found {num_labels} unique intents. Mapping saved to label_mapping.json")

    print(f"Loading lightweight embedding model: {EMBED_MODEL_NAME}...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
        print("\n" + "="*60)
        print(f"🚀 Starting Fold {fold}/5")
        print("="*60)
        
        train_texts, test_texts = texts[train_idx], texts[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]
        
        # 1. SVM 實驗
        res_svm = train_eval_traditional(texts, labels, train_idx, test_idx, embed_model, "SVM", fold)
        all_results.append(res_svm)
        
        # 2. Random Forest 實驗
        res_rf = train_eval_traditional(texts, labels, train_idx, test_idx, embed_model, "Random_Forest", fold)
        all_results.append(res_rf)
        
        # 3. Small BERT 實驗
        res_bert = train_eval_bert(train_texts, train_labels, test_texts, test_labels, num_labels, fold)
        all_results.append(res_bert)

    # ==========================================
    # 5. 輸出報告與圖表
    # ==========================================
    print("\n" + "="*60)
    print("📊 Final 5-Fold Cross-Validation Report Generated")
    print("="*60)
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("experiment_results_with_latency.csv", index=False)
    print("✅ Experiment metrics saved to: experiment_results_with_latency.csv")
    
    # 產生並儲存圖表 (這次絕對不會有 seaborn 的 errorbar 報錯了)
    generate_plots(df_results)
    
    summary = df_results.groupby("Model").mean(numeric_only=True).reset_index()
    summary = summary[["Model", "Raw_Accuracy", "System_Safe_Rate", "Latency_per_Sample_ms", "Overhead_Rate", "Fatal_Errors"]]
    print("\n--- Average Metrics Across 5 Folds ---")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()