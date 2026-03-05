import json
import joblib
import numpy as np
import gc
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer

# ==========================================
# 0. Global Configurations
# ==========================================
ALL_DATA_PATH = "stage2_data.json" 
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
FINAL_MODEL_SAVE_PATH = "intent_router_svm_final.pkl"
CONFIDENCE_THRESHOLD = 0.80  # 🚨 Core defense mechanism: Confidence fallback threshold

# ==========================================
# 1. Helper Functions (Data Loading)
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

# ==========================================
# 2. Training Function (Single Fold)
# ==========================================
def train_fold(train_texts, train_labels, embed_model, fold_idx):
    print(f"[Fold {fold_idx}] Extracting features (Embeddings) for training...")
    X_train = embed_model.encode(train_texts, show_progress_bar=False)
    y_train = train_labels
    
    print(f"[Fold {fold_idx}] Training SVM Classifier...")
    # Using RBF kernel with probability=True to extract Confidence Scores later
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Save the model for this specific fold
    joblib.dump(svm_model, f"svm_model_fold_{fold_idx}.pkl")
    
    return svm_model

# ==========================================
# 3. Inference and Evaluation Function (Single Fold)
# ==========================================
def evaluate_fold_with_fallback(svm_model, embed_model, test_texts, test_labels, fold_idx):
    print(f"[Fold {fold_idx}] Starting inference and validation ({len(test_texts)} test samples)...")
    
    X_test = embed_model.encode(test_texts, show_progress_bar=False)
    y_test = test_labels
    
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)
    
    raw_correct = 0
    total_filtered = 0       # Total samples routed to cloud because Conf < 0.80
    errors_intercepted = 0   # Samples that SVM got WRONG, but were safely routed to cloud
    fatal_errors = []        # Samples SVM got WRONG with Conf >= 0.80 (Failed defense)

    for i in tqdm(range(len(test_texts)), desc=f"Evaluating Fold {fold_idx}"):
        expected = int(y_test[i])
        actual = int(y_pred[i])
        confidence = float(np.max(y_prob[i]))
        
        is_correct = (expected == actual)
        is_low_conf = (confidence < CONFIDENCE_THRESHOLD)
        
        if is_correct:
            raw_correct += 1
            
        # 💡 Fallback Logic Evaluation
        if is_low_conf:
            total_filtered += 1
            if not is_correct:
                errors_intercepted += 1 # Successfully caught an error!
        else:
            if not is_correct:
                fatal_errors.append({
                    "input_text": test_texts[i],
                    "expected_label": expected,
                    "actual_predicted": actual,
                    "confidence_score": round(confidence, 4),
                    "reason": "High confidence but wrong prediction (Fatal Error)"
                })
                
    total = len(test_texts)
    raw_accuracy = raw_correct / total
    system_safe_rate = (raw_correct + errors_intercepted) / total 
    
    print(f"\n[Fold {fold_idx}] 📊 Test Results:")
    print(f"  ▶ Raw SVM Accuracy: {raw_accuracy*100:.1f}% ({raw_correct}/{total})")
    print(f"  ▶ 🛡️ Fallback Triggered: {total_filtered} samples filtered (Confidence < {CONFIDENCE_THRESHOLD})")
    print(f"      ↳ Errors successfully intercepted among filtered: {errors_intercepted} samples")
    print(f"  ▶ ⚠️ Fatal Errors (Wrong & High Conf): {len(fatal_errors)} samples")
    print(f"  ▶ 🌟 System-Level Safe Rate: {system_safe_rate*100:.1f}%")
    
    # Export the fatal errors to JSON for hard-case analysis
    if fatal_errors:
        with open(f"fatal_errors_fold_{fold_idx}.json", 'w', encoding='utf-8') as f:
            json.dump(fatal_errors, f, indent=2, ensure_ascii=False)
            
    return raw_accuracy, system_safe_rate, total_filtered, errors_intercepted

# ==========================================
# 4. Main Program: 5-Fold Cross-Validation Pipeline
# ==========================================
def main():
    print(f"Loading all data from: {ALL_DATA_PATH}")
    texts, labels = load_data(ALL_DATA_PATH)
    total_samples = len(texts)
    
    if total_samples == 0:
        print("Data loading failed. Exiting.")
        return

    print(f"Loading lightweight embedding model: {EMBED_MODEL_NAME}...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    raw_acc_list = []
    sys_safe_list = []
    total_filtered_list = []
    total_intercepted_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
        print("\n" + "="*60)
        print(f"🚀 Starting Fold {fold}/5")
        print("="*60)
        
        train_texts, train_labels = texts[train_idx], labels[train_idx]
        test_texts, test_labels = texts[test_idx], labels[test_idx]
        
        print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")
        
        # 1. Train
        svm_model = train_fold(train_texts, train_labels, embed_model, fold)
        
        # 2. Evaluate with Fallback
        raw_acc, sys_safe, filtered, intercepted = evaluate_fold_with_fallback(
            svm_model, embed_model, test_texts, test_labels, fold
        )
        
        raw_acc_list.append(raw_acc)
        sys_safe_list.append(sys_safe)
        total_filtered_list.append(filtered)
        total_intercepted_list.append(intercepted)
        
        # 3. Free up memory
        print(f"\n[Fold {fold}] Clearing memory...")
        del svm_model
        gc.collect()

    # ==========================================
    # 5. Output Final Report
    # ==========================================
    print("\n" + "="*60)
    print(f"📊 SVM Semantic Router (Conf Thresh: {CONFIDENCE_THRESHOLD}) - Final Report")
    print("="*60)
    
    for i in range(5):
        print(f"Fold {i+1}: Raw Acc {raw_acc_list[i]*100:.2f}% | Safe Rate {sys_safe_list[i]*100:.2f}% | Filtered: {total_filtered_list[i]} (Intercepted {total_intercepted_list[i]} errors)")
        
    avg_raw_acc = np.mean(raw_acc_list)
    avg_sys_safe = np.mean(sys_safe_list)
    
    total_filtered_sum = sum(total_filtered_list)
    total_intercepted_sum = sum(total_intercepted_list)
    false_alarms = total_filtered_sum - total_intercepted_sum
    
    # Calculate Trade-off Metrics
    overhead_rate = (false_alarms / total_samples) * 100 if total_samples > 0 else 0
    fallback_precision = (total_intercepted_sum / total_filtered_sum) * 100 if total_filtered_sum > 0 else 0
    
    print("-" * 60)
    print(f"🔹 Average Raw SVM Accuracy               : {avg_raw_acc*100:.2f}%")
    print(f"🔹 Average System Safe Rate               : {avg_sys_safe*100:.2f}%")
    print(f"✨ Safety Improvement from Architecture   : +{(avg_sys_safe - avg_raw_acc)*100:.2f}%")
    print("-" * 60)
    print("⚖️  Trade-off & Cost Analysis:")
    print(f"🛡️ Total samples filtered to cloud        : {total_filtered_sum} / {total_samples}")
    print(f"🎯 Total SVM errors successfully caught   : {total_intercepted_sum}")
    print(f"⚠️ False Alarms (Correct but sent to cloud) : {false_alarms}")
    print(f"   ↳ System Overhead Rate                 : {overhead_rate:.2f}%")
    print(f"   ↳ Fallback Precision                   : {fallback_precision:.2f}%")
    print("=" * 60)
    
    # ==========================================
    # 6. Train Final Model on 100% Data
    # ==========================================
    print("\nTraining final model on ALL data for deployment...")
    final_X = embed_model.encode(texts, show_progress_bar=True)
    final_svm = SVC(kernel='rbf', probability=True, random_state=42)
    final_svm.fit(final_X, labels)
    
    joblib.dump(final_svm, FINAL_MODEL_SAVE_PATH)
    print(f"✅ Final deployment model saved to {FINAL_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()