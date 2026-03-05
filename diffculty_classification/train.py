import json
import joblib
import numpy as np
import gc
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

# ==========================================
# 0. Global Configurations
# ==========================================
# 你的資料格式應該是: [{"text": "指令", "label": 0}, {"text": "複雜指令", "label": 1}, ...]
# 0: Simple (交給地端), 1: Complex (交給雲端)
ALL_DATA_PATH = "stage2_data.json" 
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
FINAL_MODEL_SAVE_PATH = "intent_router_svm_final.pkl"

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
    print(f"\n[Fold {fold_idx}] Extracting features (Embeddings) for training...")
    # 將訓練集文字轉換為向量
    X_train = embed_model.encode(train_texts, show_progress_bar=False)
    y_train = train_labels
    
    print(f"[Fold {fold_idx}] Training SVM Classifier...")
    # 使用 RBF kernel，開啟 probability=True 以便未來取 Confidence Score
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # (可選) 儲存每一折的模型
    joblib.dump(svm_model, f"svm_model_fold_{fold_idx}.pkl")
    
    return svm_model

# ==========================================
# 3. Inference and Evaluation Function (Single Fold)
# ==========================================
def evaluate_fold(svm_model, embed_model, test_texts, test_labels, fold_idx):
    print(f"\n[Fold {fold_idx}] Starting inference and validation ({len(test_texts)} test samples)...")
    
    # 將測試集文字轉換為向量
    X_test = embed_model.encode(test_texts, show_progress_bar=False)
    y_test = test_labels
    
    # 預測
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)
    
    correct_count = 0
    failed_reports = []

    # 逐筆比對並記錄錯誤
    for i in tqdm(range(len(test_texts)), desc=f"Evaluating Fold {fold_idx}"):
        expected = int(y_test[i])
        actual = int(y_pred[i])
        confidence = float(np.max(y_prob[i]))
        
        if expected == actual:
            correct_count += 1
        else:
            failed_reports.append({
                "input_text": test_texts[i],
                "expected_label": expected,
                "actual_predicted": actual,
                "confidence_score": round(confidence, 4),
                "reason": "SVM classified into wrong category"
            })
            
    accuracy = correct_count / len(test_texts)
    print(f"[Fold {fold_idx}] Accuracy: {accuracy*100:.1f}% ({correct_count}/{len(test_texts)})")
    
    # 將預測失敗的案例輸出成 JSON，方便分析 Hard cases
    if failed_reports:
        with open(f"error_report_svm_fold_{fold_idx}.json", 'w', encoding='utf-8') as f:
            json.dump(failed_reports, f, indent=2, ensure_ascii=False)
            
    return accuracy

# ==========================================
# 4. Main Program: 5-Fold Cross-Validation Pipeline
# ==========================================
def main():
    print(f"Loading all data from: {ALL_DATA_PATH}")
    texts, labels = load_data(ALL_DATA_PATH)
    
    if len(texts) == 0:
        print("Data loading failed. Exiting.")
        return

    print(f"Loading lightweight embedding model: {EMBED_MODEL_NAME}...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        
    # Setup 5-Fold split 
    # (分類任務強烈建議使用 StratifiedKFold 確保每一折的 0/1 比例平均)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
        print("\n" + "="*50)
        print(f"🚀 Starting Fold {fold}/5")
        print("="*50)
        
        # Split data
        train_texts, train_labels = texts[train_idx], labels[train_idx]
        test_texts, test_labels = texts[test_idx], labels[test_idx]
        
        print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")
        
        # 1. Train
        svm_model = train_fold(train_texts, train_labels, embed_model, fold)
        
        # 2. Evaluate
        acc = evaluate_fold(svm_model, embed_model, test_texts, test_labels, fold)
        fold_accuracies.append(acc)
        
        # 3. Free up memory
        print(f"[Fold {fold}] Clearing memory...")
        del svm_model
        gc.collect()

    # Output final report
    print("\n" + "="*50)
    print("📊 SVM Semantic Router 5-Fold Cross-Validation Final Report")
    print("="*50)
    for i, acc in enumerate(fold_accuracies, start=1):
        print(f"Fold {i} Accuracy : {acc*100:.2f}%")
        
    avg_acc = sum(fold_accuracies) / len(fold_accuracies)
    print(f"🏆 Average Accuracy: {avg_acc*100:.2f}%")
    print("="*50)
    
    # 5. Train Final Model on 100% of the data for deployment
    print("\nTraining final model on ALL data for deployment...")
    final_X = embed_model.encode(texts, show_progress_bar=True)
    final_svm = SVC(kernel='rbf', probability=True, random_state=42)
    final_svm.fit(final_X, labels)
    
    joblib.dump(final_svm, FINAL_MODEL_SAVE_PATH)
    print(f"✅ Final deployment model saved to {FINAL_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()