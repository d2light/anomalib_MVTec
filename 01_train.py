from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from datetime import datetime
import json

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.metrics import AUROC, F1Score, Evaluator
from torchvision.transforms.v2 import Resize

from sklearn.metrics import (
    precision_recall_curve, f1_score, roc_curve, 
    accuracy_score, auc as auc_score
)

import mlflow
import mlflow.pytorch

# ============================================
# 1. 설정
# ============================================
DATA_ROOT = "./datasets/MVTecAD/capsule"
CATEGORY = "capsule"
RESULTS_DIR = "./results"
MLFLOW_TRACKING_URI = "./mlruns"

# MLflow 설정
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(f"EfficientAD_{CATEGORY}")

# ============================================
# 2. 데이터셋 설정
# ============================================
resize_transform = Resize(size=(256, 256))

datamodule = Folder(
    name=CATEGORY,
    root=DATA_ROOT,
    normal_dir="train/good",
    abnormal_dir=[
        "test/crack",
        "test/faulty_imprint",
        "test/poke",
        "test/scratch",
        "test/squeeze",
    ],
    normal_test_dir="test/good",
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
    augmentations=resize_transform,
)

datamodule.setup()

# ============================================
# 3. 모델 설정
# ============================================
test_metrics = [
    AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
    F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
]
evaluator = Evaluator(test_metrics=test_metrics)

pre_processor = EfficientAd.configure_pre_processor(image_size=(256, 256))

model = EfficientAd(
    teacher_out_channels=384,
    model_size="small",
    lr=0.0001,
    weight_decay=0.00001,
    padding=False,
    pad_maps=True,
    evaluator=evaluator,
    pre_processor=pre_processor,
)

# ============================================
# 4. 학습 엔진 설정
# ============================================
engine = Engine(
    max_epochs=40,
    accelerator="auto",
    devices=1,
    default_root_dir=RESULTS_DIR,
)

# ============================================
# 5. MLflow 학습 시작
# ============================================
# 모델 버전 확인 (학습 전 기존 버전 확인하여 다음 버전 예측)
results_path = Path(RESULTS_DIR) / "EfficientAd" / CATEGORY
next_version = 0
if results_path.exists():
    versions = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('v')]
    if versions:
        # 기존 버전 중 최대값 찾기
        version_numbers = [int(v.name[1:]) for v in versions if v.name[1:].isdigit()]
        if version_numbers:
            next_version = max(version_numbers) + 1
        else:
            next_version = 1
    else:
        next_version = 0
else:
    next_version = 0

# Run name 생성: 제품명_모델버전_년월일_시분초
run_name = f"{CATEGORY}_v{next_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"📝 MLflow Run Name: {run_name}")
print(f"📦 예상 모델 버전: v{next_version}")

with mlflow.start_run(run_name=run_name):
    # 하이퍼파라미터 로깅
    mlflow.log_params({
        "model": "EfficientAD",
        "category": CATEGORY,
        "model_size": "small",
        "lr": 0.0001,
        "weight_decay": 0.00001,
        "max_epochs": 40,
        "image_size": "256x256",
    })
    
    # 모델 학습
    engine.fit(datamodule=datamodule, model=model)
    
    # 학습 후 체크포인트 경로 찾기
    results_path = Path(RESULTS_DIR) / "EfficientAd" / CATEGORY
    versions = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('v')]
    if versions:
        latest_version = max(versions, key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 0)
        actual_version = latest_version.name  # v0, v1, v2 등
        actual_version_num = int(actual_version[1:]) if actual_version[1:].isdigit() else 0
        
        # 실제 버전을 태그로 저장
        mlflow.set_tag("model_version", actual_version)
        mlflow.set_tag("model_version_number", str(actual_version_num))
        
        ckpt_path = latest_version / "weights" / "lightning" / "model.ckpt"
        if not ckpt_path.exists():
            ckpt_path = list(latest_version.glob("**/model.ckpt"))[0]
        
        # 체크포인트를 artifact로 저장
        mlflow.log_artifact(str(ckpt_path), "checkpoints")
        print(f"💾 체크포인트 저장: {ckpt_path}")
        print(f"📦 실제 모델 버전: {actual_version}")
        
        # 예상 버전과 실제 버전이 다른 경우 알림
        if actual_version_num != next_version:
            print(f"⚠️ 예상 버전(v{next_version})과 실제 버전({actual_version})이 다릅니다.")
    
    # 모델 평가
    test_results = engine.test(datamodule=datamodule, model=model)
    
    # 평가 결과 로깅
    for result in test_results:
        for key, value in result.items():
            mlflow.log_metric(key, value)
    
    # ============================================
    # 6. Test 셋 전체 예측 및 Threshold 계산
    # ============================================
    print("🔍 Test 셋 전체 예측 중...")
    predictions = engine.predict(model=model, datamodule=datamodule)
    
    # 예측 결과 수집
    y_test = []
    y_scores = []
    paths = []
    anomaly_images = []  # 불량으로 판단한 이미지들
    
    def extract_defect_type(image_path: str) -> str:
        """이미지 경로에서 불량 유형 추출"""
        if image_path is None:
            return "unknown"
        path_str = str(image_path).replace("\\", "/")
        if "/test/crack" in path_str or "/crack/" in path_str:
            return "crack"
        elif "/test/faulty_imprint" in path_str or "/faulty_imprint/" in path_str:
            return "faulty_imprint"
        elif "/test/poke" in path_str or "/poke/" in path_str:
            return "poke"
        elif "/test/scratch" in path_str or "/scratch/" in path_str:
            return "scratch"
        elif "/test/squeeze" in path_str or "/squeeze/" in path_str:
            return "squeeze"
        elif "/test/good" in path_str or "/train/good" in path_str or "/good/" in path_str:
            return "good"
        else:
            return "unknown"
    
    for batch in predictions:
        batch_size = batch.image.shape[0]
        for i in range(batch_size):
            image_path = batch.image_path[i] if hasattr(batch, 'image_path') else None
            gt_label = batch.gt_label[i].item() if hasattr(batch, 'gt_label') else None
            pred_score = batch.pred_score[i].item() if hasattr(batch, 'pred_score') else None
            pred_label = batch.pred_label[i].item() if hasattr(batch, 'pred_label') else None
            
            if gt_label is not None and pred_score is not None:
                y_test.append(int(gt_label))
                y_scores.append(float(pred_score))
                paths.append(str(image_path) if image_path else f"unknown_{i}")
                
                # 불량으로 판단한 이미지 저장 (pred_label == 1, good 포함)
                if pred_label == 1:
                    defect_type = extract_defect_type(image_path)
                    img = batch.image[i].permute(1, 2, 0).cpu().numpy()
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    img = (img * 255).astype(np.uint8)
                    
                    # Anomaly map이 있으면 오버레이
                    if hasattr(batch, 'anomaly_map') and batch.anomaly_map is not None:
                        anomaly_map = batch.anomaly_map[i].squeeze().cpu().numpy()
                        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
                        
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        axes[0].imshow(img)
                        axes[0].set_title(f"Original\nScore: {pred_score:.3f}")
                        axes[0].axis('off')
                        
                        axes[1].imshow(anomaly_map, cmap='hot')
                        axes[1].set_title("Anomaly Map")
                        axes[1].axis('off')
                        
                        axes[2].imshow(img)
                        axes[2].imshow(anomaly_map, cmap='hot', alpha=0.5)
                        axes[2].set_title("Overlay")
                        axes[2].axis('off')
                        
                        plt.tight_layout()
                        img_name = Path(image_path).stem if image_path else f"unknown_{i}"
                        # 불량 유형별 폴더로 저장
                        save_path = f"anomaly_images/{defect_type}/{img_name}.png"
                        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        anomaly_images.append(save_path)
    
    # ============================================
    # 7. Threshold 계산 함수
    # ============================================
    def calculateThreshold(y_test: list, y_scores: list, path: list) -> tuple:
        """양품, 불량을 결정하는 threshold value를 결정하는 함수"""
        if 0 not in y_test:
            return None, None, None, None, None, None
        
        y_test = np.array(y_test)
        y_scores = np.array(y_scores)
        
        # Precision-Recall curve로 threshold 후보 찾기
        # anomalib: 양품=0, 불량=1, 높은 score=비정상
        # pos_label=1: 불량이 positive class
        thresholds = precision_recall_curve(y_test, y_scores, pos_label=1)[2]
        
        # F1 스코어를 최대화하는 threshold 값 찾기
        # anomalib: 양품=0, 불량=1, 높은 score=비정상
        # 따라서 score >= threshold면 비정상(1), score < threshold면 정상(0)
        f1Scores = [f1_score(y_test, (y_scores >= threshold).astype(int), pos_label=1) for threshold in thresholds]
        threshold = thresholds[np.argmax(f1Scores)]
        
        # AUROC 계산 (pos_label=1: 불량이 positive class)
        fpr, tpr = roc_curve(y_true=y_test, y_score=y_scores, pos_label=1)[:2]
        auc = auc_score(fpr, tpr) * 100
        
        # DataFrame 생성
        thresholdDf = pd.DataFrame({
            "product_true": ["OK" if i == 0 else "NG" for i in y_test],
            "y_scores": y_scores
        })
        
        # Good threshold: 실제 정상인 모든 샘플들의 최대 anomaly score
        goodDf = thresholdDf[thresholdDf['product_true'] == 'OK']
        goodThreshold = goodDf["y_scores"].max() if len(goodDf) > 0 else None
        
        # Bad threshold: 실제 불량인 모든 샘플들의 최소 anomaly score
        badDf = thresholdDf[thresholdDf['product_true'] == 'NG']
        badThreshold = badDf["y_scores"].min() if len(badDf) > 0 else None
        
        # Best threshold: good과 bad의 평균으로 산정
        if goodThreshold is not None and badThreshold is not None:
            bestThreshold = (goodThreshold + badThreshold) / 2
        elif goodThreshold is not None:
            # good만 있는 경우
            bestThreshold = goodThreshold
        elif badThreshold is not None:
            # bad만 있는 경우
            bestThreshold = badThreshold
        else:
            # 둘 다 없는 경우 fallback
            bestThreshold = threshold
        
        # Accuracy 계산
        # anomalib: 양품=0, 불량=1, 높은 score=비정상
        # 따라서 score >= threshold면 비정상(1), score < threshold면 정상(0)
        y_pred = [1 if i >= bestThreshold else 0 for i in y_scores]
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred) * 100
        
        return bestThreshold, goodThreshold, badThreshold, accuracy, auc, threshold
    
    # Threshold 계산
    bestThreshold, goodThreshold, badThreshold, accuracy, auc, initial_threshold = calculateThreshold(
        y_test, y_scores, paths
    )
    
    if bestThreshold is not None:
        # ============================================
        # 모델 내부 속성으로 threshold 저장
        # ============================================
        # register_buffer를 사용하여 state_dict에 포함 (모델 저장 시 함께 저장됨)
        model.register_buffer("best_threshold", torch.tensor(bestThreshold, dtype=torch.float32))
        if goodThreshold is not None:
            model.register_buffer("good_threshold", torch.tensor(goodThreshold, dtype=torch.float32))
        else:
            model.register_buffer("good_threshold", torch.tensor(0.0, dtype=torch.float32))
        
        if badThreshold is not None:
            model.register_buffer("bad_threshold", torch.tensor(badThreshold, dtype=torch.float32))
        else:
            model.register_buffer("bad_threshold", torch.tensor(0.0, dtype=torch.float32))
        
        # 추가 메타데이터를 모델 속성으로 저장 (state_dict에는 포함되지 않지만 모델 객체에 저장됨)
        model.threshold_accuracy = float(accuracy)
        model.threshold_auc = float(auc)
        model.threshold_category = CATEGORY
        model.initial_threshold = float(initial_threshold)
        
        print(f"✅ Threshold를 모델 속성으로 저장 완료:")
        print(f"  - best_threshold: {bestThreshold:.4f}")
        print(f"  - good_threshold: {goodThreshold:.4f}" if goodThreshold is not None else "  - good_threshold: None")
        print(f"  - bad_threshold: {badThreshold:.4f}" if badThreshold is not None else "  - bad_threshold: None")
        
        # Threshold 관련 메트릭 로깅
        mlflow.log_metrics({
            "best_threshold": bestThreshold,
            "good_threshold": goodThreshold if goodThreshold is not None else 0.0,
            "bad_threshold": badThreshold if badThreshold is not None else 0.0,
            "threshold_accuracy": accuracy,
            "threshold_auc": auc,
        })
        
        # Threshold 정보를 JSON 파일로 저장 (모델과 함께 저장)
        threshold_info = {
            "best_threshold": float(bestThreshold),
            "good_threshold": float(goodThreshold) if goodThreshold is not None else None,
            "bad_threshold": float(badThreshold) if badThreshold is not None else None,
            "threshold_accuracy": float(accuracy),
            "threshold_auc": float(auc),
            "initial_threshold": float(initial_threshold),
            "category": CATEGORY,
            "model_name": "EfficientAD",
        }
        with open("threshold_info.json", "w", encoding="utf-8") as f:
            json.dump(threshold_info, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact("threshold_info.json")
        
        # 예측 결과 DataFrame 저장
        # anomalib: 양품=0, 불량=1, 높은 score=비정상
        # 따라서 score >= threshold면 비정상(1), score < threshold면 정상(0)
        df = pd.DataFrame({
            "image_path": paths,
            "gt_label": y_test,
            "pred_score": y_scores,
            "pred_label": [1 if score >= bestThreshold else 0 for score in y_scores],
        })
        df.to_csv("predictions.csv", index=False, encoding='utf-8-sig')
        mlflow.log_artifact("predictions.csv")
    
    # ============================================
    # 8. 불량 이미지들을 Artifact로 저장 (불량 유형별 폴더 구조 유지)
    # ============================================
    if anomaly_images:
        # anomaly_images 폴더 전체를 artifact로 저장 (불량 유형별 폴더 구조 유지)
        mlflow.log_artifacts("anomaly_images", "anomaly_segmentation")
    
    # 모델 저장 (threshold 정보와 함께)
    from mlflow.models import infer_signature
    
    # Signature 생성용 샘플 데이터
    sample_batch = next(iter(datamodule.test_dataloader()))
    sample_input = sample_batch["image"][:1]  # 첫 번째 이미지만
    
    # 모델을 eval 모드로 설정하고 예측
    model.eval()
    with torch.no_grad():
        sample_output = model(sample_input)
    
    # Signature 추론 (output이 dict인 경우 처리)
    if isinstance(sample_output, dict):
        output_data = sample_output.get("pred_score", sample_output.get("anomaly_map", list(sample_output.values())[0]))
        if hasattr(output_data, "numpy"):
            output_data = output_data.numpy()
    else:
        output_data = sample_output.numpy() if hasattr(sample_output, "numpy") else sample_output
    
    signature = infer_signature(sample_input.numpy(), output_data)
    
    # Threshold 정보를 태그로 저장 (모델과 함께 추적 가능)
    if bestThreshold is not None:
        mlflow.set_tag("best_threshold", str(bestThreshold))
        if goodThreshold is not None:
            mlflow.set_tag("good_threshold", str(goodThreshold))
        if badThreshold is not None:
            mlflow.set_tag("bad_threshold", str(badThreshold))
        mlflow.set_tag("threshold_category", CATEGORY)
    
    # 모델 저장 (signature와 input_example 포함)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        signature=signature,
        input_example=sample_input.numpy(),
    )
    
    # 모델 내부 threshold 속성 확인
    if bestThreshold is not None:
        print("\n📊 모델 내부 threshold 속성 확인:")
        print(f"  - model.best_threshold: {model.best_threshold.item():.4f}")
        print(f"  - model.good_threshold: {model.good_threshold.item():.4f}")
        print(f"  - model.bad_threshold: {model.bad_threshold.item():.4f}")
        print(f"  - model.threshold_accuracy: {model.threshold_accuracy:.2f}%")
        print(f"  - model.threshold_auc: {model.threshold_auc:.2f}%")
        print(f"  - model.threshold_category: {model.threshold_category}")
        print("\n💡 모델 로드 시 threshold 사용 방법:")
        print("  loaded_model = mlflow.pytorch.load_model(model_uri)")
        print("  threshold = loaded_model.best_threshold.item()")
        print("  # 또는")
        print("  threshold = loaded_model.best_threshold.cpu().numpy()")
    
    print("\n✅ MLflow 로깅 완료!")

print("🎉 모든 작업 완료!")
