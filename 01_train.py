from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from datetime import datetime

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
    max_epochs=30,
    accelerator="auto",
    devices=1,
    default_root_dir=RESULTS_DIR,
)

# ============================================
# 5. MLflow 학습 시작
# ============================================
# Run name 생성: 제품종류_년월일_시분초
run_name = f"{CATEGORY}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name):
    # 하이퍼파라미터 로깅
    mlflow.log_params({
        "model": "EfficientAD",
        "category": CATEGORY,
        "model_size": "small",
        "lr": 0.0001,
        "weight_decay": 0.00001,
        "max_epochs": 1000,
        "image_size": "256x256",
    })
    
    # 모델 학습
    engine.fit(datamodule=datamodule, model=model)
    
    # 학습 후 체크포인트 경로 찾기
    results_path = Path(RESULTS_DIR) / "EfficientAd" / CATEGORY
    versions = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('v')]
    if versions:
        latest_version = max(versions, key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 0)
        ckpt_path = latest_version / "weights" / "lightning" / "model.ckpt"
        if not ckpt_path.exists():
            ckpt_path = list(latest_version.glob("**/model.ckpt"))[0]
        
        # 체크포인트를 artifact로 저장
        mlflow.log_artifact(str(ckpt_path), "checkpoints")
    
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
        thresholds = precision_recall_curve(y_test, y_scores, pos_label=0)[2]
        
        # F1 스코어를 최대화하는 threshold 값 찾기
        f1Scores = [f1_score(y_test, (y_scores <= threshold).astype(int), pos_label=0) for threshold in thresholds]
        threshold = thresholds[np.argmax(f1Scores)]
        
        # AUROC 계산
        fpr, tpr = roc_curve(y_true=y_test, y_score=y_scores, pos_label=0)[:2]
        auc = auc_score(fpr, tpr) * 100
        
        # DataFrame 생성
        productTrue = ["NG" if i == 0 else "OK" for i in y_test]
        productPred = ["NG" if score >= threshold else "OK" for score in y_scores]
        thresholdDf = pd.DataFrame({
            "product_true": productTrue,
            "product_pred": productPred,
            "y_scores": y_scores
        })
        
        # Good threshold (양품 최대 anomaly score)
        goodDf = thresholdDf[(thresholdDf['product_true'] == 'OK') & (thresholdDf['product_pred'] == 'OK')]
        goodThreshold = goodDf["y_scores"].max() if len(goodDf) > 0 else None
        
        # Bad threshold (불량 최저 anomaly score)
        badDf = thresholdDf[(thresholdDf['product_true'] == 'NG') & (thresholdDf['product_pred'] == 'NG')]
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
        y_pred = [0 if i >= bestThreshold else 1 for i in y_scores]
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred) * 100
        
        return bestThreshold, goodThreshold, badThreshold, accuracy, auc, threshold
    
    # Threshold 계산
    bestThreshold, goodThreshold, badThreshold, accuracy, auc, initial_threshold = calculateThreshold(
        y_test, y_scores, paths
    )
    
    if bestThreshold is not None:
        # Threshold 관련 메트릭 로깅
        mlflow.log_metrics({
            "best_threshold": bestThreshold,
            "good_threshold": goodThreshold if goodThreshold is not None else 0.0,
            "bad_threshold": badThreshold if badThreshold is not None else 0.0,
            "threshold_accuracy": accuracy,
            "threshold_auc": auc,
        })
        
        # 예측 결과 DataFrame 저장
        df = pd.DataFrame({
            "image_path": paths,
            "gt_label": y_test,
            "pred_score": y_scores,
            "pred_label": [0 if score >= bestThreshold else 1 for score in y_scores],
        })
        df.to_csv("predictions.csv", index=False, encoding='utf-8-sig')
        mlflow.log_artifact("predictions.csv")
    
    # ============================================
    # 8. 불량 이미지들을 Artifact로 저장 (불량 유형별 폴더 구조 유지)
    # ============================================
    if anomaly_images:
        # anomaly_images 폴더 전체를 artifact로 저장 (불량 유형별 폴더 구조 유지)
        mlflow.log_artifacts("anomaly_images", "anomaly_segmentation")
    
    # 모델 저장
    mlflow.pytorch.log_model(model, "model")
    
    print("✅ MLflow 로깅 완료!")

print("🎉 모든 작업 완료!")
