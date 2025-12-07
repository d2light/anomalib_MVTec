import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from anomalib.data.utils import ValSplitMode, TestSplitMode
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
TEST_ROOT = "./datasets/MVTecAD/capsule/test"
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

# 학습용 데이터셋: 양품(good)만 사용
train_datamodule = Folder(
    name=f"{CATEGORY}",
    root=DATA_ROOT,
    normal_dir="train/good",
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
    augmentations=resize_transform,
    # val_split_ratio=0,  # Validation split 없음
    test_split_mode=TestSplitMode.NONE,
    val_split_mode=ValSplitMode.SYNTHETIC,
    val_split_ratio=0.25,
)

train_datamodule.setup()
print(f"✅ 학습 데이터셋 로드 완료!")
print(f"   - 학습용 양품 이미지: {len(train_datamodule.train_dataloader().dataset)}개")

# Threshold/평가용 데이터셋: test 전체 (good + defects)
# TEST_ROOT 전체를 test로 사용하고, val은 test와 동일하게 설정
test_datamodule = Folder(
    name=f"{CATEGORY}_test",
    root=TEST_ROOT,          # ./capsule/test
    normal_dir="good",
    normal_test_dir="good",
    abnormal_dir=[
        "crack",
        "faulty_imprint",
        "poke",
        "scratch",
        "squeeze",
    ],
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
    augmentations=resize_transform,
    test_split_mode=TestSplitMode.FROM_DIR,   # TEST_ROOT 전체를 test로 사용
    val_split_mode=ValSplitMode.SAME_AS_TEST,    # test에서 val 분리 (하지만 ratio=0이므로 동일)
    seed=42,
)

test_datamodule.setup()

# 실제 파일 개수와 dataset 개수 비교
all_png = list(Path(TEST_ROOT).rglob("*.png"))
# test_dataloader 사용 (TEST_ROOT 전체가 test로 설정됨)
test_dataloader = test_datamodule.test_dataloader()
test_dataset = test_dataloader.dataset

print("✅ Threshold/평가용 데이터셋 로드 완료!")
print(f"   - 실제 png 파일 개수 (TEST_ROOT): {len(all_png)}")
print(f"   - test_dataset 크기             : {len(test_dataset)}개 이미지")
print(f"   - test 배치 수                  : {len(test_dataloader)}개 배치")
if len(test_dataset) < len(all_png):
    print(f"   ⚠️ 경고: 데이터셋 크기({len(test_dataset)})가 실제 파일 수({len(all_png)})보다 적습니다!")
    print(f"      차이: {len(all_png) - len(test_dataset)}개 파일이 누락되었습니다.")
    print(f"   💡 디버깅: test_datamodule 구조 확인 중...")
    if hasattr(test_datamodule, 'test_set'):
        print(f"      - test_set 크기: {len(test_datamodule.test_set) if hasattr(test_datamodule.test_set, '__len__') else 'N/A'}")
else:
    print(f"   ✅ 모든 이미지가 로드되었습니다!")

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
    pad_maps=False,
    evaluator=evaluator,
    pre_processor=pre_processor,
)

# ============================================
# 4. 학습 엔진 설정
# ============================================
epochs = 5

engine = Engine(
    max_epochs=epochs,
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

# 타임스탬프 생성 (학습 전에 생성하여 일관성 유지)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Run name 생성: 제품명_모델버전_년월일_시분초
# 주의: 학습 후 실제 버전이 다를 수 있으므로, 학습 후 확인 필요
run_name = f"{CATEGORY}_v{next_version}_{timestamp}"

print(f"📝 MLflow Run Name: {run_name}")
print(f"📦 예상 모델 버전: v{next_version}")
print(f"⏰ 타임스탬프: {timestamp}")

with mlflow.start_run(run_name=run_name):
    # 예상 버전을 태그로 저장
    mlflow.set_tag("expected_version", f"v{next_version}")
    mlflow.set_tag("timestamp", timestamp)
    # 하이퍼파라미터 로깅
    mlflow.log_params({
        "model": "EfficientAD",
        "category": CATEGORY,
        "model_size": "small",
        "lr": 0.0001,
        "weight_decay": 0.00001,
        "max_epochs": epochs,
        "image_size": "256x256",
    })
    
    # 모델 학습 (양품만 사용)
    print("🚀 모델 학습을 시작합니다... (양품 데이터만 사용)")
    engine.fit(datamodule=train_datamodule, model=model)
    
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
        mlflow.set_tag("actual_version", actual_version)  # 명확한 태그
        
        ckpt_path = latest_version / "weights" / "lightning" / "model.ckpt"
        if not ckpt_path.exists():
            ckpt_path = list(latest_version.glob("**/model.ckpt"))[0]
        
        # 체크포인트를 artifact로 저장
        mlflow.log_artifact(str(ckpt_path), "checkpoints")
        print(f"💾 체크포인트 저장: {ckpt_path}")
        print(f"📦 실제 모델 버전: {actual_version}")
        
        # 예상 버전과 실제 버전 비교 및 일치 여부 확인
        if actual_version_num != next_version:
            print(f"\n⚠️ 경고: 예상 버전(v{next_version})과 실제 버전({actual_version})이 다릅니다!")
            print(f"   - MLflow Run Name: {run_name}")
            print(f"   - 실제 모델 버전: {actual_version}")
            print(f"   - MLflow에서 'actual_version' 태그로 실제 버전을 확인하세요.")
            mlflow.set_tag("version_mismatch", "true")
            mlflow.set_tag("version_mismatch_info", f"expected_v{next_version}_actual_{actual_version}")
        else:
            print(f"✅ 버전 일치 확인: 예상 버전(v{next_version})과 실제 버전({actual_version})이 일치합니다.")
            mlflow.set_tag("version_mismatch", "false")
    
    # 모델 평가 (Test 데이터셋으로)
    print("📊 모델 평가 중... (Test 데이터셋 사용)")
    test_results = engine.test(datamodule=test_datamodule, model=model)
    
    # 평가 결과 로깅
    for result in test_results:
        for key, value in result.items():
            mlflow.log_metric(key, value)
    
    # ============================================
    # 6. Test 셋 전체 예측 및 Threshold 계산
    # ============================================
    print("🔍 Test 셋 전체 예측 중... (Threshold 계산용)")
    predictions = engine.predict(model=model, datamodule=test_datamodule)
    
    print(f"✅ 예측 완료! 총 {len(predictions)} 배치")
    
    # 예측 결과 수집
    y_test = []
    y_scores = []
    paths = []
    anomaly_images = []  # 불량으로 판단한 이미지들
    total_processed = 0
    
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
            total_processed += 1
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
        
        # DataFrame 생성 (초기 threshold로 예측 결과 포함)
        productTrue = ["OK" if i == 0 else "NG" for i in y_test]
        productPred = ["OK" if score < threshold else "NG" for score in y_scores]
        thresholdDf = pd.DataFrame({
            "product_true": productTrue,
            "product_pred": productPred,
            "y_scores": y_scores
        })
        
        # Good threshold: 실제 정상이고 예측도 정상인 샘플들의 최대 anomaly score
        # (모델이 올바르게 예측한 정상 샘플만 고려)
        goodDf = thresholdDf[(thresholdDf['product_true'] == 'OK') & (thresholdDf['product_pred'] == 'OK')]
        goodThreshold = goodDf["y_scores"].max() if len(goodDf) > 0 else None
        
        # Bad threshold: 실제 불량이고 예측도 불량인 샘플들의 최소 anomaly score
        # (모델이 올바르게 예측한 불량 샘플만 고려)
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
        
        # Threshold 정보를 JSON 파일로 저장 (모델 저장 경로에 함께 저장)
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
        
        # 모델 저장 경로에 threshold_info.json 저장
        if versions:
            threshold_json_path = latest_version / "threshold_info.json"
            with open(threshold_json_path, "w", encoding="utf-8") as f:
                json.dump(threshold_info, f, indent=2, ensure_ascii=False)
            print(f"💾 Threshold 정보 저장: {threshold_json_path}")
        
        # MLflow에도 artifact로 저장
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
    sample_batch = next(iter(test_datamodule.test_dataloader()))
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
