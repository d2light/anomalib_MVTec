from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import json

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.metrics import AUROC, F1Score, Evaluator
from torchvision.transforms.v2 import Resize

# PyTorch 2.6+ 호환성: PreProcessor 클래스를 안전한 글로벌로 추가
from anomalib.pre_processing.pre_processor import PreProcessor
torch.serialization.add_safe_globals([PreProcessor])

print("✅ 라이브러리 임포트 완료!")

# ============================================
# 1. 설정
# ============================================
DATA_ROOT = "./datasets/MVTecAD/capsule"
CATEGORY = "capsule"
RESULTS_DIR = "./results"
OUTPUT_DIR = "./predictions"

# 체크포인트 경로 찾기 (가장 최신 버전)
results_path = Path(RESULTS_DIR) / "EfficientAd" / CATEGORY
if results_path.exists():
    # v0, v1, v2... 중 가장 최신 버전 찾기
    versions = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('v')]
    if versions:
        latest_version = max(versions, key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 0)
        ckpt_path = latest_version / "weights" / "lightning" / "model.ckpt"
        if not ckpt_path.exists():
            # 다른 경로 시도
            ckpt_path = list(latest_version.glob("**/model.ckpt"))
            if ckpt_path:
                ckpt_path = ckpt_path[0]
            else:
                raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {results_path}")
    else:
        raise FileNotFoundError(f"버전 폴더를 찾을 수 없습니다: {results_path}")
else:
    raise FileNotFoundError(f"결과 폴더를 찾을 수 없습니다: {results_path}")

print(f"📁 체크포인트 경로: {ckpt_path}")

# ============================================
# 1-1. Threshold 정보 로드 (모델 저장 경로에서)
# ============================================
threshold_info = None
threshold_json_path = ckpt_path.parent.parent / "threshold_info.json"
if not threshold_json_path.exists():
    # 다른 경로 시도 (체크포인트와 같은 디렉토리)
    threshold_json_path = ckpt_path.parent / "threshold_info.json"
    if not threshold_json_path.exists():
        # 버전 디렉토리에서 찾기
        version_dir = ckpt_path.parent.parent.parent if "weights" in str(ckpt_path) else ckpt_path.parent.parent
        threshold_json_path = version_dir / "threshold_info.json"

if threshold_json_path.exists():
    with open(threshold_json_path, "r", encoding="utf-8") as f:
        threshold_info = json.load(f)
    print(f"✅ Threshold 정보 로드: {threshold_json_path}")
    print(f"   - Best Threshold: {threshold_info.get('best_threshold', 'N/A')}")
    print(f"   - Good Threshold: {threshold_info.get('good_threshold', 'N/A')}")
    print(f"   - Bad Threshold: {threshold_info.get('bad_threshold', 'N/A')}")
else:
    print(f"⚠️ Threshold 정보 파일을 찾을 수 없습니다: {threshold_json_path}")
    print(f"   모델 내부 threshold 정보를 확인합니다...")

# ============================================
# 2. 데이터셋 로드 (테스트용)
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
print("✅ 테스트 데이터셋 로드 완료!")

# ============================================
# 3. 모델 및 엔진 설정
# ============================================
# 평가 지표 설정
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

engine = Engine(
    accelerator="auto",
    devices=1,
)

print("✅ 모델 및 엔진 설정 완료!")

# ============================================
# 4. 체크포인트에서 가중치 로드
# ============================================
print(f"📦 체크포인트에서 가중치 로드 중: {ckpt_path}")


try:
    checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
except TypeError:
    checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)

# PyTorch Lightning 체크포인트에서 state_dict 추출
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    # PyTorch Lightning은 'model.' prefix를 사용하므로 제거
    state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v 
                  for k, v in state_dict.items()}
elif isinstance(checkpoint, dict):
    # state_dict가 직접 있는 경우
    state_dict = checkpoint
else:
    raise ValueError(f"체크포인트 형식을 인식할 수 없습니다: {type(checkpoint)}")

# 모델에 가중치 로드
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
if missing_keys:
    print(f"⚠️ 누락된 키: {len(missing_keys)}개 (일부는 정상일 수 있습니다)")
if unexpected_keys:
    print(f"⚠️ 예상치 못한 키: {len(unexpected_keys)}개")

model.eval()

print("✅ 가중치 로드 완료!")

# ============================================
# 모델 내부 threshold 정보 확인 (JSON 우선, 없으면 모델 내부)
# ============================================
print("\n" + "="*60)
print("📊 Threshold 정보 확인")
print("="*60)

# JSON에서 읽은 threshold 정보 우선 사용
if threshold_info:
    best_threshold = threshold_info.get('best_threshold')
    good_threshold = threshold_info.get('good_threshold')
    bad_threshold = threshold_info.get('bad_threshold')
    threshold_accuracy = threshold_info.get('threshold_accuracy')
    threshold_auc = threshold_info.get('threshold_auc')
    threshold_category = threshold_info.get('category')
    
    print("📄 Threshold 정보 출처: JSON 파일")
    if best_threshold is not None:
        print(f"✅ Best Threshold: {best_threshold:.6f}")
    else:
        print("❌ Best Threshold: 저장되지 않음")
    
    if good_threshold is not None and good_threshold > 0:
        print(f"✅ Good Threshold: {good_threshold:.6f}")
    elif good_threshold is not None:
        print(f"⚠️ Good Threshold: {good_threshold:.6f} (설정되지 않음)")
    else:
        print("❌ Good Threshold: 저장되지 않음")
    
    if bad_threshold is not None and bad_threshold > 0:
        print(f"✅ Bad Threshold: {bad_threshold:.6f}")
    elif bad_threshold is not None:
        print(f"⚠️ Bad Threshold: {bad_threshold:.6f} (설정되지 않음)")
    else:
        print("❌ Bad Threshold: 저장되지 않음")
    
    if threshold_accuracy is not None:
        print(f"📈 Threshold Accuracy: {threshold_accuracy:.2f}%")
    if threshold_auc is not None:
        print(f"📈 Threshold AUROC: {threshold_auc:.2f}%")
    if threshold_category:
        print(f"📦 Category: {threshold_category}")
else:
    # JSON이 없으면 모델 내부 threshold 확인
    print("📄 Threshold 정보 출처: 모델 내부")
    
    if hasattr(model, 'best_threshold'):
        best_threshold = model.best_threshold.item()
        print(f"✅ Best Threshold: {best_threshold:.6f}")
    else:
        best_threshold = None
        print("❌ Best Threshold: 저장되지 않음")
    
    if hasattr(model, 'good_threshold'):
        good_threshold = model.good_threshold.item()
        if good_threshold > 0:
            print(f"✅ Good Threshold: {good_threshold:.6f}")
        else:
            print(f"⚠️ Good Threshold: {good_threshold:.6f} (설정되지 않음)")
    else:
        good_threshold = None
        print("❌ Good Threshold: 저장되지 않음")
    
    if hasattr(model, 'bad_threshold'):
        bad_threshold = model.bad_threshold.item()
        if bad_threshold > 0:
            print(f"✅ Bad Threshold: {bad_threshold:.6f}")
        else:
            print(f"⚠️ Bad Threshold: {bad_threshold:.6f} (설정되지 않음)")
    else:
        bad_threshold = None
        print("❌ Bad Threshold: 저장되지 않음")
    
    if hasattr(model, 'threshold_accuracy'):
        threshold_accuracy = model.threshold_accuracy
        print(f"📈 Threshold Accuracy: {threshold_accuracy:.2f}%")
    if hasattr(model, 'threshold_auc'):
        threshold_auc = model.threshold_auc
        print(f"📈 Threshold AUROC: {threshold_auc:.2f}%")
    if hasattr(model, 'threshold_category'):
        threshold_category = model.threshold_category
        print(f"📦 Category: {threshold_category}")

print("="*60)

# Threshold 사용 안내
if best_threshold is not None:
    print(f"\n💡 예측 시 Threshold 사용 방법:")
    print(f"   - Best Threshold ({best_threshold:.6f})를 사용하여 예측 레이블 결정")
    print(f"   - anomalib 규칙: 양품=0, 불량=1, 높은 score=비정상")
    print(f"   - pred_score >= {best_threshold:.6f} → 비정상 (1)")
    print(f"   - pred_score < {best_threshold:.6f} → 정상 (0)")
    if good_threshold is not None and good_threshold > 0:
        print(f"   - Good Threshold: {good_threshold:.6f}")
    if bad_threshold is not None and bad_threshold > 0:
        print(f"   - Bad Threshold: {bad_threshold:.6f}")
else:
    print(f"\n⚠️ Threshold 정보가 없습니다. 모델이 학습 시 threshold를 저장하지 않았을 수 있습니다.")
    print(f"   예측 시 기본 threshold를 사용합니다.")

print()

# ============================================
# 5. 예측 수행
# ============================================
print("🔍 예측을 시작합니다...")
predictions = engine.predict(
    model=model,
    datamodule=datamodule,
)

print(f"✅ 예측 완료! 총 {len(predictions)} 배치")

# ============================================
# 6. 예측 결과 수집
# ============================================
results = []

for batch in predictions:
    batch_size = batch.image.shape[0]
    
    for i in range(batch_size):
        # 이미지 경로
        image_path = batch.image_path[i] if hasattr(batch, 'image_path') else f"unknown_{i}"
        
        # 실제 레이블 (0: 정상, 1: 비정상)
        gt_label = batch.gt_label[i].item() if hasattr(batch, 'gt_label') else None
        
        # 예측 레이블 (0: 정상, 1: 비정상)
        pred_label = batch.pred_label[i].item() if hasattr(batch, 'pred_label') else None
        
        # 예측 점수
        pred_score = batch.pred_score[i].item() if hasattr(batch, 'pred_score') else None
        
        # 레이블을 텍스트로 변환
        gt_label_text = "정상" if gt_label == 0 else "비정상"
        pred_label_text = "정상" if pred_label == 0 else "비정상"
        
        # Threshold 정보 추가
        result_dict = {
            "image_path": str(image_path),
            "gt_label": gt_label,
            "gt_label_text": gt_label_text,
            "pred_label": pred_label,
            "pred_label_text": pred_label_text,
            "pred_score": pred_score,
            "correct": gt_label == pred_label if gt_label is not None and pred_label is not None else None,
        }
        
        # Threshold 정보 추가 (있는 경우)
        if best_threshold is not None:
            result_dict["best_threshold"] = best_threshold
            result_dict["threshold_기준_예측"] = "정상" if pred_score >= best_threshold else "비정상"
            result_dict["score_vs_threshold"] = pred_score - best_threshold if pred_score is not None else None
        if good_threshold is not None and good_threshold > 0:
            result_dict["good_threshold"] = good_threshold
        if bad_threshold is not None and bad_threshold > 0:
            result_dict["bad_threshold"] = bad_threshold
        
        results.append(result_dict)

# DataFrame 생성
df = pd.DataFrame(results)

print(f"\n📊 예측 결과 요약:")
print(f"  - 총 이미지 수: {len(df)}")
print(f"  - 정확도: {df['correct'].sum() / len(df) * 100:.2f}%")
print(f"  - 정상 이미지: {(df['gt_label'] == 0).sum()}개")
print(f"  - 비정상 이미지: {(df['gt_label'] == 1).sum()}개")

# Threshold 정보 표시
if best_threshold is not None:
    print(f"\n📊 사용된 Threshold:")
    print(f"  - Best Threshold: {best_threshold:.6f}")
    if good_threshold is not None and good_threshold > 0:
        print(f"  - Good Threshold: {good_threshold:.6f}")
    if bad_threshold is not None and bad_threshold > 0:
        print(f"  - Bad Threshold: {bad_threshold:.6f}")
    
    # Threshold 기준으로 예측된 이미지 수
    threshold_based_normal = (df['pred_score'] >= best_threshold).sum()
    threshold_based_abnormal = (df['pred_score'] < best_threshold).sum()
    print(f"  - Threshold 기준 정상 예측: {threshold_based_normal}개")
    print(f"  - Threshold 기준 비정상 예측: {threshold_based_abnormal}개")

# ============================================
# 7. Confusion Matrix 생성
# ============================================
if df['gt_label'].notna().all() and df['pred_label'].notna().all():
    # Confusion Matrix 계산
    cm = confusion_matrix(df['gt_label'], df['pred_label'], labels=[0, 1])
    
    # 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['정상', '비정상'],
        yticklabels=['정상', '비정상'],
        cbar_kws={'label': '개수'}
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('실제 레이블', fontsize=12)
    plt.xlabel('예측 레이블', fontsize=12)
    plt.tight_layout()
    
    # 저장
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✅ Confusion Matrix 저장: {output_path / 'confusion_matrix.png'}")
    plt.close()
    
    # Classification Report
    report = classification_report(
        df['gt_label'],
        df['pred_label'],
        target_names=['정상', '비정상'],
        output_dict=True
    )
    
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_path / 'classification_report.csv', encoding='utf-8-sig')
    print(f"✅ Classification Report 저장: {output_path / 'classification_report.csv'}")
    
    print("\n📈 Classification Report:")
    print(classification_report(
        df['gt_label'],
        df['pred_label'],
        target_names=['정상', '비정상']
    ))

# ============================================
# 8. CSV 파일로 저장
# ============================================
output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True)

# 예측 결과 CSV 저장
csv_path = output_path / 'predictions.csv'
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 예측 결과 CSV 저장: {csv_path}")

# 요약 통계 저장
summary = {
    "총 이미지 수": len(df),
    "정확도 (%)": f"{df['correct'].sum() / len(df) * 100:.2f}",
    "정상 이미지 수": int((df['gt_label'] == 0).sum()),
    "비정상 이미지 수": int((df['gt_label'] == 1).sum()),
    "정상으로 예측": int((df['pred_label'] == 0).sum()),
    "비정상으로 예측": int((df['pred_label'] == 1).sum()),
    "평균 예측 점수": f"{df['pred_score'].mean():.4f}",
    "최소 예측 점수": f"{df['pred_score'].min():.4f}",
    "최대 예측 점수": f"{df['pred_score'].max():.4f}",
}

# Threshold 정보 추가
if best_threshold is not None:
    summary["best_threshold"] = f"{best_threshold:.6f}"
    if good_threshold is not None and good_threshold > 0:
        summary["good_threshold"] = f"{good_threshold:.6f}"
    if bad_threshold is not None and bad_threshold > 0:
        summary["bad_threshold"] = f"{bad_threshold:.6f}"
    
    # Threshold 기준 통계
    threshold_based_normal = (df['pred_score'] >= best_threshold).sum()
    threshold_based_abnormal = (df['pred_score'] < best_threshold).sum()
    summary["threshold_기준_정상_예측"] = int(threshold_based_normal)
    summary["threshold_기준_비정상_예측"] = int(threshold_based_abnormal)
else:
    summary["best_threshold"] = "N/A"
    summary["good_threshold"] = "N/A"
    summary["bad_threshold"] = "N/A"

summary_df = pd.DataFrame([summary])
summary_df.to_csv(output_path / 'summary.csv', index=False, encoding='utf-8-sig')
print(f"✅ 요약 통계 CSV 저장: {output_path / 'summary.csv'}")

print("\n🎉 모든 작업 완료!")
print(f"📁 결과 파일 위치: {output_path}")

