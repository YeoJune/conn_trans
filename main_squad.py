import torch
import numpy as np
import time
import json
import warnings
import math  # math 임포트 추가 (SQuADDataset에서 사용될 수 있음)

# 설정 파일 로드 (프로젝트 루트를 기준으로 경로 설정 필요)
from configs.squad_config import get_squad_config  # SQuAD용 설정
from datasets.squad_dataset import SQuADDataset
from models.base_conn_trans import ConnectionTransformer
from models.conn_trans_ffn import ConnTransWithFFN
from models.standard_transformer import StandardTransformer
from training.trainer import train_model  # SQuAD용으로 수정된 trainer 사용
from utils.visualization import visualize_connection_matrix, analyze_reasoning_evolution, print_comparison_results


# from utils.metrics import compute_squad_em_f1 # 필요시 EM/F1 계산 함수 임포트

# JSON 인코더 (결과 저장 시 필요할 수 있음)
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (torch.Tensor)): return o.cpu().tolist()
        if isinstance(o, (torch.float32, torch.float64, torch.float16)): return float(o)
        if isinstance(o, (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8)): return int(o)
        return super(NpEncoder, self).default(o)


def main_squad():
    CFG = get_squad_config()  # SQuAD 설정 로드
    print("🚀 CONNECTION TRANSFORMER - SQuAD 1.1 IMPLEMENTATION (with AutoTokenizer)")
    print(f"   Using Tokenizer: {CFG['tokenizer_name']}")
    print("=" * 70)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = False # 필요시 설정

    # 시드 고정 (재현성을 위해)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("\n📦 SQuAD 1.1 Data Loading with AutoTokenizer...")
    try:
        # SQuADDataset 클래스 변수 tokenizer 초기화 (스크립트 실행 시 한 번만 토크나이저 로드되도록)
        SQuADDataset.tokenizer = None

        train_dataset = SQuADDataset(split='train',
                                     max_seq_len=CFG["max_seq_len"],
                                     tokenizer_name=CFG["tokenizer_name"],
                                     doc_stride=CFG["doc_stride"],
                                     max_query_length=CFG["max_query_length"])
        # val_dataset은 train_dataset에서 초기화된 tokenizer를 공유 (SQuADDataset 내부 로직)
        val_dataset = SQuADDataset(split='validation',  # SQuAD는 'validation' 스플릿 사용
                                   max_seq_len=CFG["max_seq_len"],
                                   tokenizer_name=CFG["tokenizer_name"],
                                   doc_stride=CFG["doc_stride"],
                                   max_query_length=CFG["max_query_length"])
        print("✅ SQuAD Data loading successful")
    except Exception as e:
        print(f"❌ SQuAD Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return {}  # 오류 발생 시 빈 dict 반환 또는 프로그램 종료

    # DataLoader num_workers 설정
    cpu_cores = torch.multiprocessing.cpu_count() if hasattr(torch.multiprocessing, 'cpu_count') else 2
    # nw = min(4, CFG["batch_size"] // 4 if CFG["batch_size"] >=4 else 0, cpu_cores // 2 if cpu_cores > 1 else 0)
    # SQuAD는 데이터 전처리가 복잡할 수 있으므로, num_workers는 0으로 시작하거나 낮게 설정하는 것이 안정적일 수 있음
    nw = 0 if not torch.cuda.is_available() else min(2, cpu_cores // 2 if cpu_cores > 1 else 0)  # 디버깅 시 0 추천
    print(f"Using num_workers: {nw} for DataLoaders.")

    train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle=True, num_workers=nw,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=CFG["batch_size"], shuffle=False, num_workers=nw,
                            pin_memory=torch.cuda.is_available())

    vocab_size = train_dataset.tokenizer.vocab_size  # 토크나이저로부터 vocab_size 가져옴
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"  ✅ Device: {device}, Tokenizer Vocab Size: {vocab_size:,}")
    print(f"  🔢 Train features: {len(train_dataset):,} (SQuAD는 example당 여러 feature 생성 가능)")
    print(f"  🔢 Val features: {len(val_dataset):,}")
    print(f"  📦 Batch size: {CFG['batch_size']}, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    results_squad = {}

    # 학습할 모델 정의
    models_to_train_squad = [
        ("Connection Transformer", ConnectionTransformer),
        ("Standard Transformer", StandardTransformer),
        ("ConnTrans + FFN", ConnTransWithFFN),
    ]

    # 샘플 데이터 준비 (analyze_reasoning_evolution용)
    sample_batch_for_trace = None
    if len(val_loader) > 0:
        try:
            sample_batch_for_trace = next(iter(val_loader))  # DataLoader에서 첫 배치 가져오기
            print(f"  📊 Prepared a sample batch from validation loader for reasoning trace.")
        except Exception as e_sample:
            print(f"  ⚠️ Could not get sample batch from val_loader: {e_sample}")
            sample_batch_for_trace = None  # 실패 시 None으로 설정

    for model_name, model_class in models_to_train_squad:
        print("\n" + "=" * 60 + f"\n▶️ SQuAD EXPERIMENT: {model_name}" + "\n" + "=" * 60)
        if torch.cuda.is_available(): torch.cuda.empty_cache()  # 이전 모델 메모리 정리

        # 모델 초기화 시 CONFIG 값 전달
        # 각 모델 클래스의 __init__ 시그니처에 맞게 인자 전달
        if model_class == StandardTransformer:
            model_instance = model_class(
                vocab_size=vocab_size,
                d_model=CFG["d_model"],
                num_heads=CFG.get("num_heads", 8),  # base_config에 num_heads 추가 필요 또는 기본값 사용
                num_layers=CFG.get("num_transformer_layers", CFG["num_reasoning_steps"]),
                ffn_dim_multiplier=CFG.get("ffn_dim_multiplier", 4),
                dropout=CFG.get("dropout", 0.1),
                max_seq_len=CFG["max_seq_len"]
            )
        elif model_class == ConnTransWithFFN:
            model_instance = model_class(
                vocab_size=vocab_size,
                d_model=CFG["d_model"],
                num_slots=CFG["num_slots"],
                num_reasoning_steps=CFG["num_reasoning_steps"],
                max_seq_len=CFG["max_seq_len"],
                connection_init_std=CFG.get("connection_init_std", 0.01),  # base_conn_trans에서 사용
                spectral_radius_limit=CFG.get("spectral_radius_limit", 0.95),  # base_conn_trans에서 사용
                ffn_dim_multiplier=CFG.get("ffn_dim_multiplier", 4),
                dropout=CFG.get("dropout", 0.1)
            )
        else:  # ConnectionTransformer (base)
            model_instance = model_class(
                vocab_size=vocab_size,
                d_model=CFG["d_model"],
                num_slots=CFG["num_slots"],
                num_reasoning_steps=CFG["num_reasoning_steps"],
                max_seq_len=CFG["max_seq_len"],
                connection_init_std=CFG.get("connection_init_std", 0.01),
                spectral_radius_limit=CFG.get("spectral_radius_limit", 0.95)
            )

        # 모델 파라미터 수 출력 (모델 __init__에서 이미 출력됨)
        # total_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        # print(f"   {model_name}: {total_params:,} trainable parameters")

        metric_val = train_model(model_instance, train_loader, val_loader, CFG, device,
                                 model_name)  # SQuAD용 train_model 사용
        results_squad[model_name] = metric_val

        # 학습 후 분석 (metric_val이 의미있는 값일 때만)
        if isinstance(metric_val, float) and metric_val > 0.0 and sample_batch_for_trace:
            if hasattr(model_instance, 'C') and model_instance.C is not None:
                try:
                    visualize_connection_matrix(model_instance, f"{model_name.replace(' ', '_')}_squad_C.png",
                                                f" ({model_name} SQuAD)")
                except Exception as e_vis_cm:
                    print(f"Error visualizing CM for {model_name}: {e_vis_cm}")

            if hasattr(model_instance, 'get_reasoning_trace'):
                try:
                    analyze_reasoning_evolution(model_instance, sample_batch_for_trace,
                                                f"{model_name.replace(' ', '_')}_squad_evo.png", model_name)
                except Exception as e_vis_evo:
                    print(f"Error analyzing/visualizing evo for {model_name}: {e_vis_evo}")

        del model_instance  # 학습 완료된 모델 메모리에서 명시적으로 해제
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print_comparison_results(results_squad, metric_name="Val Span Acc (SQuAD)")

    print(f"\n💾 Saving SQuAD Experimental Results...")
    squad_exp_results = {
        "experiment_type": "squad_1.1_conn_trans_comparison_autotokenizer_refactored",
        "dataset": "SQuAD 1.1",
        "tokenizer": CFG["tokenizer_name"],
        "config_hyperparameters": CFG,  # config -> config_hyperparameters
        "results_metric": "Validation Span Accuracy",  # 사용된 평가지표 명시
        "model_accuracies": results_squad,  # results -> model_accuracies
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }
    squad_results_filename = f"squad_results_autotok_refactored_{squad_exp_results['timestamp']}.json"
    try:
        with open(squad_results_filename, "w") as f:
            json.dump(squad_exp_results, f, indent=2, cls=NpEncoder)  # NpEncoder 사용
        print(f"  📄 SQuAD Results: {squad_results_filename}")
    except Exception as e_json_squad:
        print(f"⚠️ Error saving SQuAD JSON: {e_json_squad}")

    print(f"\n✨ SQuAD Experiment Completed!")
    return results_squad


if __name__ == "__main__":
    # 이 파일이 직접 실행될 때 main_squad() 함수를 호출합니다.
    # main_babi.py는 별도로 실행해야 합니다.

    warnings.filterwarnings("ignore", category=UserWarning)  # 일반적인 UserWarning 무시
    # 특정 라이브러리 경고 무시 (필요시)
    # warnings.filterwarnings("ignore", message="The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option")

    print("=" * 20 + " Starting SQuAD Experiments " + "=" * 20)
    try:
        final_results_squad = main_squad()
        if final_results_squad:
            print(f"\n🎉 SQuAD Experiments Final Results (Val Span Acc):")
            for model, acc in final_results_squad.items():
                print(f"  - {model}: {acc:.4f}")
    except KeyboardInterrupt:
        print(f"\n🛑 SQuAD Experiment interrupted by user.")
    except Exception as e_squad_main:
        print(f"\n❌ SQuAD Main Experiment failed: {e_squad_main}")
        import traceback

        traceback.print_exc()

    print("\n\n" + "=" * 20 + " SQuAD Script Finished " + "=" * 20)