# utils/result_manager.py
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch

class ResultManager:
    """통합 결과 관리자 - 학습 중/후 파일을 체계적으로 분리"""
    
    def __init__(self, base_dir: str, model_type: str, dataset: str, model_size: str):
        self.base_dir = Path(base_dir)
        self.model_type = model_type
        self.dataset = dataset
        self.model_size = model_size
        
        # 실험 식별자 생성
        self.timestamp = time.strftime("%Y%m%d_%H%M")
        self.exp_id = f"{self.timestamp}_{model_type}_{dataset}_{model_size}"
        
        # 디렉토리 구조 설정
        self.exp_dir = self.base_dir / "experiments" / self.exp_id
        self.analysis_dir = self.base_dir / "analysis" / self.exp_id
        
        # 디렉토리 생성
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # 실시간 메트릭 추적
        self.metrics = {
            'train_losses': [],
            'eval_losses': [],
            'eval_accuracies': [],
            'reasoning_steps': []
        }
        
        print(f"📁 실험 디렉토리: {self.exp_dir}")
        print(f"📊 분석 디렉토리: {self.analysis_dir}")
    
    def save_config(self, config):
        """실험 설정 저장 (학습 시작 시)"""
        config_path = self.exp_dir / "config.json"
        config_dict = vars(config) if hasattr(config, '__dict__') else config
        
        # 메타데이터 추가
        config_dict.update({
            'experiment_id': self.exp_id,
            'timestamp': self.timestamp,
            'model_type': self.model_type,
            'dataset': self.dataset,
            'model_size': self.model_size
        })
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"💾 설정 저장: {config_path.name}")
    
    def update_metrics(self, epoch: int, train_loss: float, eval_loss: float, 
                      accuracy: float, reasoning_steps: Optional[float] = None):
        """실시간 메트릭 업데이트"""
        self.metrics['train_losses'].append(train_loss)
        self.metrics['eval_losses'].append(eval_loss)
        self.metrics['eval_accuracies'].append(accuracy)
        
        if reasoning_steps is not None:
            self.metrics['reasoning_steps'].append(reasoning_steps)
        
        # 실시간 메트릭 저장
        metrics_path = self.exp_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_training(self, message: str):
        """실시간 훈련 로그"""
        log_path = self.exp_dir / "training_log.txt"
        timestamp = time.strftime("%H:%M:%S")
        
        with open(log_path, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def save_checkpoint(self, model, optimizer, epoch: int, accuracy: float, is_best: bool = False):
        """체크포인트 저장"""
        if is_best:
            checkpoint_path = self.exp_dir / "model_best.pt"
        else:
            checkpoint_path = self.exp_dir / f"model_epoch_{epoch}.pt"
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'total_parameters': total_params,
            'experiment_id': self.exp_id
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            print(f"🏆 최고 모델 저장: {checkpoint_path.name}")
    
    def finalize_training(self, best_accuracy: float, model, predictions: List[str], targets: List[str]):
        """훈련 완료 후 최종 분석 수행"""
        print(f"\n📊 최종 분석 시작...")
        
        # 1. 최종 리포트 생성
        self._generate_final_report(best_accuracy, model)
        
        # 2. 최종 시각화 생성
        self._generate_final_visualizations(model, predictions, targets)
        
        # 3. 결과 요약 저장 (파라미터 수 포함)
        self._save_experiment_summary(best_accuracy, model, predictions, targets)
        
        print(f"✅ 분석 완료: {self.analysis_dir}")
        return self.analysis_dir
    
    def _generate_final_report(self, best_accuracy: float, model):
        """종합 마크다운 리포트 생성"""
        report_path = self.analysis_dir / "report.md"
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        with open(report_path, 'w') as f:
            f.write(f"# 실험 결과: {self.exp_id}\n\n")
            
            # 기본 정보
            f.write("## 📋 실험 정보\n")
            f.write(f"- **모델**: {self.model_type.title()}\n")
            f.write(f"- **데이터셋**: {self.dataset.upper()}\n")
            f.write(f"- **크기**: {self.model_size}\n")
            f.write(f"- **실행 시간**: {self.timestamp}\n")
            f.write(f"- **최고 정확도**: {best_accuracy:.4f}\n")
            f.write(f"- **총 파라미터**: {total_params:,}\n\n")
            
            # 훈련 진행
            f.write("## 📈 훈련 진행\n")
            if self.metrics['train_losses']:
                f.write(f"- **최종 훈련 손실**: {self.metrics['train_losses'][-1]:.4f}\n")
                f.write(f"- **최종 평가 손실**: {self.metrics['eval_losses'][-1]:.4f}\n")
                f.write(f"- **총 에포크**: {len(self.metrics['train_losses'])}\n")
            
            if self.metrics['reasoning_steps']:
                avg_steps = sum(self.metrics['reasoning_steps']) / len(self.metrics['reasoning_steps'])
                f.write(f"- **평균 추론 단계**: {avg_steps:.2f}\n")
            
            # 모델 분석
            f.write("\n## 🔍 모델 분석\n")
            if hasattr(model, 'get_connection_analysis'):
                analysis = model.get_connection_analysis()
                f.write(f"- **연결 희소성**: {analysis.get('sparsity_ratio', 0):.4f}\n")
                f.write(f"- **최대 연결 강도**: {analysis.get('max_connection', 0):.4f}\n")
                if 'orthogonality_quality' in analysis:
                    f.write(f"- **직교성 품질**: {analysis['orthogonality_quality']:.4f}\n")
            
            f.write(f"- **총 파라미터**: {total_params:,}\n")
            
            f.write(f"\n## 📁 파일 위치\n")
            f.write(f"- **실험 데이터**: `experiments/{self.exp_id}/`\n")
            f.write(f"- **분석 결과**: `analysis/{self.exp_id}/`\n")
        
        print(f"📋 리포트 생성: {report_path.name}")
    
    def _generate_final_visualizations(self, model, predictions: List[str], targets: List[str]):
        """최종 시각화 생성 (학습 후 한 번만)"""
        try:
            from .visualization_manager import VisualizationManager
            
            vis_manager = VisualizationManager(self.analysis_dir)
            
            # 1. 최종 학습 곡선
            vis_manager.plot_final_training_curves(
                self.metrics['train_losses'],
                self.metrics['eval_accuracies'],
                self.metrics.get('reasoning_steps')
            )
            
            # 2. 연결 행렬 (Connection Transformer만)
            if self.model_type == "connection" and hasattr(model, 'get_connection_analysis'):
                vis_manager.plot_connection_matrix(model)
            
            # 3. 정확도 분석
            if predictions and targets:
                vis_manager.plot_accuracy_breakdown(predictions, targets, self.dataset)
            
            print(f"📊 시각화 완료")
            
        except Exception as e:
            print(f"⚠️ 시각화 오류: {str(e)[:50]}...")
    
    def _save_experiment_summary(self, best_accuracy: float, model, predictions: List[str], targets: List[str]):
        """실험 요약 저장 (파라미터 수 포함)"""
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            'experiment_id': self.exp_id,
            'model_type': self.model_type,
            'dataset': self.dataset,
            'model_size': self.model_size,
            'timestamp': self.timestamp,
            'best_accuracy': best_accuracy,
            'total_parameters': total_params,
            'final_metrics': {
                'train_loss': self.metrics['train_losses'][-1] if self.metrics['train_losses'] else 0,
                'eval_loss': self.metrics['eval_losses'][-1] if self.metrics['eval_losses'] else 0,
                'num_epochs': len(self.metrics['train_losses'])
            },
            'sample_predictions': predictions[:3],
            'sample_targets': targets[:3]
        }
        
        summary_path = self.analysis_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"💾 요약 저장: {summary_path.name}")