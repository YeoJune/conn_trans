# analyze_results.py
"""
훈련 완료 후 결과 분석 전용 스크립트 - ComparisonAnalyzer 사용
"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Analyze Connection Transformer experiment results")
    parser.add_argument("--output_dir", 
                       type=str, 
                       default="./outputs",
                       help="Directory containing experiments and analysis folders")
    parser.add_argument("--force", 
                       action="store_true",
                       help="Force re-analysis even if recent comparison exists")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"❌ Output directory does not exist: {output_dir}")
        return 1
    
    print(f"🔍 Analyzing experiment results in: {output_dir}")
    print("-" * 50)
    
    try:
        from utils.comparison_analyzer import ComparisonAnalyzer
        
        # 비교 분석 실행
        analyzer = ComparisonAnalyzer(str(output_dir))
        success = analyzer.analyze_all_experiments()
        
        if success:
            # 요약 정보 출력
            summary = analyzer.get_comparison_summary()
            
            print(f"\n📊 Analysis Results:")
            print(f"   Total experiments: {summary['total_experiments']}")
            print(f"   Datasets: {', '.join(summary['datasets'])}")
            print(f"   Models: {', '.join(summary['models'])}")
            print(f"   Best accuracy: {summary['best_accuracy']:.4f}")
            print(f"   Average accuracy: {summary['average_accuracy']:.4f}")
            print(f"   Results saved in: {summary['comparison_dir']}")
            
            print(f"\n✅ Analysis completed successfully!")
            print(f"📋 Check the markdown report and visualizations in the comparison directory.")
            
            return 0
        else:
            print(f"❌ Analysis failed - insufficient experiment data")
            print(f"💡 Run some training experiments first using main.py")
            return 1
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())