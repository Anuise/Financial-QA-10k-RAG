"""
資料處理主程式 - 整合所有模組執行完整的資料工程流程

執行方式:
    python scripts/process_data.py --input data/raw/Financial-QA-10k.csv --output-dir data/processed
"""

import argparse
import time
from pathlib import Path
import sys

# 將 src 加入 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_engineering import (
    CSVParser,
    TextCleaner,
    ContextProcessor,
    QualityValidator,
    OutputFormatter
)


def main():
    """主執行流程"""
    parser = argparse.ArgumentParser(
        description='處理 Financial-QA-10k 資料集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    python scripts/process_data.py --input data/raw/Financial-QA-10k.csv
    python scripts/process_data.py --input data/raw/Financial-QA-10k.csv --output-dir output
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        type=Path,
        help='輸入 CSV 檔案路徑'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed'),
        help='輸出目錄 (預設: data/processed)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=50,
        help='最小字數 (預設: 50)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help='最大字數 (預設: 2048)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='最大 Token 數 (預設: 512)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Financial-QA-10k 資料工程流程")
    print("=" * 70)
    print(f"輸入檔案: {args.input}")
    print(f"輸出目錄: {args.output_dir}")
    print("=" * 70 + "\n")

    start_time = time.time()

    try:
        # 1. 載入 CSV
        print("步驟 1/5: 載入 CSV")
        print("-" * 70)
        csv_parser = CSVParser()
        df = csv_parser.load(args.input)
        csv_parser.print_statistics()

        # 2. 清洗與處理 Context
        print("步驟 2/5: 清洗與處理 Context")
        print("-" * 70)
        text_cleaner = TextCleaner()
        context_processor = ContextProcessor(text_cleaner)
        chunks = context_processor.process(df, max_tokens=args.max_tokens)
        print()

        # 3. 品質驗證
        print("步驟 3/5: 品質驗證")
        print("-" * 70)
        quality_validator = QualityValidator(
            min_length=args.min_length,
            max_length=args.max_length
        )
        quality_analysis_dir = args.output_dir / 'quality_analysis'
        quality_report = quality_validator.generate_report(chunks, quality_analysis_dir)

        # 4. 輸出 JSONL
        print("步驟 4/5: 輸出 JSONL")
        print("-" * 70)
        output_formatter = OutputFormatter()
        output_formatter.save_jsonl(chunks, args.output_dir / 'chunks.jsonl')
        print()

        # 5. 生成處理報告
        print("步驟 5/5: 生成處理報告")
        print("-" * 70)
        processing_time = time.time() - start_time

        processing_report = {
            'input_file': str(args.input),
            'output_dir': str(args.output_dir),
            'processing_time_seconds': round(processing_time, 2),
            'statistics': {
                'total_input_rows': len(df),
                'total_output_chunks': len(chunks),
                'deduplication_ratio': round(len(chunks) / len(df), 3),
                'avg_qa_pairs_per_chunk': round(
                    sum(len(c['qa_pairs']) for c in chunks) / len(chunks), 2
                ),
            },
            'quality_report': quality_report,
            'parameters': {
                'min_length': args.min_length,
                'max_length': args.max_length,
                'max_tokens': args.max_tokens
            }
        }

        output_formatter.save_report(
            processing_report,
            args.output_dir / 'processing_report.json'
        )

        # 完成
        print("\n" + "=" * 70)
        print("處理完成!")
        print("=" * 70)
        print(f"處理時間: {processing_time:.2f} 秒")
        print(f"輸入筆數: {len(df):,}")
        print(f"輸出 Chunks: {len(chunks):,}")
        print(f"去重比例: {len(chunks) / len(df):.1%}")
        print(f"\n輸出檔案:")
        print(f"  - {args.output_dir / 'chunks.jsonl'}")
        print(f"  - {args.output_dir / 'processing_report.json'}")
        print(f"  - {quality_analysis_dir / 'length_distribution.png'}")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
