"""CSV 解析器 - 載入並驗證 Financial-QA-10k 資料集"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd


class CSVParser:
    """CSV 解析器"""

    REQUIRED_COLUMNS = ['question', 'answer', 'context', 'ticker', 'filing']

    def __init__(self):
        self.df: pd.DataFrame = None

    def load(self, file_path: Path) -> pd.DataFrame:
        """
        載入 CSV 並驗證必要欄位

        Args:
            file_path: CSV 檔案路徑

        Returns:
            載入的 DataFrame

        Raises:
            FileNotFoundError: 檔案不存在
            ValueError: 缺少必要欄位
        """
        if not file_path.exists():
            raise FileNotFoundError(f"CSV 檔案不存在: {file_path}")

        # 載入 CSV
        self.df = pd.read_csv(file_path)

        # 驗證必要欄位
        missing_columns = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"缺少必要欄位: {missing_columns}")

        print(f"✓ 成功載入 CSV: {len(self.df)} 筆資料")
        return self.df

    def get_statistics(self) -> Dict[str, Any]:
        """
        取得欄位統計資訊

        Returns:
            統計資訊字典
        """
        if self.df is None:
            raise ValueError("尚未載入資料，請先呼叫 load()")

        stats = {
            "total_rows": len(self.df),
            "columns": list(self.df.columns),
            "null_counts": self.df.isnull().sum().to_dict(),
            "unique_tickers": self.df['ticker'].nunique(),
            "unique_filings": self.df['filing'].nunique(),
            "context_stats": {
                "avg_length": self.df['context'].str.len().mean(),
                "min_length": self.df['context'].str.len().min(),
                "max_length": self.df['context'].str.len().max(),
            }
        }

        return stats

    def print_statistics(self):
        """列印統計資訊"""
        stats = self.get_statistics()

        print("\n" + "=" * 50)
        print("資料集統計")
        print("=" * 50)
        print(f"總筆數: {stats['total_rows']:,}")
        print(f"欄位數: {len(stats['columns'])}")
        print(f"\n欄位列表: {', '.join(stats['columns'])}")
        print(f"\n唯一 Ticker 數量: {stats['unique_tickers']}")
        print(f"唯一 Filing 數量: {stats['unique_filings']}")
        print(f"\nContext 長度統計:")
        print(f"  平均: {stats['context_stats']['avg_length']:.0f} 字元")
        print(f"  最小: {stats['context_stats']['min_length']} 字元")
        print(f"  最大: {stats['context_stats']['max_length']} 字元")

        print(f"\n空值統計:")
        for col, count in stats['null_counts'].items():
            if count > 0:
                print(f"  {col}: {count}")
        print("=" * 50 + "\n")
