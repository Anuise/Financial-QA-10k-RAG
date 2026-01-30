"""輸出格式化器 - 將處理結果輸出為 JSONL 與 JSON 報告"""

import json
from pathlib import Path
from typing import Dict, List


class OutputFormatter:
    """輸出格式化器"""

    def format_chunk(self, chunk_data: Dict, index: int) -> Dict:
        """
        格式化為標準 JSONL 結構

        Args:
            chunk_data: Chunk 原始資料
            index: Chunk 索引

        Returns:
            格式化後的 Chunk
        """
        # 生成 document_id (ticker_filing)
        document_id = f"{chunk_data['ticker']}_{chunk_data['filing']}"

        # 生成 chunk_id
        chunk_id = f"{document_id}_chunk_{index:04d}"

        formatted = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "text": chunk_data['text'],
            "metadata": {
                "ticker": chunk_data['ticker'],
                "filing": chunk_data['filing'],
                "qa_pairs": chunk_data['qa_pairs'],
                "word_count": chunk_data['word_count'],
                "token_count": chunk_data['token_count'],
                "hash": chunk_data['hash']
            }
        }

        return formatted

    def save_jsonl(self, chunks: List[Dict], output_path: Path):
        """
        儲存為 JSONL 格式

        Args:
            chunks: Chunk 列表
            output_path: 輸出檔案路徑
        """
        # 確保輸出目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 格式化並儲存
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                formatted_chunk = self.format_chunk(chunk, i)
                f.write(json.dumps(formatted_chunk, ensure_ascii=False) + '\n')

        print(f"✓ JSONL 已儲存: {output_path} ({len(chunks)} 筆)")

    def save_report(self, report: Dict, output_path: Path):
        """
        儲存處理報告為 JSON

        Args:
            report: 報告資料
            output_path: 輸出檔案路徑
        """
        # 確保輸出目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"✓ 報告已儲存: {output_path}")
