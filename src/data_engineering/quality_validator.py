"""品質驗證器 - 驗證 Chunk 品質並生成分析報告"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from datasketch import MinHash, MinHashLSH


class QualityValidator:
    """品質驗證器"""

    def __init__(self, min_length: int = 50, max_length: int = 2048):
        """
        初始化驗證器

        Args:
            min_length: 最小字數
            max_length: 最大字數
        """
        self.min_length = min_length
        self.max_length = max_length

    def validate_chunk(self, chunk: Dict, chunk_id: str) -> Tuple[bool, Optional[str]]:
        """
        驗證單一 Chunk 是否符合品質標準

        Args:
            chunk: Chunk 資料
            chunk_id: Chunk ID

        Returns:
            (是否通過, 錯誤訊息)
        """
        text = chunk['text']

        # 檢查長度
        word_count = chunk['word_count']
        if word_count < self.min_length:
            return False, f"Chunk {chunk_id} 過短: {word_count} 字"
        if word_count > self.max_length:
            return False, f"Chunk {chunk_id} 過長: {word_count} 字"

        # 檢查是否為空白或重複字元
        unique_chars = len(set(text.replace(' ', '')))
        if unique_chars < 10:
            return False, f"Chunk {chunk_id} 包含過多重複字元"

        # 檢查是否包含可讀文字
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
        if alpha_ratio < 0.5:
            return False, f"Chunk {chunk_id} 字母比例過低: {alpha_ratio:.2%}"

        return True, None

    def analyze_length_distribution(self, chunks: List[Dict], output_dir: Path) -> Dict:
        """
        分析長度分布，生成直方圖

        Args:
            chunks: Chunk 列表
            output_dir: 輸出目錄

        Returns:
            統計資訊
        """
        word_counts = [chunk['word_count'] for chunk in chunks]
        token_counts = [chunk['token_count'] for chunk in chunks]

        # 建立輸出目錄
        output_dir.mkdir(parents=True, exist_ok=True)

        # 繪製字數分布
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(word_counts, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(word_counts), color='red', linestyle='--',
                    label=f'平均: {np.mean(word_counts):.0f}')
        plt.axvline(np.median(word_counts), color='green', linestyle='--',
                    label=f'中位數: {np.median(word_counts):.0f}')
        plt.xlabel('字數')
        plt.ylabel('頻率')
        plt.title('Chunk 字數分布')
        plt.legend()

        # 繪製 Token 分布
        plt.subplot(1, 2, 2)
        plt.hist(token_counts, bins=50, edgecolor='black', alpha=0.7, color='orange')
        plt.axvline(np.mean(token_counts), color='red', linestyle='--',
                    label=f'平均: {np.mean(token_counts):.0f}')
        plt.axvline(np.median(token_counts), color='green', linestyle='--',
                    label=f'中位數: {np.median(token_counts):.0f}')
        plt.xlabel('Token 數')
        plt.ylabel('頻率')
        plt.title('Chunk Token 分布')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'length_distribution.png', dpi=150)
        plt.close()

        stats = {
            'word_count': {
                'mean': float(np.mean(word_counts)),
                'median': float(np.median(word_counts)),
                'std': float(np.std(word_counts)),
                'min': int(np.min(word_counts)),
                'max': int(np.max(word_counts))
            },
            'token_count': {
                'mean': float(np.mean(token_counts)),
                'median': float(np.median(token_counts)),
                'std': float(np.std(token_counts)),
                'min': int(np.min(token_counts)),
                'max': int(np.max(token_counts))
            }
        }

        print(f"✓ 長度分布圖已儲存: {output_dir / 'length_distribution.png'}")
        return stats

    def detect_duplicates(self, chunks: List[Dict], threshold: float = 0.8) -> List[Tuple[int, int]]:
        """
        MinHash LSH 重複檢測

        Args:
            chunks: Chunk 列表
            threshold: 相似度閾值 (0-1)

        Returns:
            重複 Chunk 的索引對列表
        """
        # 建立 LSH 索引
        lsh = MinHashLSH(threshold=threshold, num_perm=128)

        # 為每個 Chunk 建立 MinHash
        minhashes = []
        for i, chunk in enumerate(chunks):
            m = MinHash(num_perm=128)
            for word in chunk['text'].split():
                m.update(word.encode('utf-8'))
            lsh.insert(f"chunk_{i}", m)
            minhashes.append(m)

        # 找出重複對
        duplicates = []
        for i, m in enumerate(minhashes):
            results = lsh.query(m)
            for result in results:
                j = int(result.split('_')[1])
                if i < j:  # 避免重複記錄
                    duplicates.append((i, j))

        if duplicates:
            print(f"⚠ 發現 {len(duplicates)} 對高度相似的 Chunks (閾值: {threshold})")
        else:
            print(f"✓ 無重複 Chunks (閾值: {threshold})")

        return duplicates

    def generate_report(self, chunks: List[Dict], output_dir: Path) -> Dict:
        """
        生成完整品質報告

        Args:
            chunks: Chunk 列表
            output_dir: 輸出目錄

        Returns:
            完整報告
        """
        print("\n" + "=" * 50)
        print("品質驗證")
        print("=" * 50)

        # 1. 驗證每個 Chunk
        validation_errors = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i:04d}"
            is_valid, error_msg = self.validate_chunk(chunk, chunk_id)
            if not is_valid:
                validation_errors.append(error_msg)

        if validation_errors:
            print(f"⚠ 發現 {len(validation_errors)} 個品質問題:")
            for error in validation_errors[:5]:  # 只顯示前 5 個
                print(f"  - {error}")
            if len(validation_errors) > 5:
                print(f"  ... 以及其他 {len(validation_errors) - 5} 個問題")
        else:
            print("✓ 所有 Chunks 通過基本驗證")

        # 2. 長度分布分析
        length_stats = self.analyze_length_distribution(chunks, output_dir)

        # 3. 重複檢測
        duplicates = self.detect_duplicates(chunks)

        report = {
            'total_chunks': len(chunks),
            'validation_errors': len(validation_errors),
            'length_stats': length_stats,
            'duplicates': len(duplicates),
            'errors': validation_errors
        }

        print("=" * 50 + "\n")
        return report
