"""Context 處理器 - 去重、長度檢查、Metadata 綁定"""

import hashlib
from typing import Dict, List, Tuple
import pandas as pd
import tiktoken


class ContextProcessor:
    """Context 處理器"""

    def __init__(self, cleaner):
        """
        初始化處理器

        Args:
            cleaner: TextCleaner 實例
        """
        self.cleaner = cleaner
        # 使用 cl100k_base (GPT-4 tokenizer)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _hash_context(self, text: str) -> str:
        """計算 Context 的 MD5 hash"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def deduplicate(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        去重 Context，並將相關 Q&A 映射至 Metadata

        Args:
            df: 包含 question, answer, context, ticker, filing 的 DataFrame

        Returns:
            {context_hash: {'text': str, 'ticker': str, 'filing': str, 'qa_pairs': List[Dict]}}
        """
        context_map = {}

        for _, row in df.iterrows():
            # 清洗 Context
            cleaned_context = self.cleaner.clean(row['context'])

            if not cleaned_context:
                continue

            # 計算 hash
            ctx_hash = self._hash_context(cleaned_context)

            # 若 Context 已存在，新增 Q&A 對
            if ctx_hash in context_map:
                context_map[ctx_hash]['qa_pairs'].append({
                    'question': row['question'],
                    'answer': row['answer']
                })
            else:
                # 新建 Context 記錄
                context_map[ctx_hash] = {
                    'text': cleaned_context,
                    'ticker': row['ticker'],
                    'filing': row['filing'],
                    'qa_pairs': [{
                        'question': row['question'],
                        'answer': row['answer']
                    }]
                }

        print(f"✓ 去重完成: {len(df)} 筆原始資料 → {len(context_map)} 個唯一 Context")
        return context_map

    def check_token_length(self, text: str, max_tokens: int = 512) -> Tuple[bool, int]:
        """
        檢查 Token 長度是否超過限制

        Args:
            text: 文本
            max_tokens: 最大 Token 數

        Returns:
            (是否超過限制, Token 數量)
        """
        tokens = self.tokenizer.encode(text)
        token_count = len(tokens)
        return token_count > max_tokens, token_count

    def process(self, df: pd.DataFrame, max_tokens: int = 512) -> List[Dict]:
        """
        完整處理流程，產出 Chunk 列表

        Args:
            df: 原始 DataFrame
            max_tokens: 最大 Token 數限制

        Returns:
            Chunk 列表
        """
        # 1. 去重
        context_map = self.deduplicate(df)

        # 2. 轉換為 Chunk 列表
        chunks = []
        long_context_count = 0

        for ctx_hash, ctx_data in context_map.items():
            # 檢查 Token 長度
            is_too_long, token_count = self.check_token_length(ctx_data['text'], max_tokens)

            if is_too_long:
                long_context_count += 1
                # 註記但仍保留
                print(f"⚠ Context 超過 {max_tokens} tokens: {token_count} tokens (hash: {ctx_hash[:8]}...)")

            chunk = {
                'hash': ctx_hash,
                'text': ctx_data['text'],
                'ticker': ctx_data['ticker'],
                'filing': ctx_data['filing'],
                'qa_pairs': ctx_data['qa_pairs'],
                'token_count': token_count,
                'word_count': len(ctx_data['text'].split())
            }

            chunks.append(chunk)

        print(f"✓ 處理完成: {len(chunks)} 個 Chunks")
        if long_context_count > 0:
            print(f"⚠ 其中 {long_context_count} 個超過 {max_tokens} tokens")

        return chunks
