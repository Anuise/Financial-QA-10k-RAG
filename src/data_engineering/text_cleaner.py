"""文本清洗器 - 移除 HTML、正規化 Unicode、標準化空白字元"""

import re
import unicodedata
from bs4 import BeautifulSoup


class TextCleaner:
    """文本清洗器"""

    # Unicode 替換映射表
    UNICODE_REPLACEMENTS = {
        '\u00a0': ' ',      # Non-breaking space
        '\u2019': "'",      # Right single quotation mark
        '\u201c': '"',      # Left double quotation mark
        '\u201d': '"',      # Right double quotation mark
        '\u2013': '-',      # En dash
        '\u2014': '--',     # Em dash
    }

    def clean_html(self, text: str) -> str:
        """
        移除 HTML 標籤，保留純文本

        Args:
            text: 原始文本

        Returns:
            清洗後的文本
        """
        soup = BeautifulSoup(text, 'html.parser')

        # 移除 script 與 style 標籤
        for tag in soup(['script', 'style']):
            tag.decompose()

        # 取得純文本
        clean_text = soup.get_text()

        return clean_text

    def normalize_unicode(self, text: str) -> str:
        """
        標準化 Unicode 字元

        Args:
            text: 原始文本

        Returns:
            正規化後的文本
        """
        # NFKC 正規化 (相容性分解 + 標準組合)
        text = unicodedata.normalize('NFKC', text)

        # 替換常見亂碼
        for old, new in self.UNICODE_REPLACEMENTS.items():
            text = text.replace(old, new)

        return text

    def normalize_whitespace(self, text: str) -> str:
        """
        標準化空白字元

        Args:
            text: 原始文本

        Returns:
            標準化後的文本
        """
        # 移除多餘空白
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # 移除行首行尾空白
        text = '\n'.join(line.strip() for line in text.split('\n'))

        # 移除多餘空格
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def clean(self, text: str) -> str:
        """
        完整清洗流程

        Args:
            text: 原始文本

        Returns:
            清洗後的文本
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. 移除 HTML
        text = self.clean_html(text)

        # 2. Unicode 正規化
        text = self.normalize_unicode(text)

        # 3. 空白字元處理
        text = self.normalize_whitespace(text)

        return text
