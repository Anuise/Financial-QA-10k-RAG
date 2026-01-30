# 階段四:重排序與生成 (Reranking & Generation)

## 1. 系統定位與核心價值 (Context & Value)

### 1.1 階段定位

重排序與生成是 RAG 系統的最後一哩路,負責將階段三產出的 Top-20 候選文檔進行精細排序,並透過 LLM 生成自然、準確且可驗證的答案。這個階段是使用者直接感知系統品質的關鍵環節。

### 1.2 核心痛點

階段三的混合檢索雖然能快速篩選出候選文檔,但仍存在兩個問題:

**問題一:排序粗糙**

- **現象**:Top-20 中可能包含相關但不精確的文檔
- **原因**:向量相似度與 BM25 分數都是基於淺層特徵 (詞彙重疊、向量距離)
- **影響**:若直接將 Top-5 送入 LLM,可能包含雜訊資訊

**問題二:LLM 幻覺**

- **現象**:LLM 可能生成看似合理但實際錯誤的答案
- **原因**:缺乏對 Context 的深度理解與約束機制
- **影響**:使用者無法信任系統輸出

**解決方案**:

1. **Cross-Encoder Reranking**:使用雙塔模型對 (Query, Document) 對進行精細評分
2. **Prompt Engineering**:透過結構化提示詞約束 LLM 行為
3. **Source Citation**:強制 LLM 標註答案來源,提升可驗證性

### 1.3 預期效果

完成本階段後,系統將具備:

1. **高精度排序**:Top-5 文檔的相關性 > 90%
2. **忠實生成**:答案完全基於檢索到的 Context,無幻覺
3. **可追溯性**:每個答案都標註來源 Chunk ID
4. **自然流暢**:生成的答案符合人類閱讀習慣

---

> **架構銜接說明**:
> 了解 Reranking 與 LLM 生成的必要性後,下一層將說明如何透過「Cross-Encoder 重排序→Context 組裝→Prompt 生成→答案驗證」的流程,實現高品質的問答系統。

---

## 2. 工作流程與架構 (Workflow & Architecture)

### 2.1 整體流程

重排序與生成遵循以下處理流程:

```
階段三產出: Top-20 Candidates
    ↓
[2.2] Cross-Encoder Reranking
    - 載入 BGE-Reranker-v2-m3
    - 對每個 (Query, Chunk) 對計分
    - 重新排序,取 Top-5
    ↓
[2.3] Context Assembly
    - 組裝 Top-5 Chunks
    - 控制 Context Window 長度
    - 新增 Metadata (來源、日期等)
    ↓
[2.4] Prompt Engineering
    - 套用提示詞模板
    - 注入 Role, Constraints, Examples
    - 生成 LLM 輸入
    ↓
[2.5] LLM Generation
    - 呼叫 LLM API (GPT-4/Claude/Llama)
    - 生成答案
    ↓
[2.6] Answer Validation
    - 檢查來源引用
    - 驗證事實一致性
    - 計算 Confidence Score
    ↓
返回最終答案 + 來源引用
```

### 2.2 Cross-Encoder Reranking

**為何需要 Reranking?**

階段三的檢索模型 (Bi-Encoder) 將 Query 與 Document 分別編碼,無法捕捉兩者的交互關係。Cross-Encoder 則同時編碼 (Query, Document) 對,能更精確地評估相關性。

**架構對比**:

| 特性           | Bi-Encoder (BGE-M3)   | Cross-Encoder (Reranker) |
| -------------- | --------------------- | ------------------------ |
| **編碼方式**   | 分別編碼 Query 與 Doc | 聯合編碼 (Query, Doc)    |
| **計算複雜度** | O(N)                  | O(N²)                    |
| **適用階段**   | 初篩 (Top-K=50)       | 精排 (Top-K=5)           |
| **推理速度**   | 快 (~50 docs/sec)     | 慢 (~5 pairs/sec)        |
| **準確度**     | 中                    | 高                       |

**實作流程**:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CrossEncoderReranker:
    """Cross-Encoder 重排序器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # 移至 GPU (若可用)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],  # [(chunk_id, text), ...]
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        重排序候選文檔

        Args:
            query: 使用者查詢
            candidates: 候選文檔 [(chunk_id, text), ...]
            top_k: 返回前 K 個結果

        Returns:
            重排序後的結果 [(chunk_id, score), ...]
        """
        scores = []

        for chunk_id, text in candidates:
            # 組合 Query 與 Document
            inputs = self.tokenizer(
                query,
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            # 前向傳播
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits[0][0].item()  # 相關性分數

            scores.append((chunk_id, score))

        # 排序並返回 Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 2.3 Context Assembly

**目標**:將 Top-5 Chunks 組裝為結構化的 Context,供 LLM 使用。

**挑戰**:

1. **長度限制**:LLM 有最大 Token 限制 (如 GPT-4: 8K tokens)
2. **資訊密度**:需要在有限空間內包含最多相關資訊
3. **可讀性**:Context 需要有清晰的結構,方便 LLM 理解

**實作**:

```python
from typing import List, Dict

class ContextAssembler:
    """Context 組裝器"""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def assemble(
        self,
        chunks: List[Dict],
        query: str
    ) -> str:
        """
        組裝 Context

        Args:
            chunks: Chunk 列表,每個包含 {chunk_id, text, metadata}
            query: 使用者查詢

        Returns:
            組裝後的 Context 字串
        """
        context_parts = []
        current_tokens = 0

        for i, chunk in enumerate(chunks, 1):
            # 估算 Token 數 (粗略:1 token ≈ 4 字元)
            chunk_tokens = len(chunk['text']) // 4

            # 檢查是否超過限制
            if current_tokens + chunk_tokens > self.max_tokens:
                break

            # 組裝單個 Chunk
            chunk_context = self._format_chunk(chunk, i)
            context_parts.append(chunk_context)
            current_tokens += chunk_tokens

        # 組合所有 Chunks
        full_context = "\n\n---\n\n".join(context_parts)

        return full_context

    def _format_chunk(self, chunk: Dict, index: int) -> str:
        """
        格式化單個 Chunk

        Returns:
            格式化後的字串
        """
        metadata = chunk['metadata']

        # 提取關鍵 Metadata
        source = metadata.get('document_id', 'Unknown')
        section = metadata.get('section', 'N/A')
        has_table = metadata.get('has_table', False)

        # 組裝
        formatted = f"""[Source {index}]
Document: {source}
Section: {section}
Contains Table: {'Yes' if has_table else 'No'}

Content:
{chunk['text']}
"""
        return formatted
```

### 2.4 Prompt Engineering

**目標**:設計結構化提示詞,引導 LLM 生成高品質答案。

**提示詞結構**:

```
[System Role] → 定義 LLM 的角色與專業領域
[Task Description] → 說明任務目標
[Constraints] → 約束條件 (如:僅使用 Context 資訊)
[Context] → 檢索到的文檔
[Query] → 使用者問題
[Output Format] → 期望的輸出格式
[Examples] (Optional) → Few-shot 範例
```

**實作**:

```python
class PromptBuilder:
    """Prompt 建構器"""

    SYSTEM_ROLE = """You are a financial analyst assistant specializing in 10-K annual reports.
Your role is to provide accurate, data-driven answers based solely on the provided context."""

    CONSTRAINTS = """
CRITICAL CONSTRAINTS:
1. Only use information from the provided [Source] sections below
2. If the answer is not in the context, explicitly state "I cannot find this information in the provided documents"
3. Always cite the source number (e.g., [Source 1]) when making claims
4. Do not make assumptions or use external knowledge
5. If numbers are mentioned, quote them exactly as they appear
"""

    OUTPUT_FORMAT = """
OUTPUT FORMAT:
- Provide a clear, concise answer
- Use bullet points for multiple items
- Include source citations in [Source X] format
- End with a confidence level: High/Medium/Low
"""

    def build_prompt(
        self,
        query: str,
        context: str,
        include_examples: bool = False
    ) -> str:
        """
        建構完整 Prompt

        Args:
            query: 使用者查詢
            context: 組裝後的 Context
            include_examples: 是否包含 Few-shot 範例

        Returns:
            完整 Prompt
        """
        prompt_parts = [
            self.SYSTEM_ROLE,
            self.CONSTRAINTS,
            self.OUTPUT_FORMAT
        ]

        # Few-shot Examples (可選)
        if include_examples:
            prompt_parts.append(self._get_examples())

        # Context
        prompt_parts.append(f"\n--- CONTEXT ---\n{context}\n")

        # Query
        prompt_parts.append(f"\n--- QUESTION ---\n{query}\n")

        # Instruction
        prompt_parts.append("\n--- YOUR ANSWER ---")

        return "\n".join(prompt_parts)

    def _get_examples(self) -> str:
        """Few-shot 範例"""
        return """
--- EXAMPLES ---

Example 1:
Q: What was Apple's revenue in Q4 2023?
A: According to [Source 1], Apple reported total revenue of $89.5 billion in Q4 2023.
Confidence: High

Example 2:
Q: What is the company's strategy for AI development?
A: I cannot find specific information about AI development strategy in the provided documents.
Confidence: N/A
"""
```

### 2.5 LLM Generation

**模型選擇**:

| 模型              | 優勢                       | 劣勢                     | 適用場景   |
| ----------------- | -------------------------- | ------------------------ | ---------- |
| **GPT-4**         | 最高品質,強大推理能力      | 成本高 ($0.03/1K tokens) | 生產環境   |
| **Claude 3**      | 長 Context (200K),安全性高 | API 限制較多             | 長文檔分析 |
| **Llama 3 (70B)** | 開源,可自託管              | 需要 GPU 資源            | 本地部署   |
| **GPT-3.5-Turbo** | 成本低 ($0.002/1K tokens)  | 品質較低                 | 開發測試   |

**實作**:

```python
import openai
from typing import Optional

class LLMGenerator:
    """LLM 生成器"""

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 500
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> Dict:
        """
        生成答案

        Args:
            prompt: 完整 Prompt
            system_message: 系統訊息 (可選)

        Returns:
            {
                'answer': 生成的答案,
                'usage': Token 使用統計,
                'model': 使用的模型
            }
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        # 呼叫 OpenAI API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        return {
            'answer': response.choices[0].message.content,
            'usage': response.usage,
            'model': response.model
        }
```

---

> **細節銜接說明**:
> 確立了「Reranking→Context Assembly→Prompt→Generation」的流程後,以下將深入說明答案驗證機制、Confidence Score 計算與錯誤處理策略。

---

## 3. 技術規格與實作細節 (Detailed Specification)

### 3.1 Answer Validation

#### 3.1.1 來源引用檢查

**目標**:確保答案中的每個事實都有對應的來源引用。

```python
import re

class AnswerValidator:
    """答案驗證器"""

    def validate_citations(self, answer: str, num_sources: int) -> Dict:
        """
        驗證來源引用

        Args:
            answer: LLM 生成的答案
            num_sources: 提供的來源數量

        Returns:
            {
                'has_citations': 是否包含引用,
                'cited_sources': 引用的來源列表,
                'uncited_sources': 未被引用的來源,
                'invalid_citations': 無效的引用 (如 [Source 99])
            }
        """
        # 提取所有引用
        citation_pattern = r'\[Source (\d+)\]'
        citations = re.findall(citation_pattern, answer)
        cited_sources = set(int(c) for c in citations)

        # 檢查有效性
        valid_sources = set(range(1, num_sources + 1))
        invalid_citations = cited_sources - valid_sources
        uncited_sources = valid_sources - cited_sources

        return {
            'has_citations': len(cited_sources) > 0,
            'cited_sources': sorted(cited_sources),
            'uncited_sources': sorted(uncited_sources),
            'invalid_citations': sorted(invalid_citations)
        }
```

#### 3.1.2 事實一致性檢查

**目標**:驗證答案中的數值與 Context 是否一致。

```python
class FactChecker:
    """事實檢查器"""

    def check_numerical_consistency(
        self,
        answer: str,
        context: str
    ) -> Dict:
        """
        檢查數值一致性

        Returns:
            {
                'numbers_in_answer': 答案中的數值,
                'verified_numbers': 在 Context 中找到的數值,
                'unverified_numbers': 無法驗證的數值
            }
        """
        # 提取數值 (包含貨幣符號、百分比等)
        number_pattern = r'\$?\d{1,3}(,\d{3})*(\.\d+)?[BMK]?%?'

        answer_numbers = set(re.findall(number_pattern, answer))
        context_numbers = set(re.findall(number_pattern, context))

        verified = answer_numbers & context_numbers
        unverified = answer_numbers - context_numbers

        return {
            'numbers_in_answer': sorted(answer_numbers),
            'verified_numbers': sorted(verified),
            'unverified_numbers': sorted(unverified)
        }
```

### 3.2 Confidence Score 計算

**Confidence Score 組成**:

$$
\text{Confidence} = w_1 \cdot S_{\text{rerank}} + w_2 \cdot S_{\text{citation}} + w_3 \cdot S_{\text{fact}}
$$

其中:

- $S_{\text{rerank}}$: Reranker 分數 (0-1)
- $S_{\text{citation}}$: 引用完整度 (0-1)
- $S_{\text{fact}}$: 事實一致性 (0-1)
- $w_1, w_2, w_3$: 權重 (預設 0.5, 0.3, 0.2)

**實作**:

```python
class ConfidenceCalculator:
    """Confidence Score 計算器"""

    def __init__(
        self,
        w_rerank: float = 0.5,
        w_citation: float = 0.3,
        w_fact: float = 0.2
    ):
        self.w_rerank = w_rerank
        self.w_citation = w_citation
        self.w_fact = w_fact

    def calculate(
        self,
        rerank_scores: List[float],
        citation_info: Dict,
        fact_check: Dict
    ) -> Dict:
        """
        計算 Confidence Score

        Returns:
            {
                'overall_confidence': 總體信心分數 (0-1),
                'confidence_level': 信心等級 (High/Medium/Low),
                'breakdown': 各項分數明細
            }
        """
        # 1. Reranker Score (取 Top-1 分數)
        s_rerank = self._normalize_rerank_score(rerank_scores[0])

        # 2. Citation Score
        s_citation = self._calculate_citation_score(citation_info)

        # 3. Fact Consistency Score
        s_fact = self._calculate_fact_score(fact_check)

        # 4. 加權平均
        overall = (
            self.w_rerank * s_rerank +
            self.w_citation * s_citation +
            self.w_fact * s_fact
        )

        # 5. 分級
        if overall >= 0.7:
            level = "High"
        elif overall >= 0.4:
            level = "Medium"
        else:
            level = "Low"

        return {
            'overall_confidence': overall,
            'confidence_level': level,
            'breakdown': {
                'rerank_score': s_rerank,
                'citation_score': s_citation,
                'fact_score': s_fact
            }
        }

    def _normalize_rerank_score(self, score: float) -> float:
        """正規化 Reranker 分數至 [0, 1]"""
        # BGE-Reranker 分數範圍約為 [-10, 10]
        return (score + 10) / 20

    def _calculate_citation_score(self, citation_info: Dict) -> float:
        """計算引用分數"""
        if not citation_info['has_citations']:
            return 0.0

        # 引用覆蓋率
        total_sources = len(citation_info['cited_sources']) + len(citation_info['uncited_sources'])
        coverage = len(citation_info['cited_sources']) / total_sources

        # 懲罰無效引用
        penalty = len(citation_info['invalid_citations']) * 0.2

        return max(0, coverage - penalty)

    def _calculate_fact_score(self, fact_check: Dict) -> float:
        """計算事實一致性分數"""
        total_numbers = len(fact_check['numbers_in_answer'])

        if total_numbers == 0:
            return 1.0  # 無數值,預設通過

        verified_ratio = len(fact_check['verified_numbers']) / total_numbers
        return verified_ratio
```

### 3.3 錯誤處理

#### 3.3.1 LLM API 失敗處理

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustLLMGenerator(LLMGenerator):
    """帶重試機制的 LLM 生成器"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_with_retry(self, prompt: str) -> Dict:
        """
        帶重試的生成

        處理:
        - Rate Limit 錯誤 → 指數退避重試
        - Timeout 錯誤 → 重試
        - 其他錯誤 → 返回降級答案
        """
        try:
            return self.generate(prompt)
        except openai.error.RateLimitError as e:
            print(f"⚠️ Rate limit hit, retrying...")
            raise  # 觸發重試
        except openai.error.Timeout as e:
            print(f"⚠️ Timeout, retrying...")
            raise
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return self._fallback_answer()

    def _fallback_answer(self) -> Dict:
        """降級答案"""
        return {
            'answer': "I apologize, but I'm currently unable to generate an answer. Please try again later.",
            'usage': {},
            'model': 'fallback'
        }
```

#### 3.3.2 Context 過長處理

```python
class AdaptiveContextAssembler(ContextAssembler):
    """自適應 Context 組裝器"""

    def assemble_adaptive(
        self,
        chunks: List[Dict],
        query: str,
        max_tokens: int
    ) -> str:
        """
        自適應組裝 Context

        策略:
        1. 若 Top-5 超過限制,逐步減少至 Top-3
        2. 若仍超過,截斷每個 Chunk 的長度
        3. 優先保留包含表格的 Chunks
        """
        # 1. 嘗試 Top-5
        context = self.assemble(chunks[:5], query)
        if self._estimate_tokens(context) <= max_tokens:
            return context

        # 2. 降至 Top-3
        context = self.assemble(chunks[:3], query)
        if self._estimate_tokens(context) <= max_tokens:
            return context

        # 3. 截斷策略
        return self._truncate_chunks(chunks[:3], max_tokens)

    def _estimate_tokens(self, text: str) -> int:
        """估算 Token 數"""
        return len(text) // 4

    def _truncate_chunks(self, chunks: List[Dict], max_tokens: int) -> str:
        """截斷 Chunks"""
        # 實作細節略
        pass
```

### 3.4 效能優化

#### 3.4.1 Reranking 批次化

```python
class BatchedReranker(CrossEncoderReranker):
    """批次化 Reranker"""

    def rerank_batched(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        batch_size: int = 8,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        批次化重排序

        優勢:
        - 減少 GPU 呼叫次數
        - 提升吞吐量 3-5x
        """
        all_scores = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]

            # 批次編碼
            texts = [text for _, text in batch]
            inputs = self.tokenizer(
                [query] * len(batch),
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            # 批次推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits[:, 0].cpu().numpy()

            # 記錄結果
            for (chunk_id, _), score in zip(batch, scores):
                all_scores.append((chunk_id, float(score)))

        # 排序並返回 Top-K
        all_scores.sort(key=lambda x: x[1], reverse=True)
        return all_scores[:top_k]
```

#### 3.4.2 Prompt 快取

```python
class CachedPromptBuilder(PromptBuilder):
    """帶快取的 Prompt 建構器"""

    def __init__(self):
        super().__init__()
        self._template_cache = {}

    def build_prompt_cached(
        self,
        query: str,
        context: str
    ) -> str:
        """
        使用快取的模板建構 Prompt

        優勢:
        - 避免重複字串拼接
        - 減少記憶體分配
        """
        # 快取模板
        if 'base_template' not in self._template_cache:
            self._template_cache['base_template'] = "\n".join([
                self.SYSTEM_ROLE,
                self.CONSTRAINTS,
                self.OUTPUT_FORMAT
            ])

        template = self._template_cache['base_template']

        return f"{template}\n\n--- CONTEXT ---\n{context}\n\n--- QUESTION ---\n{query}\n\n--- YOUR ANSWER ---"
```

### 3.5 輸出格式

#### 3.5.1 標準回應結構

```json
{
  "query": "What was Apple's revenue in Q4 2023?",
  "answer": "According to [Source 1], Apple reported total revenue of $89.5 billion in Q4 2023, representing a 2% year-over-year decline.",
  "sources": [
    {
      "source_id": 1,
      "chunk_id": "AAPL_10K_2023_chunk_042",
      "document": "AAPL_10K_2023",
      "section": "Item 8. Financial Statements",
      "rerank_score": 8.42,
      "excerpt": "Total net sales for the fourth quarter of fiscal 2023 were $89.5 billion..."
    }
  ],
  "confidence": {
    "overall_confidence": 0.87,
    "confidence_level": "High",
    "breakdown": {
      "rerank_score": 0.92,
      "citation_score": 1.0,
      "fact_score": 0.67
    }
  },
  "metadata": {
    "retrieval_time_ms": 145,
    "reranking_time_ms": 320,
    "generation_time_ms": 1850,
    "total_time_ms": 2315,
    "model": "gpt-4",
    "tokens_used": {
      "prompt": 3240,
      "completion": 87,
      "total": 3327
    }
  }
}
```

---

## 4. 與下一階段的銜接

完成重排序與生成後,系統已能提供高品質的問答服務。**階段五:系統評估**將:

1. 使用 RAGAS 框架評估 RAG 系統的整體品質
2. 計算 Context Recall, Faithfulness, Answer Correctness 等指標
3. 建立基準線 (Baseline) 並進行 A/B 測試

> **關鍵依賴**:
>
> - 階段五需要收集本階段產出的 (Query, Answer, Sources) 三元組
> - Confidence Score 將作為評估指標的參考
> - 來源引用資訊將用於計算 Faithfulness
