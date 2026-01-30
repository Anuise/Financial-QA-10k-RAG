# 階段五:系統評估 (Evaluation)

## 1. 系統定位與核心價值 (Context & Value)

### 1.1 階段定位

系統評估是 RAG 系統開發的品質保證環節,負責透過量化指標與人工評測,全面驗證系統在檢索準確度、答案忠實度與整體可用性上的表現。這個階段的產出將直接指導後續的優化方向與部署決策。

### 1.2 核心痛點

RAG 系統的評估具有以下挑戰:

**挑戰一:多維度品質**

- **檢索品質**:是否找到了所有相關文檔?
- **生成品質**:答案是否忠實於原文?是否流暢自然?
- **端到端品質**:整體使用者體驗如何?

**挑戰二:缺乏標準答案**

- 財務問答往往沒有唯一正解
- 需要評估答案的「合理性」而非「正確性」

**挑戰三:評估成本**

- 人工評測耗時且主觀
- 自動化評估需要精心設計的指標

**解決方案**:

1. **RAGAS 框架**:使用專為 RAG 設計的評估框架
2. **多層級評估**:分別評估檢索、生成與端到端效果
3. **混合評估**:結合自動化指標與人工抽樣

### 1.3 預期效果

完成本階段後,系統將具備:

1. **量化基準**:建立 Context Recall, Faithfulness 等關鍵指標的基準值
2. **弱點識別**:明確系統在哪些查詢類型上表現不佳
3. **優化方向**:基於數據驅動的改進建議
4. **部署信心**:透過評估報告證明系統可用性

---

> **架構銜接說明**:
> 了解評估的必要性後,下一層將說明如何透過「測試集準備→RAGAS 評估→人工驗證→報告生成」的流程,建立完整的評估體系。

---

## 2. 工作流程與架構 (Workflow & Architecture)

### 2.1 整體流程

系統評估遵循以下處理流程:

```
[2.2] 測試集準備 (Test Set Preparation)
    - 從 Kaggle Dataset 提取 Q&A 對
    - 人工標註相關文檔 (Ground Truth)
    - 分層抽樣 (事實型/分析型/比較型)
    ↓
[2.3] 檢索評估 (Retrieval Evaluation)
    - Recall@K: 檢索到的相關文檔比例
    - MRR: 第一個相關文檔的平均排名
    - NDCG: 考慮排序品質的指標
    ↓
[2.4] 生成評估 (Generation Evaluation)
    - Faithfulness: 答案是否忠於 Context
    - Answer Relevancy: 答案是否切題
    - Answer Correctness: 與標準答案的相似度
    ↓
[2.5] 端到端評估 (End-to-End Evaluation)
    - RAGAS Score: 綜合評分
    - Latency: 回應時間
    - Cost: Token 消耗成本
    ↓
[2.6] 人工驗證 (Human Evaluation)
    - 抽樣 50 個查詢
    - 評估答案品質 (1-5 分)
    - 識別系統性錯誤
    ↓
[2.7] 評估報告生成 (Report Generation)
    - 指標儀表板
    - 錯誤案例分析
    - 優化建議
```

### 2.2 測試集準備

**資料來源**:

- **Kaggle Dataset**: `financial-q-and-a-10k` 包含 1,500+ 問答對
- **自訂測試集**: 針對特定場景補充 100 個查詢

**測試集結構**:

```json
{
  "test_cases": [
    {
      "query_id": "test_001",
      "query": "What was Apple's revenue in Q4 2023?",
      "query_type": "factual",
      "ground_truth_answer": "Apple reported total revenue of $89.5 billion in Q4 2023.",
      "relevant_chunks": ["AAPL_10K_2023_chunk_042", "AAPL_10K_2023_chunk_043"],
      "difficulty": "easy"
    }
  ]
}
```

**分層抽樣**:

```python
from typing import List, Dict
import random

class TestSetBuilder:
    """測試集建構器"""

    def stratified_sample(
        self,
        all_queries: List[Dict],
        sample_size: int = 200,
        stratify_by: str = 'query_type'
    ) -> List[Dict]:
        """
        分層抽樣

        Args:
            all_queries: 所有查詢
            sample_size: 樣本數量
            stratify_by: 分層依據 (query_type/difficulty)

        Returns:
            抽樣後的測試集
        """
        # 1. 按類型分組
        groups = {}
        for query in all_queries:
            key = query[stratify_by]
            if key not in groups:
                groups[key] = []
            groups[key].append(query)

        # 2. 計算每組樣本數
        samples_per_group = sample_size // len(groups)

        # 3. 從每組抽樣
        sampled = []
        for group_queries in groups.values():
            sampled.extend(random.sample(group_queries, min(samples_per_group, len(group_queries))))

        return sampled
```

### 2.3 RAGAS 評估框架

**RAGAS (Retrieval Augmented Generation Assessment)** 是專為 RAG 系統設計的評估框架。

**核心指標**:

| 指標                   | 評估對象   | 計算方式                          | 目標值 |
| ---------------------- | ---------- | --------------------------------- | ------ |
| **Context Recall**     | 檢索品質   | 相關文檔被檢索到的比例            | > 0.85 |
| **Context Precision**  | 檢索精確度 | Top-K 中相關文檔的比例            | > 0.70 |
| **Faithfulness**       | 答案忠實度 | 答案中可被 Context 驗證的陳述比例 | > 0.90 |
| **Answer Relevancy**   | 答案相關性 | 答案與問題的相關程度              | > 0.80 |
| **Answer Correctness** | 答案正確性 | 與標準答案的語意相似度            | > 0.75 |

**實作**:

```python
from ragas import evaluate
from ragas.metrics import (
    context_recall,
    context_precision,
    faithfulness,
    answer_relevancy,
    answer_correctness
)
from datasets import Dataset

class RAGASEvaluator:
    """RAGAS 評估器"""

    def __init__(self):
        self.metrics = [
            context_recall,
            context_precision,
            faithfulness,
            answer_relevancy,
            answer_correctness
        ]

    def evaluate(
        self,
        test_cases: List[Dict],
        rag_system
    ) -> Dict:
        """
        評估 RAG 系統

        Args:
            test_cases: 測試案例
            rag_system: RAG 系統實例

        Returns:
            評估結果
        """
        # 1. 執行 RAG 系統,收集結果
        results = []
        for case in test_cases:
            response = rag_system.query(case['query'])

            results.append({
                'question': case['query'],
                'answer': response['answer'],
                'contexts': [s['excerpt'] for s in response['sources']],
                'ground_truth': case['ground_truth_answer'],
                'ground_truth_contexts': case['relevant_chunks']
            })

        # 2. 轉換為 RAGAS Dataset 格式
        dataset = Dataset.from_list(results)

        # 3. 執行評估
        evaluation_result = evaluate(
            dataset,
            metrics=self.metrics
        )

        return evaluation_result
```

### 2.4 指標詳解

#### 2.4.1 Context Recall

**定義**:Ground Truth 中的相關文檔有多少被檢索到。

**公式**:

$$
\text{Context Recall} = \frac{|\text{Retrieved} \cap \text{Relevant}|}{|\text{Relevant}|}
$$

**實作**:

```python
def calculate_context_recall(
    retrieved_chunks: List[str],
    relevant_chunks: List[str]
) -> float:
    """
    計算 Context Recall

    Args:
        retrieved_chunks: 檢索到的 Chunk IDs
        relevant_chunks: 相關的 Chunk IDs (Ground Truth)

    Returns:
        Recall 分數 (0-1)
    """
    retrieved_set = set(retrieved_chunks)
    relevant_set = set(relevant_chunks)

    if not relevant_set:
        return 0.0

    intersection = retrieved_set & relevant_set
    return len(intersection) / len(relevant_set)
```

#### 2.4.2 Faithfulness

**定義**:答案中的陳述有多少可以被 Context 驗證。

**計算方式**:

1. 將答案拆分為多個陳述 (Claims)
2. 使用 LLM 判斷每個陳述是否可被 Context 支持
3. 計算被支持的陳述比例

**實作**:

```python
import openai

class FaithfulnessChecker:
    """忠實度檢查器"""

    def check_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> Dict:
        """
        檢查答案忠實度

        Returns:
            {
                'faithfulness_score': 忠實度分數,
                'claims': 所有陳述,
                'supported_claims': 被支持的陳述,
                'unsupported_claims': 未被支持的陳述
            }
        """
        # 1. 提取陳述
        claims = self._extract_claims(answer)

        # 2. 驗證每個陳述
        supported = []
        unsupported = []

        context_text = "\n\n".join(contexts)

        for claim in claims:
            is_supported = self._verify_claim(claim, context_text)
            if is_supported:
                supported.append(claim)
            else:
                unsupported.append(claim)

        # 3. 計算分數
        score = len(supported) / len(claims) if claims else 0.0

        return {
            'faithfulness_score': score,
            'claims': claims,
            'supported_claims': supported,
            'unsupported_claims': unsupported
        }

    def _extract_claims(self, answer: str) -> List[str]:
        """使用 LLM 提取陳述"""
        prompt = f"""Extract all factual claims from the following answer.
Return each claim as a separate line.

Answer: {answer}

Claims:"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        claims_text = response.choices[0].message.content
        claims = [c.strip() for c in claims_text.split('\n') if c.strip()]

        return claims

    def _verify_claim(self, claim: str, context: str) -> bool:
        """驗證單個陳述"""
        prompt = f"""Given the following context, can the claim be verified?

Context: {context}

Claim: {claim}

Answer with only 'Yes' or 'No'."""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content.strip().lower()
        return answer == 'yes'
```

#### 2.4.3 Answer Correctness

**定義**:答案與標準答案的語意相似度。

**計算方式**:

1. 使用 Embedding 模型編碼答案與標準答案
2. 計算 Cosine 相似度
3. 結合 F1 Score (詞彙重疊)

**實作**:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class AnswerCorrectnessEvaluator:
    """答案正確性評估器"""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def evaluate(
        self,
        generated_answer: str,
        ground_truth: str
    ) -> Dict:
        """
        評估答案正確性

        Returns:
            {
                'correctness_score': 正確性分數,
                'semantic_similarity': 語意相似度,
                'f1_score': F1 分數
            }
        """
        # 1. 語意相似度
        emb_gen = self.embedding_model.encode([generated_answer])[0]
        emb_gt = self.embedding_model.encode([ground_truth])[0]

        semantic_sim = cosine_similarity(
            emb_gen.reshape(1, -1),
            emb_gt.reshape(1, -1)
        )[0][0]

        # 2. F1 Score (詞彙重疊)
        f1 = self._calculate_f1(generated_answer, ground_truth)

        # 3. 綜合分數 (加權平均)
        correctness = 0.7 * semantic_sim + 0.3 * f1

        return {
            'correctness_score': correctness,
            'semantic_similarity': semantic_sim,
            'f1_score': f1
        }

    def _calculate_f1(self, answer: str, ground_truth: str) -> float:
        """計算 F1 Score"""
        answer_tokens = set(answer.lower().split())
        gt_tokens = set(ground_truth.lower().split())

        if not answer_tokens or not gt_tokens:
            return 0.0

        intersection = answer_tokens & gt_tokens

        precision = len(intersection) / len(answer_tokens)
        recall = len(intersection) / len(gt_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
```

---

> **細節銜接說明**:
> 確立了「RAGAS 評估→指標計算」的流程後,以下將深入說明人工評測設計、A/B 測試方法與評估報告生成。

---

## 3. 技術規格與實作細節 (Detailed Specification)

### 3.1 人工評測設計

#### 3.1.1 評測維度

**評測表單**:

| 維度         | 評分標準 (1-5 分)           | 權重 |
| ------------ | --------------------------- | ---- |
| **相關性**   | 答案是否直接回答問題?       | 30%  |
| **準確性**   | 答案中的事實是否正確?       | 30%  |
| **完整性**   | 答案是否涵蓋問題的所有面向? | 20%  |
| **流暢性**   | 答案是否自然易讀?           | 10%  |
| **可驗證性** | 來源引用是否充分?           | 10%  |

**評分指南**:

```
相關性:
5 - 完全切題,直接回答問題
4 - 大部分相關,有少量離題
3 - 部分相關,但包含無關資訊
2 - 勉強相關,大部分離題
1 - 完全不相關

準確性:
5 - 所有事實均正確
4 - 絕大部分正確,有微小錯誤
3 - 部分正確,有明顯錯誤
2 - 大部分錯誤
1 - 完全錯誤或幻覺
```

#### 3.1.2 評測流程

```python
class HumanEvaluationManager:
    """人工評測管理器"""

    def create_evaluation_task(
        self,
        test_cases: List[Dict],
        sample_size: int = 50
    ) -> str:
        """
        建立評測任務

        Returns:
            評測表單的 CSV 路徑
        """
        import csv

        # 隨機抽樣
        sampled = random.sample(test_cases, sample_size)

        # 生成 CSV
        output_path = 'evaluation_form.csv'
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Query ID', 'Query', 'Generated Answer', 'Ground Truth',
                'Relevance (1-5)', 'Accuracy (1-5)', 'Completeness (1-5)',
                'Fluency (1-5)', 'Verifiability (1-5)', 'Comments'
            ])

            for case in sampled:
                writer.writerow([
                    case['query_id'],
                    case['query'],
                    case['generated_answer'],
                    case['ground_truth_answer'],
                    '',  # 待填寫
                    '',
                    '',
                    '',
                    '',
                    ''
                ])

        return output_path

    def analyze_results(self, evaluation_csv: str) -> Dict:
        """分析評測結果"""
        import pandas as pd

        df = pd.read_csv(evaluation_csv)

        # 計算各維度平均分
        dimensions = ['Relevance (1-5)', 'Accuracy (1-5)', 'Completeness (1-5)',
                      'Fluency (1-5)', 'Verifiability (1-5)']

        scores = {}
        for dim in dimensions:
            scores[dim] = df[dim].mean()

        # 計算加權總分
        weights = [0.3, 0.3, 0.2, 0.1, 0.1]
        overall = sum(scores[dim] * w for dim, w in zip(dimensions, weights))

        return {
            'dimension_scores': scores,
            'overall_score': overall,
            'num_evaluated': len(df)
        }
```

### 3.2 A/B 測試

#### 3.2.1 實驗設計

**對照組**:

- **Baseline**: 純向量檢索 + GPT-3.5
- **Variant A**: 混合檢索 (Alpha=0.5) + GPT-4
- **Variant B**: 混合檢索 (Alpha=0.3) + GPT-4 + Reranking

**評估指標**:

- RAGAS Score
- 平均延遲
- 成本 ($/query)

**實作**:

```python
class ABTestRunner:
    """A/B 測試執行器"""

    def run_experiment(
        self,
        test_cases: List[Dict],
        systems: Dict[str, callable]
    ) -> pd.DataFrame:
        """
        執行 A/B 測試

        Args:
            test_cases: 測試案例
            systems: {系統名稱: 系統函數}

        Returns:
            結果 DataFrame
        """
        results = []

        for case in test_cases:
            for system_name, system_func in systems.items():
                # 執行查詢
                start_time = time.time()
                response = system_func(case['query'])
                latency = time.time() - start_time

                # 記錄結果
                results.append({
                    'query_id': case['query_id'],
                    'system': system_name,
                    'answer': response['answer'],
                    'latency_ms': latency * 1000,
                    'cost': self._calculate_cost(response),
                    'ragas_score': None  # 稍後計算
                })

        return pd.DataFrame(results)

    def _calculate_cost(self, response: Dict) -> float:
        """計算查詢成本"""
        # GPT-4: $0.03/1K prompt tokens, $0.06/1K completion tokens
        prompt_cost = response['metadata']['tokens_used']['prompt'] / 1000 * 0.03
        completion_cost = response['metadata']['tokens_used']['completion'] / 1000 * 0.06
        return prompt_cost + completion_cost
```

#### 3.2.2 統計顯著性檢驗

```python
from scipy import stats

class StatisticalAnalyzer:
    """統計分析器"""

    def compare_systems(
        self,
        results_df: pd.DataFrame,
        metric: str = 'ragas_score'
    ) -> Dict:
        """
        比較系統間的統計顯著性

        Args:
            results_df: A/B 測試結果
            metric: 比較指標

        Returns:
            統計檢驗結果
        """
        systems = results_df['system'].unique()

        if len(systems) != 2:
            raise ValueError("Only supports 2-system comparison")

        # 提取兩組數據
        group_a = results_df[results_df['system'] == systems[0]][metric]
        group_b = results_df[results_df['system'] == systems[1]][metric]

        # t 檢驗
        t_stat, p_value = stats.ttest_ind(group_a, group_b)

        # 效應量 (Cohen's d)
        cohens_d = (group_a.mean() - group_b.mean()) / np.sqrt(
            (group_a.std()**2 + group_b.std()**2) / 2
        )

        return {
            'system_a': systems[0],
            'system_b': systems[1],
            'mean_a': group_a.mean(),
            'mean_b': group_b.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_effect_size(cohens_d)
        }

    def _interpret_effect_size(self, d: float) -> str:
        """解釋效應量"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
```

### 3.3 評估報告生成

#### 3.3.1 報告結構

```markdown
# RAG 系統評估報告

## 1. 執行摘要

- 測試日期: 2024-01-30
- 測試案例數: 200
- 整體 RAGAS Score: 0.78

## 2. 檢索評估

- Context Recall: 0.87
- Context Precision: 0.72
- MRR: 0.65

## 3. 生成評估

- Faithfulness: 0.92
- Answer Relevancy: 0.81
- Answer Correctness: 0.74

## 4. 效能指標

- 平均延遲: 2.3s
- P95 延遲: 4.1s
- 平均成本: $0.08/query

## 5. 錯誤案例分析

### 案例 1: 數值提取失敗

- Query: "What is the debt-to-equity ratio?"
- Issue: 未能從表格中正確提取數值
- Root Cause: Chunking 截斷了表格

## 6. 優化建議

1. 改進表格處理邏輯
2. 調整 Alpha 參數至 0.3
3. 增加 Reranking 階段
```

#### 3.3.2 視覺化儀表板

```python
import matplotlib.pyplot as plt
import seaborn as sns

class EvaluationReporter:
    """評估報告生成器"""

    def generate_dashboard(
        self,
        evaluation_results: Dict,
        output_dir: Path
    ):
        """生成視覺化儀表板"""

        # 1. RAGAS 指標雷達圖
        self._plot_ragas_radar(evaluation_results, output_dir / 'ragas_radar.png')

        # 2. 延遲分布直方圖
        self._plot_latency_distribution(evaluation_results, output_dir / 'latency_dist.png')

        # 3. 錯誤類型分布
        self._plot_error_distribution(evaluation_results, output_dir / 'error_dist.png')

    def _plot_ragas_radar(self, results: Dict, output_path: Path):
        """RAGAS 指標雷達圖"""
        metrics = ['Context Recall', 'Context Precision', 'Faithfulness',
                   'Answer Relevancy', 'Answer Correctness']
        values = [results[m] for m in metrics]

        # 雷達圖
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('RAGAS Metrics', size=16, pad=20)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
```

### 3.4 持續評估

#### 3.4.1 評估自動化

```python
class ContinuousEvaluator:
    """持續評估器"""

    def __init__(self, test_set: List[Dict], rag_system):
        self.test_set = test_set
        self.rag_system = rag_system
        self.history = []

    def run_daily_evaluation(self):
        """每日評估"""
        results = RAGASEvaluator().evaluate(self.test_set, self.rag_system)

        # 記錄歷史
        self.history.append({
            'date': datetime.now().isoformat(),
            'results': results
        })

        # 檢查退化
        if self._detect_regression(results):
            self._send_alert(results)

        return results

    def _detect_regression(self, current_results: Dict) -> bool:
        """檢測效能退化"""
        if len(self.history) < 2:
            return False

        previous = self.history[-2]['results']

        # 若任一指標下降超過 5%,視為退化
        for metric in ['context_recall', 'faithfulness', 'answer_correctness']:
            if current_results[metric] < previous[metric] * 0.95:
                return True

        return False

    def _send_alert(self, results: Dict):
        """發送警報"""
        print(f"⚠️ Performance regression detected!")
        print(f"Current RAGAS Score: {results['ragas_score']:.3f}")
```

---

## 4. 與下一階段的銜接

完成系統評估後,我們已建立了量化基準並識別了優化方向。**階段六:部署與介面**將:

1. 基於評估結果選擇最佳配置 (Alpha, Top-K 等)
2. 建立 FastAPI Backend 與 Streamlit Frontend
3. 部署效能監控與日誌系統

> **關鍵依賴**:
>
> - 階段六將使用本階段確定的最佳參數配置
> - 評估報告將作為系統文檔的一部分
> - 持續評估機制將整合至部署環境
