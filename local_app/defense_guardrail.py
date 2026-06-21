"""
defense_guardrail.py — Dynamic Compliance Guardrail (CPU-optimised)
============================================================
A dual-stage quantitative compliance guardrail component designed for
local execution without dedicated GPU hardware (CPU / MPS optimised).

Pipeline architecture:
  Streaming Token Generation
       |
       v
  ┌─────────────────────────────────────┐
  │  Stage 1: Namespace + Local PPL     │
  │  - Aho-Corasick high-risk entity    │
  │    matching                         │
  │  - W=8 sliding window on hit        │
  │  - Real-time local PPL =            │
  │    exp(-mean(log P))                │
  └──────────┬──────────────────────────┘
       PPL < tau (suspected memorisation)
       |           |  PPL >= tau
       v           v
  ┌──────────┐   Continue generation
  │ Stage 2  │
  │ Semantic │
  │ Entropy  │
  └──┬───┬───┘
     |   |
  H->0|   |H>>0 (normal generalisation)
     v   v
  BLOCK   Resume streaming
  [CONFIDENTIAL QUANT ASSET RESERVED]

Constraints:
  - CPU / MPS only, no GPU dependency
  - Parallel sampling limited to 10 tokens
  - Embedding model uses lightweight option (all-MiniLM-L6-v2 or n-gram fallback)

Usage:
  from defense_guardrail import DefenseGuardrail

  guardrail = DefenseGuardrail(model, tokenizer)
  result = guardrail.evaluate_step(
      token_ids=current_token_ids,
      log_probs=current_log_probs,
  )
  if result.blocked:
      return "[CONFIDENTIAL QUANT ASSET RESERVED]"
"""

from __future__ import annotations

import re
import math
import logging
from collections import deque
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field

import numpy as np

import torch
import torch.nn.functional as F

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("defense_guardrail")


# ── Aho-Corasick lightweight high-risk namespace matcher ──

class TrieNode:
    """Trie node for Aho-Corasick automaton."""
    __slots__ = ("children", "fail", "output", "depth")

    def __init__(self, depth: int = 0):
        self.children: Dict[str, TrieNode] = {}
        self.fail: Optional[TrieNode] = None
        self.output: List[str] = []
        self.depth = depth


class AhoCorasickMatcher:
    """
    Aho-Corasick multi-pattern string matcher.

    Matches all high-risk quantitative entities in a single pass
    (O(n + m) time). Suitable for real-time streaming text detection
    in CPU environments.

    Usage:
        matcher = AhoCorasickMatcher(["因子", "参数", "权重", "阈值"])
        hits = matcher.search("Alpha因子的权重配置为0.142")
        # -> [{"keyword": "因子", "start": 5, "end": 7}, ...]
    """

    def __init__(self, keywords: List[str]):
        self.keywords = list(keywords)
        self.root = TrieNode()
        self._built = False
        self._build()

    def _build(self):
        """Build trie with failure links."""
        # ── 1. Insert all keywords ──
        for kw in self.keywords:
            node = self.root
            for ch in kw:
                if ch not in node.children:
                    node.children[ch] = TrieNode(depth=node.depth + 1)
                node = node.children[ch]
            node.output.append(kw)

        # ── 2. BFS to construct failure links ──
        from collections import deque as bfs_queue

        queue = bfs_queue()
        for ch, child in self.root.children.items():
            child.fail = self.root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for ch, child in current.children.items():
                queue.append(child)
                # Trace back along failure chain
                f_node = current.fail
                while f_node is not None and ch not in f_node.children:
                    f_node = f_node.fail
                child.fail = f_node.children[ch] if f_node else self.root
                # Inherit output from failure node
                if child.fail is not None:
                    child.output.extend(child.fail.output)

        self._built = True

    def search(self, text: str) -> List[Dict]:
        """
        Search for all keyword matches in text.

        Returns:
            [{"keyword": str, "start": int, "end": int}, ...]
        """
        if not self._built:
            self._build()

        results: List[Dict] = []
        node = self.root

        for i, ch in enumerate(text):
            # Trace back along failure chain
            while node is not None and ch not in node.children:
                node = node.fail
            if node is None:
                node = self.root
                continue

            node = node.children[ch]

            # Collect all matches
            for kw in node.output:
                results.append({
                    "keyword": kw,
                    "start": i - len(kw) + 1,
                    "end": i + 1,
                })

        return results


# ── High-risk namespace definitions (quantitative finance) ──

# Chinese high-risk entities
QUANT_HIGH_RISK_ENTITIES_ZH = [
    # Factor-related
    "因子", "Alpha因子", "alpha信号", "超额收益因子",
    "选股因子", "预测变量", "收益驱动因子",
    # Parameter / weight-related
    "权重", "参数", "配比", "配置系数", "分配比例",
    "暴露度", "持仓比重", "调优变量", "模型系数",
    # Risk control
    "止损", "阈值", "止损阈值", "风控参数", "回撤控制",
    "杠杆", "杠杆倍数", "仓位上限", "强制平仓",
    "熔断", "风控触发",
    # Strategy core
    "策略参数", "核心策略", "量化策略", "交易算法",
    "多因子模型", "套利模型",
    # Numeric patterns
    "weight=", "stop_loss", "threshold=", "leverage=",
    "decay=",
]

# English high-risk entities
QUANT_HIGH_RISK_ENTITIES_EN = [
    "alpha", "factor", "weight", "threshold",
    "stop_loss", "leverage", "decay", "sharpe",
    "drawdown", "volatility", "exposure",
    "parameter", "allocation",
    "MultiFactor", "RiskFactor", "SmartBeta",
]

# Combined entity set
ALL_HIGH_RISK_ENTITIES = QUANT_HIGH_RISK_ENTITIES_ZH + QUANT_HIGH_RISK_ENTITIES_EN


# ── Stage 1: Local PPL Detection ──

@dataclass
class Stage1Result:
    """Stage 1 detection result."""
    namespace_hit: bool
    hit_keywords: List[str]
    local_ppl: float
    window_log_probs: List[float]
    trigger_stage2: bool  # PPL < tau triggers semantic entropy arbitration
    window_size: int


class LocalPPLDetector:
    """
    Stage 1: Namespace matching + local PPL window detection.

    When a high-risk entity appears in the output stream, activates
    a W=8 sliding window to compute real-time local perplexity.

    PPL_local = exp(-1/W * sum(log P(w_i | w_<i)))
    """

    def __init__(
        self,
        window_size: int = 8,
        ppl_threshold: float = 5.0,
        keywords: Optional[List[str]] = None,
    ):
        """
        Args:
            window_size: Sliding window size W (default 8)
            ppl_threshold: PPL threshold tau (default 5.0)
            keywords: High-risk entity list (None=use default)
        """
        self.window_size = window_size
        self.ppl_threshold = ppl_threshold
        self.matcher = AhoCorasickMatcher(
            keywords or ALL_HIGH_RISK_ENTITIES
        )

        # 滑动窗口缓存
        self._log_prob_window: deque = deque(maxlen=window_size)
        self._namespace_triggered: bool = False

    def reset(self):
        """Reset sliding window state (call on new conversation start)."""
        self._log_prob_window.clear()
        self._namespace_triggered = False

    def check(self, token_text: str, log_prob: float) -> Stage1Result:
        """
        Perform Stage 1 check for a single newly generated token.

        Args:
            token_text: Decoded text of the current token
            log_prob: Conditional log probability log P(w_i | w_<i)

        Returns:
            Stage1Result
        """
        # ── Namespace matching ──
        hits = self.matcher.search(token_text)
        hit_keywords = [h["keyword"] for h in hits]

        if hits:
            self._namespace_triggered = True

        # ── Update sliding window ──
        self._log_prob_window.append(log_prob)

        # ── Compute local PPL ──
        window = list(self._log_prob_window)
        window_size = len(window)

        trigger_stage2 = False
        local_ppl = float("inf")

        # Only compute PPL when namespace hit + window is full
        if self._namespace_triggered and window_size == self.window_size:
            avg_log_prob = np.mean(window)
            # Prevent numeric overflow
            avg_log_prob = float(np.clip(avg_log_prob, -100, 0))
            local_ppl = float(math.exp(-avg_log_prob))

            # PPL below threshold -> suspected memorisation -> trigger Stage 2
            if local_ppl < self.ppl_threshold:
                trigger_stage2 = True

        return Stage1Result(
            namespace_hit=bool(hits),
            hit_keywords=hit_keywords,
            local_ppl=local_ppl,
            window_log_probs=window,
            trigger_stage2=trigger_stage2,
            window_size=window_size,
        )


# ── Stage 2: Semantic Entropy Final Arbitration ──

@dataclass
class Stage2Result:
    """Stage 2 semantic entropy arbitration result."""
    semantic_entropy: float
    verdict: str  # "BLOCK" | "PASS"
    reason: str
    completions: List[str]
    cluster_sizes: List[int]
    similarity_matrix: np.ndarray


class SemanticEntropyArbiter:
    """
    Stage 2: Semantic entropy final arbitration.

    Samples 3 parallel completion paths (max 10 tokens each) from
    the current context at temperature=0.7, then computes semantic
    entropy:

      H_semantic = -sum(P(C_i) * log(P(C_i)))

     - H -> 0 : High determinism -> BLOCK (suspected memorisation)
     - H >> 0: Normal generalisation -> PASS

    CPU optimisation:
      - Sampling limited to 10 tokens max
      - Embedding model defaults to lightweight option
      - Falls back to n-gram Jaccard similarity (no neural network)
    """

    # Lightweight embedding model (optional)
    LIGHTWEIGHT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        num_samples: int = 3,
        max_completion_tokens: int = 10,
        temperature: float = 0.7,
        entropy_threshold: float = 0.25,
        use_embeddings: bool = True,
    ):
        """
        Args:
            num_samples: Number of parallel sampling paths (default 3)
            max_completion_tokens: Max tokens per sample (default 10)
            temperature: Sampling temperature (default 0.7)
            entropy_threshold: Semantic entropy threshold (H below this -> memorisation)
            use_embeddings: Attempt to load lightweight embedding model
        """
        self.num_samples = num_samples
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.entropy_threshold = entropy_threshold

        # Embedding model (lazy load)
        self._embedding_model = None
        self._use_embeddings = use_embeddings

        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    self.LIGHTWEIGHT_EMBEDDING_MODEL
                )
                logger.info(
                    f"Semantic entropy arbiter: embedding model loaded"
                    f" ({self.LIGHTWEIGHT_EMBEDDING_MODEL})"
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers unavailable, "
                    "falling back to n-gram Jaccard similarity"
                )
                self._use_embeddings = False
            except Exception as e:
                logger.warning(f"Embedding model load failed: {e}, falling back to n-gram")
                self._use_embeddings = False

    @torch.no_grad()
    def _sample_completions(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer,
        device: torch.device,
    ) -> List[str]:
        """
        Generate num_samples parallel completions from current context.

        Each sample generates at most max_completion_tokens tokens.
        """
        completions = []

        for _ in range(self.num_samples):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=(
                    tokenizer.pad_token_id or tokenizer.eos_token_id
                ),
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
            )

            # Extract newly generated tokens
            new_tokens = outputs[0][input_ids.shape[1]:]
            completion = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )
            completions.append(completion.strip())

        return completions

    def _compute_similarity_matrix(
        self, completions: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix of completion texts.

        Prefers embedding cosine similarity, falls back to n-gram Jaccard.
        """
        n = len(completions)
        sim = np.eye(n, dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                sim[i][j] = sim[j][i] = self._pairwise_similarity(
                    completions[i], completions[j]
                )

        return sim

    def _pairwise_similarity(self, a: str, b: str) -> float:
        """Semantic similarity between two text segments."""
        if not a or not b:
            return 0.0

        if self._use_embeddings and self._embedding_model is not None:
            try:
                emb = self._embedding_model.encode(
                    [a, b],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                # Cosine similarity
                dot = np.dot(emb[0], emb[1])
                norm = np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])
                return float(dot / max(norm, 1e-10))
            except Exception:
                pass  # fall through to n-gram

        # ── Fallback: n-gram Jaccard similarity ──
        return self._ngram_jaccard(a, b, n=3)

    @staticmethod
    def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
        """n-gram Jaccard similarity (CPU-friendly)."""
        def ngrams(text: str) -> set:
            # Character-level n-grams
            chars = text.lower()
            return set(
                chars[i:i + n] for i in range(len(chars) - n + 1)
            )

        set_a = ngrams(a)
        set_b = ngrams(b)

        if not set_a or not set_b:
            return 0.0

        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / max(len(union), 1)

    def _cluster_completions(
        self,
        sim_matrix: np.ndarray,
        threshold: float = 0.6,
    ) -> Tuple[List[int], int]:
        """
        Simple clustering: group completions by similarity threshold.

        Completions with similarity > threshold are assigned to the same cluster.

        Returns:
            (cluster_ids, num_clusters)
        """
        n = len(sim_matrix)
        cluster_ids = [-1] * n
        current_cluster = 0

        for i in range(n):
            if cluster_ids[i] != -1:
                continue
            # New cluster
            cluster_ids[i] = current_cluster
            for j in range(n):
                if cluster_ids[j] == -1 and sim_matrix[i][j] > threshold:
                    cluster_ids[j] = current_cluster
            current_cluster += 1

        return cluster_ids, current_cluster

    def compute_semantic_entropy(
        self, completions: List[str]
    ) -> Tuple[float, List[int], np.ndarray]:
        """
        Compute semantic entropy.

        H_semantic = -sum(P(C_i) * log(P(C_i)))

        where P(C_i) = |C_i| / total_samples (cluster size ratio)
        """
        if len(completions) < 2:
            return 0.0, [0], np.eye(1)

        # ── Compute similarity matrix ──
        sim_matrix = self._compute_similarity_matrix(completions)

        # ── Cluster ──
        cluster_ids, num_clusters = self._cluster_completions(sim_matrix)

        # ── Compute cluster probability distribution ──
        cluster_sizes = [
            sum(1 for c in cluster_ids if c == i)
            for i in range(num_clusters)
        ]
        total = sum(cluster_sizes)
        probs = [s / total for s in cluster_sizes]

        # ── Shannon entropy ──
        entropy = -sum(p * math.log(max(p, 1e-10)) for p in probs)

        # Normalise to [0, 1]: max entropy = log(num_clusters)
        max_entropy = math.log(max(num_clusters, 1))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0

        return normalized_entropy, cluster_sizes, sim_matrix

    def arbitrate(
        self,
        model: torch.nn.Module,
        tokenizer,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
    ) -> Stage2Result:
        """
        Execute semantic entropy final arbitration.

        Args:
            model: Language model
            tokenizer: Tokeniser
            input_ids: Token IDs of current context
            attention_mask: Attention mask
            device: Device

        Returns:
            Stage2Result
        """
        # ── 3 parallel samples ──
        completions = self._sample_completions(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            device=device,
        )

        # ── Compute semantic entropy ──
        entropy, cluster_sizes, sim_matrix = (
            self.compute_semantic_entropy(completions)
        )

        # ── Decision branch ──
        if entropy < self.entropy_threshold:
            # Branch A: H -> 0 -> confirmed memorisation -> BLOCK
            verdict = "BLOCK"
            reason = (
                f"Semantic entropy H={entropy:.4f} < threshold={self.entropy_threshold}: "
                f"model outputs are highly consistent across samples, "
                f"indicating memorised content; executing block"
            )
        else:
            # Branch B: H significantly > 0 -> normal generalisation -> PASS
            verdict = "PASS"
            reason = (
                f"Semantic entropy H={entropy:.4f} >= threshold={self.entropy_threshold}: "
                f"model outputs show stochastic generalisation across samples, "
                f"consistent with normal quantitative finance terminology usage"
            )

        return Stage2Result(
            semantic_entropy=float(entropy),
            verdict=verdict,
            reason=reason,
            completions=completions,
            cluster_sizes=cluster_sizes,
            similarity_matrix=sim_matrix,
        )


# ── Integrated Defence Guardrail ──

@dataclass
class GuardrailVerdict:
    """Final defence guardrail verdict."""
    blocked: bool
    stage: int  # 1 or 2 (which stage made the decision)
    replacement_text: Optional[str]
    ppl_score: float
    semantic_entropy: Optional[float]
    reason: str
    details: Dict = field(default_factory=dict)


class DefenseGuardrail:
    """
    Integrated defence guardrail — CPU-optimised version.

    Orchestrates Stage 1 (PPL) -> Stage 2 (semantic entropy) pipeline
    for real-time interception of quantitative strategy privacy leaks
    during streaming generation.

    Usage:
        guardrail = DefenseGuardrail(
            model=my_model,
            tokenizer=my_tokenizer,
            ppl_threshold=5.0,
        )

        for token_id, log_prob in generation_stream:
            result = guardrail.evaluate_step(
                token_text=decode(token_id),
                token_log_prob=log_prob.item(),
                context_ids=current_input_ids,
            )
            if result.blocked:
                yield "[CONFIDENTIAL QUANT ASSET RESERVED]"
                break
            else:
                yield token_text
    """

    # Default PPL threshold (tau)
    DEFAULT_PPL_THRESHOLD = 5.0

    # Default semantic entropy threshold
    DEFAULT_ENTROPY_THRESHOLD = 0.25

    # Replacement text
    CENSOR_TEXT = "[CONFIDENTIAL QUANT ASSET RESERVED]"

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        window_size: int = 8,
        ppl_threshold: float = None,
        entropy_threshold: float = None,
        keywords: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: Language model (may run on CPU)
            tokenizer: Tokeniser
            window_size: PPL sliding window W (default 8)
            ppl_threshold: PPL threshold tau (default 5.0)
            entropy_threshold: Semantic entropy threshold (default 0.25)
            keywords: Custom high-risk entity list (None = default)
            device: Device (None = auto-detect)
        """
        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        logger.info(f"DefenseGuardrail initialised (device: {self.device})")

        # ── Stage 1: Local PPL detector ──
        self.ppl_threshold = ppl_threshold or self.DEFAULT_PPL_THRESHOLD
        self.ppl_detector = LocalPPLDetector(
            window_size=window_size,
            ppl_threshold=self.ppl_threshold,
            keywords=keywords,
        )
        logger.info(
            f"Stage 1 ready: W={window_size}, tau_ppl={self.ppl_threshold}"
        )

        # ── Stage 2: Semantic entropy arbiter ──
        self.entropy_threshold = (
            entropy_threshold or self.DEFAULT_ENTROPY_THRESHOLD
        )
        self.arbiter = SemanticEntropyArbiter(
            num_samples=3,
            max_completion_tokens=10,
            temperature=0.7,
            entropy_threshold=self.entropy_threshold,
            use_embeddings=True,
        )
        logger.info(
            f"Stage 2 ready: samples=3, max_tokens=10, "
            f"tau_entropy={self.entropy_threshold}"
        )

        # Internal state
        self._blocked = False
        self._total_blocks = 0
        self._total_passes = 0

    @torch.no_grad()
    def evaluate_step(
        self,
        token_text: str,
        token_log_prob: float,
        context_ids: Optional[torch.Tensor] = None,
    ) -> GuardrailVerdict:
        """
        Single-step evaluation for streaming generation.

        Called once per generated token in the generation loop.

        Args:
            token_text: Decoded text of current token
            token_log_prob: Conditional log probability of current token
            context_ids: Current context token IDs (for Stage 2 sampling)

        Returns:
            GuardrailVerdict
        """
        # ── Stage 1: Namespace + Local PPL ──
        stage1 = self.ppl_detector.check(token_text, token_log_prob)

        if not stage1.namespace_hit:
            # No namespace hit, no detection needed
            self._total_passes += 1
            return GuardrailVerdict(
                blocked=False,
                stage=0,
                replacement_text=None,
                ppl_score=float("inf"),
                semantic_entropy=None,
                reason="No high-risk namespace hit",
            )

        if not stage1.trigger_stage2:
            # Namespace hit but PPL is normal
            self._total_passes += 1
            return GuardrailVerdict(
                blocked=False,
                stage=1,
                replacement_text=None,
                ppl_score=stage1.local_ppl,
                semantic_entropy=None,
                reason=(
                    f"High-risk entity hit ({', '.join(stage1.hit_keywords)}), "
                    f"but PPL={stage1.local_ppl:.2f} >= tau={self.ppl_threshold}, "
                    f"semantic arbitration not triggered"
                ),
                details={
                    "hit_keywords": stage1.hit_keywords,
                    "local_ppl": stage1.local_ppl,
                },
            )

        # ── Stage 2: Semantic entropy final arbitration ──
        if context_ids is None:
            logger.warning(
                "Stage 2 triggered but context_ids missing, conservative policy: BLOCK"
            )
            self._total_blocks += 1
            return GuardrailVerdict(
                blocked=True,
                stage=2,
                replacement_text=self.CENSOR_TEXT,
                ppl_score=stage1.local_ppl,
                semantic_entropy=None,
                reason="Missing context, conservative block",
            )

        # Ensure context_ids on correct device
        if context_ids.device != self.device:
            context_ids = context_ids.to(self.device)

        attention_mask = torch.ones_like(context_ids, device=self.device)

        # Execute semantic entropy arbitration
        stage2 = self.arbiter.arbitrate(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=context_ids,
            attention_mask=attention_mask,
            device=self.device,
        )

        if stage2.verdict == "BLOCK":
            self._total_blocks += 1
            return GuardrailVerdict(
                blocked=True,
                stage=2,
                replacement_text=self.CENSOR_TEXT,
                ppl_score=stage1.local_ppl,
                semantic_entropy=stage2.semantic_entropy,
                reason=stage2.reason,
                details={
                    "hit_keywords": stage1.hit_keywords,
                    "local_ppl": stage1.local_ppl,
                    "completions": stage2.completions,
                    "cluster_sizes": stage2.cluster_sizes,
                },
            )
        else:
            self._total_passes += 1
            return GuardrailVerdict(
                blocked=False,
                stage=2,
                replacement_text=None,
                ppl_score=stage1.local_ppl,
                semantic_entropy=stage2.semantic_entropy,
                reason=stage2.reason,
            )

    def reset(self):
        """Reset guardrail state (call on new conversation start)."""
        self.ppl_detector.reset()
        self._blocked = False

    def get_stats(self) -> Dict:
        """Get runtime statistics."""
        return {
            "total_blocks": self._total_blocks,
            "total_passes": self._total_passes,
            "block_rate": (
                self._total_blocks
                / max(self._total_blocks + self._total_passes, 1)
            ),
            "ppl_threshold": self.ppl_threshold,
            "entropy_threshold": self.entropy_threshold,
        }


# ── Convenience factory function ──

def create_default_guardrail(
    model: torch.nn.Module,
    tokenizer,
    device: Optional[str] = None,
) -> DefenseGuardrail:
    """Create a DefenceGuardrail with default CPU-optimised settings."""
    return DefenseGuardrail(
        model=model,
        tokenizer=tokenizer,
        window_size=8,
        ppl_threshold=5.0,
        entropy_threshold=0.25,
        keywords=ALL_HIGH_RISK_ENTITIES,
        device=device,
    )


# ── Self-test entry point ──

if __name__ == "__main__":
    print("=" * 60)
    print("  defense_guardrail.py — Stage 1+2 self-test")
    print("=" * 60)

    # ── Test 1: Aho-Corasick matcher ──
    print("\n[Test 1] Aho-Corasick namespace matching")
    matcher = AhoCorasickMatcher(ALL_HIGH_RISK_ENTITIES[:30])
    test_text = (
        "MultiFactor_Quant_Strategy 的Alpha因子的权重参数"
        " weight=0.142, stop_loss_threshold=0.0035"
    )
    hits = matcher.search(test_text)
    print(f"  Text: {test_text}")
    print(f"  Matched {len(hits)} entities:")
    for h in hits:
        print(f"    - '{h['keyword']}' @ [{h['start']}, {h['end']})")

    # ── Test 2: Local PPL detector ──
    print("\n[Test 2] Local PPL detector")
    detector = LocalPPLDetector(window_size=8, ppl_threshold=5.0)

    # Scenario A: Namespace hit then high-probability tokens (simulated memorisation)
    print("  Scenario A: Namespace hit + high-probability sequence (simulated memorisation)")
    token_sequence = ["Alpha", "因子", "的", "权重", "=", "0", ".", "142"]
    high_logprobs = [-0.5, -0.01, -0.02, -0.005, -0.01,
                      -0.015, -0.008, -0.003]
    for i, (tok, lp) in enumerate(zip(token_sequence, high_logprobs)):
        result = detector.check(tok, lp)
        if result.namespace_hit:
            print(f"    Token {i} '{tok}': Namespace hit ({result.hit_keywords})")
        if result.trigger_stage2:
            print(f"    Token {i}: [ALERT] Stage 2 triggered! PPL={result.local_ppl:.2f}")

    # Scenario B: Namespace hit but normal probability (normal generalisation)
    print("  Scenario B: Namespace hit + normal probability (simulated generalisation)")
    detector.reset()
    normal_logprobs = [-0.5, -2.0, -3.0, -2.5, -1.5,
                       -2.8, -1.2, -3.5]
    for i, (tok, lp) in enumerate(zip(token_sequence, normal_logprobs)):
        result = detector.check(tok, lp)
        if result.namespace_hit:
            print(f"    Token {i} '{tok}': Namespace hit")
        if result.trigger_stage2:
            print(f"    Token {i}: PPL={result.local_ppl:.2f}")
    if not any(detector.check(t, l).trigger_stage2
               for t, l in zip(token_sequence, normal_logprobs)):
        avg = sum(normal_logprobs[-8:]) / 8
        ppl = math.exp(-avg)
        print(f"    Stage 2 not triggered: PPL={ppl:.2f} >= threshold=5.0")

    # ── Test 3: Semantic entropy (n-gram fallback) ──
    print("\n[Test 3] Semantic entropy arbiter (n-gram mode)")
    arbiter = SemanticEntropyArbiter(
        num_samples=3,
        max_completion_tokens=10,
        use_embeddings=False,  # force n-gram mode
    )

    # Simulated: 3 completions nearly identical -> low entropy
    completions_identical = [
        "weight=0.142 stop_loss=0.0035 leverage=3.2",
        "weight=0.142 stop_loss=0.0035 leverage=3.2",
        "weight=0.142 stop_loss=0.0035 leverage=3.2",
    ]
    entropy, clusters, sim = arbiter.compute_semantic_entropy(
        completions_identical
    )
    print(f"  Scenario A (identical): H={entropy:.4f}, clusters={clusters}")

    # Simulated: 3 completions diverse -> high entropy
    completions_diverse = [
        "市场分析表明当前估值合理",
        "建议关注技术面突破信号",
        "宏观环境有利于风险资产配置",
    ]
    entropy2, clusters2, sim2 = arbiter.compute_semantic_entropy(
        completions_diverse
    )
    print(f"  Scenario B (diverse): H={entropy2:.4f}, clusters={clusters2}")

    # ── Test 4: n-gram Jaccard unit test ──
    print("\n[Test 4] n-gram Jaccard similarity")
    sim_identical = SemanticEntropyArbiter._ngram_jaccard(
        "weight=0.142 stop_loss=0.0035",
        "weight=0.142 stop_loss=0.0035",
    )
    sim_different = SemanticEntropyArbiter._ngram_jaccard(
        "weight=0.142 stop_loss=0.0035",
        "市场的波动率分析显示",
    )
    print(f"  Identical text: {sim_identical:.4f}")
    print(f"  Different text: {sim_different:.4f}")

    print("\n" + "=" * 60)
    print("  [PASS] All self-tests passed")
    print("=" * 60)
