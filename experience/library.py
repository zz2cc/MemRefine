"""Experience library with ADD/DELETE/MODIFY/KEEP operations."""

import hashlib
import numpy as np
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from config import config

# Try importing, fall back gracefully
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None


class ExperienceLibrary:
    """Dynamic text rule collection for guiding LLM memory generation.

    Supports the four operations from the paper:
    - ADD: Append a new rule (with dedup)
    - DELETE: Remove a rule that degraded performance
    - MODIFY: Refine an existing rule
    - KEEP: Retain a rule unchanged

    Rules are plain text strings, stored in order, and injected into generation prompts.
    """

    def __init__(
        self,
        max_size: int = None,
        dedup_threshold: float = None,
    ):
        self.max_size = max_size or config.experience_max_size
        self.dedup_threshold = dedup_threshold or config.dedup_threshold
        self.rules: List[str] = []
        self.scores: List[float] = []          # performance delta when rule was added
        self.usage_count: List[int] = []       # how many times each rule has been kept
        self._embedder = None  # lazy load
        self._embeddings: Optional[np.ndarray] = None  # cached embeddings

    def _get_embedder(self):
        if self._embedder is None:
            try:
                print("[Experience] Loading sentence embedding model ...")
                self._embedder = SentenceTransformer(
                    config.embedding_model, device="cpu"
                )
                # Recompute embeddings if we have existing rules
                if self.rules:
                    self._embeddings = self._embedder.encode(self.rules)
            except Exception as e:
                print(f"[Experience] Could not load embedding model: {e}")
                print("[Experience] Using token-overlap fallback for dedup")
                self._embedder = False  # sentinel: fallback mode
        return self._embedder

    @staticmethod
    def _token_overlap_similarity(text1: str, text2: str) -> float:
        """Jaccard similarity over word tokens (fallback when no embedding model)."""
        t1 = set(text1.lower().split())
        t2 = set(text2.lower().split())
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    def _compute_similarity(self, new_rule: str) -> float:
        """Compute maximum similarity of new_rule against existing rules."""
        if not self.rules:
            return 0.0
        embedder = self._get_embedder()
        if embedder is False:
            # Fallback: token overlap
            return max(self._token_overlap_similarity(new_rule, r) for r in self.rules)

        new_emb = embedder.encode([new_rule])
        if self._embeddings is None or len(self._embeddings) != len(self.rules):
            self._embeddings = embedder.encode(self.rules)
        sims = cosine_similarity(new_emb, self._embeddings)[0]
        return float(np.max(sims))

    def add(self, rule: str, score_delta: float = 0.0) -> bool:
        """ADD operation: append a rule if not too similar to existing ones.

        Returns True if the rule was actually added.
        """
        rule = rule.strip()
        if len(rule) < 5:
            return False

        # Dedup check
        if self.rules and self._compute_similarity(rule) > self.dedup_threshold:
            return False

        # Exact duplicate check (as fallback)
        if rule.lower() in {r.lower() for r in self.rules}:
            return False

        self.rules.append(rule)
        self.scores.append(score_delta)
        self.usage_count.append(0)

        # Update cached embeddings (skip if fallback mode)
        if self._embedder is not None and self._embedder is not False:
            new_emb = self._embedder.encode([rule])
            if self._embeddings is not None:
                self._embeddings = np.vstack([self._embeddings, new_emb])
            else:
                self._embeddings = new_emb

        # FIFO eviction if over capacity
        self._evict_if_needed()
        return True

    def add_batch(self, rules: List[str], score_delta: float = 0.0) -> int:
        """Add multiple rules; returns number actually added."""
        added = 0
        for rule in rules:
            if self.add(rule, score_delta):
                added += 1
        return added

    def delete(self, index: int) -> bool:
        """DELETE operation: remove a rule by index."""
        if 0 <= index < len(self.rules):
            self.rules.pop(index)
            self.scores.pop(index)
            self.usage_count.pop(index)
            # Invalidate embeddings cache
            self._embeddings = None
            return True
        return False

    def delete_by_similarity(self, rule: str, threshold: float = 0.9) -> int:
        """Remove rules very similar to the given one. Returns count removed."""
        embedder = self._get_embedder()
        target_emb = embedder.encode([rule])
        sims = cosine_similarity(target_emb, embedder.encode(self.rules))[0]
        to_remove = [i for i, s in enumerate(sims) if s > threshold]
        for i in sorted(to_remove, reverse=True):
            self.delete(i)
        return len(to_remove)

    def modify(self, index: int, new_rule: str) -> bool:
        """MODIFY operation: replace an existing rule with a refined version."""
        if 0 <= index < len(self.rules):
            old_rule = self.rules[index]
            if new_rule.strip().lower() != old_rule.lower():
                self.rules[index] = new_rule.strip()
                self.usage_count[index] = 0
                self._embeddings = None
                return True
        return False

    def keep(self, index: int) -> None:
        """KEEP operation: increment usage count for a rule that proves useful."""
        if 0 <= index < len(self.usage_count):
            self.usage_count[index] += 1

    def get_recent(self, n: int = None, tags: List[str] = None) -> List[str]:
        """Return the N most recent rules, optionally filtered by [Tag]."""
        n = n or config.experience_top_n
        candidates = self.rules
        if tags:
            candidates = [r for r in candidates if any(f"[{t}]" in r for t in tags)]
        return candidates[-n:] if candidates else []

    def get_best(self, n: int = None, tags: List[str] = None) -> List[str]:
        """Return rules with highest accumulated scores, optionally filtered by tag."""
        n = n or config.experience_top_n
        if not self.rules:
            return []
        candidates = list(range(len(self.rules)))
        if tags:
            candidates = [i for i in candidates if any(f"[{t}]" in self.rules[i] for t in tags)]
        if not candidates:
            return []
        weighted = [
            (i, self.scores[i] * (1.0 + min(self.usage_count[i], 10) / 10.0))
            for i in candidates
        ]
        indices = [i for i, _ in sorted(weighted, key=lambda x: x[1], reverse=True)[:n]]
        return [self.rules[i] for i in indices]

    def get_tag_distribution(self) -> dict:
        """Count rules per tag category."""
        import re
        counts = {}
        for r in self.rules:
            match = re.match(r'\[(ACCURACY|COMPLETENESS|STRUCTURE|STYLE)\]', r)
            if match:
                tag = match.group(1)
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def _evict_if_needed(self):
        """Score-based + similarity-aware eviction.

        1. If two rules are near-duplicates (cos > 0.75), keep the higher-scored one,
           delete the other. This is the MODIFY operation — merging by elimination.
        2. When still over capacity, remove the lowest-scoring rule.
        """
        # Step 1: Merge near-duplicates (MODIFY)
        if len(self.rules) >= 2:
            embedder = self._get_embedder()
            if embedder and embedder is not False:
                embs = embedder.encode(self.rules)
                deleted = set()
                for i in range(len(self.rules)):
                    if i in deleted:
                        continue
                    for j in range(i + 1, len(self.rules)):
                        if j in deleted:
                            continue
                        sim = float(np.dot(embs[i], embs[j]) /
                                    (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j])))
                        if sim > 0.75:
                            # Keep the higher-scored one, delete the other
                            qi = self.scores[i] * (1.0 + min(self.usage_count[i], 10) / 10.0)
                            qj = self.scores[j] * (1.0 + min(self.usage_count[j], 10) / 10.0)
                            if qi >= qj:
                                deleted.add(j)
                            else:
                                deleted.add(i)

                if deleted:
                    for idx in sorted(deleted, reverse=True):
                        self.rules.pop(idx)
                        self.scores.pop(idx)
                        self.usage_count.pop(idx)
                    self._embeddings = None

        # Step 2: Score-based eviction for remaining over-capacity rules
        while len(self.rules) > self.max_size:
            quality = [
                self.scores[i] * (1.0 + min(self.usage_count[i], 10) / 10.0)
                for i in range(len(self.rules))
            ]
            worst_idx = int(np.argmin(quality))
            self.rules.pop(worst_idx)
            self.scores.pop(worst_idx)
            self.usage_count.pop(worst_idx)
            self._embeddings = None

    def mark_survivors(self):
        """Increment usage_count for all current rules — called each round.
        Rules that survive more rounds get higher usage_count, making them
        harder to prune (KEEP effect)."""
        for i in range(len(self.usage_count)):
            self.usage_count[i] += 1

    def prune_negative(self, min_rounds: int = 2) -> int:
        """DELETE: remove rules with negative score_delta that have survived
        at least min_rounds without contributing. Returns count removed.
        """
        to_remove = [
            i for i in range(len(self.rules))
            if self.scores[i] < -0.005 and self.usage_count[i] >= min_rounds
        ]
        for i in sorted(to_remove, reverse=True):
            self.rules.pop(i)
            self.scores.pop(i)
            self.usage_count.pop(i)
        if to_remove:
            self._embeddings = None
        return len(to_remove)

    def summarize(self) -> str:
        """Return a human-readable summary of the library state."""
        return (
            f"ExperienceLibrary: {len(self.rules)} rules (max {self.max_size}), "
            f"avg score delta: {np.mean(self.scores):.4f}" if self.scores else "no rules yet"
        )

    def to_dict(self) -> dict:
        return {
            "rules": self.rules,
            "scores": self.scores,
            "usage_count": self.usage_count,
        }

    def save(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.rules = data.get("rules", [])
        self.scores = data.get("scores", [])
        self.usage_count = data.get("usage_count", [])
        self._embeddings = None
