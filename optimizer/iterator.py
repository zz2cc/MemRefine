"""Main iterative optimization loop — the core of the project."""

import os
import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from config import config
from generator.llm_client import LLMClient
from generator.prompts import PromptBuilder
from evaluator.automemo import AutoMemoScorer
from evaluator.ner import EntityExtractor, compute_entity_f1
from experience.library import ExperienceLibrary
from optimizer.judge import JudgeLLM


class IterationOptimizer:
    """Orchestrates the full iterative optimization pipeline.

    For each dialogue:
      1. Generate K candidates at varied temperatures
      2. Score all candidates with enhanced BARTScore
      3. Select top-N and bottom-N for judge comparison
      4. Judge extracts writing rules → added to experience library
      5. Experience library injected into next round's prompt
      6. Repeat for N rounds
    """

    def __init__(
        self,
        entity_extractor: EntityExtractor = None,
        gen_llm: LLMClient = None,
        judge_llm: JudgeLLM = None,
        experience_lib: ExperienceLibrary = None,
    ):
        self.automemo = None
        self.entity_extractor = entity_extractor
        self.gen_llm = gen_llm
        self.judge_llm = judge_llm
        self.experience_lib = experience_lib
        self._evaluators_initialized = False

    def _ensure_evaluators(self):
        if self._evaluators_initialized:
            return
        if self.automemo is None:
            self.automemo = AutoMemoScorer()
        if self.entity_extractor is None:
            self.entity_extractor = EntityExtractor()
        if self.gen_llm is None:
            self.gen_llm = LLMClient(model=config.generator_model)
        if self.judge_llm is None:
            self.judge_llm = JudgeLLM()
        self._evaluators_initialized = True

    def compute_enhanced_score(self, dialogue: str, memory: str) -> Dict[str, float]:
        """AutoMemo: retention (70%) + entity_f1 (30%)."""
        self._ensure_evaluators()

        am_scores = self.automemo.score(dialogue, memory)
        retention = am_scores["retention"]
        consistency = am_scores["consistency"]

        entities_dialogue = self.entity_extractor.extract_entities(dialogue)
        entities_memory = self.entity_extractor.extract_entities(memory)
        entity_f1 = compute_entity_f1(entities_dialogue, entities_memory)

        composite = 0.70 * retention + 0.30 * entity_f1

        return {
            "composite": composite,
            "retention": retention,
            "entity_f1": entity_f1,
            "consistency": consistency,
            "bartscore": retention,   # alias for weakness detection
            "nli_score": consistency, # alias for weakness detection
        }

    WEAKNESS_TAG_MAP = {
        "bartscore":    ["COMPLETENESS", "STYLE"],       # low coverage → need more content + source words
        "retention":    ["COMPLETENESS", "STYLE"],       # ditto, AutoMemo alias
        "entity_f1":    ["COMPLETENESS"],                 # missing entities → need entity retention
        "nli_score":    ["ACCURACY"],                     # hallucinations → need anti-fabrication
        "consistency":  ["ACCURACY"],                     # ditto, AutoMemo alias
    }

    def generate_candidates(
        self, dialogue: str, n: int = None, base_temperature: float = 0.8,
        focus_tags: List[str] = None,
    ) -> List[str]:
        """Generate N candidate memory documents at varied temperatures.

        If focus_tags is provided, only inject experience rules matching those tags,
        targeting the current scoring weakness rather than injecting everything.
        """
        n = n or config.candidates_per_round
        self._ensure_evaluators()

        # Filter experience library by weakness tags if provided
        if focus_tags and self.experience_lib:
            experiences = self.experience_lib.get_best(config.experience_top_n, tags=focus_tags)
        elif self.experience_lib:
            experiences = self.experience_lib.get_best(config.experience_top_n)
        else:
            experiences = None

        messages = PromptBuilder.build_generation_messages(dialogue, experiences)

        # Generate at varied temperatures for diversity
        temperatures = np.linspace(base_temperature - 0.2, base_temperature + 0.2, n)
        temperatures = np.clip(temperatures, 0.3, 1.2)

        return self.gen_llm.generate_with_temperatures(messages, list(temperatures))

    def run_one_dialogue(
        self,
        dialogue: str,
        dialogue_id: str = "unknown",
        rounds: int = None,
        experience_lib = None,  # per-dialogue library (created fresh if None)
    ) -> Dict:
        """Run the full iterative optimization on a single dialogue.

        Each dialogue gets its OWN experience library — rules from horror scripts
        don't pollute medical Q&A, and vice versa.

        Returns a dict with:
          - rounds: list of per-round metrics
          - best_memory: the highest-scoring memory document
          - best_score: its composite score
          - score_trajectory: list of best scores per round
          - experience_library: the dialogue's personal rule set
        """
        rounds = rounds or config.num_rounds
        self._ensure_evaluators()

        # Per-dialogue experience library
        if experience_lib is None:
            experience_lib = ExperienceLibrary()
        self.experience_lib = experience_lib

        history = []
        best_memory = None
        best_score = -float("inf")
        best_round = -1
        no_improvement_count = 0
        prev_round_best = -float("inf")
        focus_tags = None  # Round 0: no focus, all rules injected

        for r in range(rounds):
            round_start = time.time()

            # Temperature annealing: explore (high T) → exploit (low T)
            # Paper: T=0.7 exploration, T=0.3 inference
            base_temp = 0.8 - (0.8 - 0.3) * (r / max(rounds - 1, 1))

            # --- Phase 1: Generate candidates (with weakness-targeted rules) ---
            candidates = self.generate_candidates(dialogue, focus_tags=focus_tags,
                                                   base_temperature=base_temp)

            # --- Phase 2: Score all candidates ---
            scored = []
            for cand in candidates:
                scores = self.compute_enhanced_score(dialogue, cand)
                scored.append((cand, scores))

            scored.sort(key=lambda x: x[1]["composite"], reverse=True)

            round_best_mem, round_best_scores = scored[0]
            round_worst_mem, round_worst_scores = scored[-1]
            round_avg = np.mean([s[1]["composite"] for s in scored])

            # --- Phase 3: Judge comparison (if we have enough spread) ---
            high_k = min(config.compare_top_k, len(scored))
            low_k = min(config.compare_bottom_k, len(scored))
            high_mems = [scored[i][0] for i in range(high_k)]
            low_mems = [scored[i][0] for i in range(low_k, 0, -1)]  # from worst up

            high_avg_score = np.mean([scored[i][1]["composite"] for i in range(high_k)])
            low_avg_score = np.mean([scored[i][1]["composite"] for i in range(-low_k, 0)])

            # Always call judge — even small gaps yield useful rules
            judge_output, new_rules = self.judge_llm.compare_and_extract(
                dialogue=dialogue,
                high_memories=high_mems,
                low_memories=low_mems,
                high_score=high_avg_score,
                low_score=low_avg_score,
            )

            # --- Phase 4: Detect weakness for next round ---
            # Which metric component is the lowest? Target that with matching [Tag] rules.
            components = {
                "bartscore": round_best_scores.get("bartscore", 0),
                "entity_f1": round_best_scores.get("entity_f1", 0),
                "nli_score": round_best_scores.get("nli_score", 0),
            }
            weakest = min(components, key=components.get)
            focus_tags = self.WEAKNESS_TAG_MAP.get(weakest, None)
            # Store for logging
            round_weakness = weakest

            # --- Phase 5: ADD/DELETE/MODIFY/KEEP ---
            round_delta = round_best_scores["composite"] - prev_round_best
            prev_round_best = round_best_scores["composite"]

            rules_added = 0
            if self.experience_lib:
                # KEEP: mark rules that survived this round
                self.experience_lib.mark_survivors()

                # DELETE: remove rules with negative impact over 2+ rounds
                pruned = self.experience_lib.prune_negative(min_rounds=2)

                # ADD: new rules, capped for gradual growth
                max_add = config.rules_per_round_max
                for rule in new_rules:
                    if rules_added >= max_add:
                        break
                    if self.experience_lib.add(rule, score_delta=round_delta):
                        rules_added += 1

                # MODIFY: add() internally calls _evict_if_needed() which
                # merges similar rules when library hits max_size

            # --- Track best ---
            if round_best_scores["composite"] > best_score:
                best_score = round_best_scores["composite"]
                best_memory = round_best_mem
                best_round = r
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            round_time = time.time() - round_start

            round_entry = {
                "round": r,
                "num_candidates": len(candidates),
                "best_score": round_best_scores,
                "worst_score": round_worst_scores,
                "avg_score": round_avg,
                "score_spread": round_best_scores["composite"] - round_worst_scores["composite"],
                "new_rules": new_rules,
                "rules_added": rules_added,
                "judge_output": judge_output,
                "experience_count": len(self.experience_lib.rules) if self.experience_lib else 0,
                "round_time_s": round_time,
                "best_memory": round_best_mem,
                "weakness": round_weakness,
            }
            history.append(round_entry)

            tag_str = f"→{'+'.join(focus_tags)}" if focus_tags else "→all"
            print(f"  [{dialogue_id}] Round {r}: best={best_score:.4f}, "
                  f"avg={round_avg:.4f}, lib={round_entry['experience_count']} "
                  f"(+{rules_added}), weak={weakest} {tag_str}, {round_time:.1f}s")

            # Early stopping
            if no_improvement_count >= config.early_stop_patience:
                print(f"  [{dialogue_id}] Early stop at round {r}")
                break

        # Build score trajectory
        score_trajectory = [h["best_score"]["composite"] for h in history]

        return {
            "dialogue_id": dialogue_id,
            "dialogue": dialogue,
            "best_memory": best_memory,
            "best_score": best_score,
            "best_round": best_round,
            "score_trajectory": score_trajectory,
            "history": history,
            "experience_count": len(self.experience_lib.rules) if self.experience_lib else 0,
            "experience_rules": list(self.experience_lib.rules) if self.experience_lib else [],
        }

    def run_dataset(
        self,
        dialogues: List[Dict[str, str]],
        rounds: int = None,
        save_path: str = None,
    ) -> List[Dict]:
        """Multi-round iteration — each dialogue has its OWN experience library.

        Dialogue A runs 5 rounds with its personal library.
        Dialogue B starts fresh with a new empty library.
        No cross-contamination between different domains.
        """
        rounds = rounds or config.num_rounds
        self._ensure_evaluators()

        results = []
        print(f"\n{'='*60}")
        print(f"IterationOptimizer: {len(dialogues)} dialogues × {rounds} rounds "
              f"(per-dialogue libraries, {config.experience_max_size} rules max)")
        print(f"{'='*60}\n")

        for i, d in enumerate(tqdm(dialogues, desc="Dialogues")):
            result = self.run_one_dialogue(
                dialogue=d["text"],
                dialogue_id=d.get("id", f"dial_{i}"),
                rounds=rounds,
                experience_lib=ExperienceLibrary(),  # FRESH library per dialogue
            )
            results.append(result)

        if save_path:
            self.save_results(results, save_path)

        return results

    def save_results(self, results: List[Dict], path: str):
        """Save results to JSON, stripping heavy fields."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        # Make serializable: keep essential info, strip full texts from history
        serializable = []
        for r in results:
            entry = {
                "dialogue_id": r["dialogue_id"],
                "best_score": r.get("best_score", 0),
                "best_round": r.get("best_round", 0),
                "score_trajectory": r["score_trajectory"],
                "experience_count": r["experience_count"],
                "best_memory": r.get("best_memory", ""),
                "dialogue": r["dialogue"][:500] + "..." if len(r["dialogue"]) > 500 else r["dialogue"],
            }
            # Keep per-round summary without full judge output
            entry["history"] = []
            for h in r["history"]:
                entry["history"].append({
                    "round": h["round"],
                    "best_score": h["best_score"],
                    "worst_score": h["worst_score"],
                    "avg_score": h["avg_score"],
                    "score_spread": h["score_spread"],
                    "experience_count": h["experience_count"],
                    "num_new_rules": len(h.get("new_rules", [])),
                    "round_time_s": h["round_time_s"],
                })
            serializable.append(entry)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {path}")

        # Save per-dialogue experience libraries
        all_rules = {}
        for r in results:
            rules = r.get("experience_rules", [])
            if rules:
                all_rules[r["dialogue_id"]] = rules
        if all_rules:
            exp_path = path.replace(".json", "_experiences.json")
            import json as _json
            with open(exp_path, "w", encoding="utf-8") as f:
                _json.dump(all_rules, f, ensure_ascii=False, indent=2)
            total_rules = sum(len(v) for v in all_rules.values())
            print(f"Per-dialogue experience libraries ({total_rules} total rules) saved to {exp_path}")
