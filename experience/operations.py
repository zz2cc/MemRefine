"""Experience library operations (ADD/DELETE/MODIFY/KEEP)."""

from enum import Enum
from typing import List, Tuple
from experience.library import ExperienceLibrary


class ExperienceOperation(Enum):
    ADD = "add"
    DELETE = "delete"
    MODIFY = "modify"
    KEEP = "keep"


class ExperienceManager:
    """High-level manager that applies the four operations to the library.

    Decides which operations to apply based on judge output and round-over-round
    performance deltas.
    """

    def __init__(self, library: ExperienceLibrary):
        self.library = library

    def ingest_judge_rules(
        self, new_rules: List[str], round_delta: float
    ) -> int:
        """ADD new rules from judge, weighted by the round's score improvement."""
        return self.library.add_batch(new_rules, score_delta=round_delta)

    def prune_low_performance(self, threshold: float = -0.01) -> int:
        """DELETE rules associated with negative or negligible score deltas."""
        removed = 0
        indices_to_remove = [
            i for i, s in enumerate(self.library.scores) if s < threshold
        ]
        for i in sorted(indices_to_remove, reverse=True):
            self.library.delete(i)
            removed += 1
        return removed

    def reinforce_useful(self, rules_used: List[int]) -> None:
        """KEEP: mark rules that were part of the winning prompt as useful."""
        for idx in rules_used:
            if 0 <= idx < len(self.library.rules):
                self.library.keep(idx)

    def consolidate(self, judge_output: str, round_delta: float) -> dict:
        """Run a full consolidation cycle after each round.

        Returns summary dict:
          - rules_added: int
          - rules_deleted: int
          - total_rules: int
        """
        from generator.prompts import PromptBuilder
        new_rules = PromptBuilder.parse_judge_output(judge_output)
        added = self.ingest_judge_rules(new_rules, round_delta)
        deleted = self.prune_low_performance(threshold=-0.005)

        return {
            "rules_added": added,
            "rules_deleted": deleted,
            "total_rules": len(self.library.rules),
            "new_rules": new_rules,
        }
