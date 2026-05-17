"""Judge LLM: compares high/low scoring memories and extracts writing rules."""

import numpy as np
from typing import List, Tuple
from generator.llm_client import LLMClient
from generator.prompts import PromptBuilder
from config import config


class JudgeLLM:
    """Uses an LLM to compare high and low scoring memory documents,
    then extracts actionable writing rules (experiences)."""

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient(model=config.judge_model)

    def compare_and_extract(
        self,
        dialogue: str,
        high_memories: List[str],
        low_memories: List[str],
        high_score: float,
        low_score: float,
    ) -> Tuple[str, List[str]]:
        """Run judge comparison and return (raw_output, parsed_rules)."""
        messages = PromptBuilder.build_judge_messages(
            dialogue=dialogue,
            high_memories=high_memories,
            low_memories=low_memories,
            high_score=high_score,
            low_score=low_score,
        )
        raw_output = self.llm.chat(messages, temperature=0.3)
        rules = PromptBuilder.parse_judge_output(raw_output)
        return raw_output, rules
