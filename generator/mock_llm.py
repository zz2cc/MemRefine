"""Mock LLM client for testing without API calls.

Generates plausible memory documents and judge analyses using template-based
responses, allowing the full pipeline to be tested offline.
"""

import random
import re
from typing import List, Dict


class MockLLMClient:
    """A mock LLM that generates template-based responses for testing the pipeline."""

    def __init__(self, model: str = "mock", seed: int = 42):
        self.model = model
        self.rng = random.Random(seed)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 2048,
        model: str = None,
    ) -> str:
        """Generate a mock response based on the message content."""
        # Determine context: generation vs judging
        user_content = ""
        for m in messages:
            if m["role"] == "user":
                user_content = m["content"]
                break

        if "Memory Document:" in user_content:
            return self._mock_generate_memory(user_content, temperature)
        elif "HIGH-SCORING MEMORIES" in user_content:
            return self._mock_judge_analysis()
        else:
            return "Mock response for unhandled prompt type."

    def _mock_generate_memory(self, prompt: str, temperature: float) -> str:
        """Generate a mock memory document from the dialogue."""
        # Extract dialogue text from the prompt
        dialogue = ""
        match = re.search(r'=== CONVERSATION ===\n(.*?)=== END CONVERSATION ===', prompt, re.DOTALL)
        if match:
            dialogue = match.group(1).strip()

        # Extract key information heuristically
        names = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b', dialogue)
        numbers = re.findall(r'\b(?:\$?\d+(?:\.\d+)?|#?\d{3,})\b', dialogue)
        dates = re.findall(r'\b(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b', dialogue)

        # Variable quality based on temperature + randomness
        quality = 0.45 + self.rng.random() * 0.45  # random in [0.45, 0.90]

        # Build mock memory
        lines = ["## Summary"]

        # Extract a text snippet as the "summary"
        sentences = [s.strip() for s in re.split(r'[.!?]+', dialogue) if len(s.strip().split()) > 5]
        if sentences:
            summary_idx = self.rng.randint(0, min(len(sentences) - 1, 5))
            summary = sentences[summary_idx]
            if len(summary) > 150:
                summary = summary[:150] + "..."
            lines.append(summary)
        else:
            lines.append("A conversation between customer and agent.")

        lines.append("\n## Key Facts")
        facts_count = max(1, int(3 + self.rng.random() * 5 * quality))
        available = list(set(sentences)) if sentences else ["Details discussed during the conversation."]
        self.rng.shuffle(available)
        for i in range(min(facts_count, len(available))):
            s = available[i]
            if len(s) > 120:
                s = s[:120] + "..."
            lines.append(f"- {s}")

        if names and self.rng.random() < 0.8:
            lines.append("\n## Parties Involved")
            self.rng.shuffle(names)
            for name in names[:self.rng.randint(1, min(4, len(names)))]:
                lines.append(f"- {name}")

        if (numbers or dates) and self.rng.random() < 0.85:
            lines.append("\n## Important Details")
            items = list(numbers) + list(dates)
            self.rng.shuffle(items)
            for item in items[:self.rng.randint(1, min(6, len(items)))]:
                lines.append(f"- {item}")

        if self.rng.random() < 0.7:
            lines.append("\n## Action Items")
            actions = [
                "- Follow up as needed",
                "- Document the resolution",
                "- Schedule follow-up call",
                "- Send confirmation email",
                "- Update account notes",
            ]
            self.rng.shuffle(actions)
            for a in actions[:self.rng.randint(1, 3)]:
                lines.append(a)

        result = "\n".join(lines)

        # At low quality, introduce issues
        if quality < 0.55:
            # Drop some sections
            if self.rng.random() < 0.4:
                lines = [l for l in lines if not l.startswith("## Key Facts") and not l.startswith("- ")]
            if self.rng.random() < 0.4:
                lines.append("- (Unverified claim)")
        elif quality < 0.65:
            # Slightly fewer facts
            pass

        return result

    def _mock_judge_analysis(self) -> str:
        """Generate mock judge analysis with writing rules."""
        rules = [
            "请务必在开头提供简洁的摘要，概括对话的核心主题",
            "注意记录所有提到的数字、日期和金额，不得遗漏关键信息",
            "请务必明确列出所有参与方及其角色",
            "注意避免编造对话中未出现的事实或细节",
            "请务必使用结构化的格式（摘要、关键事实、决策、行动项）",
        ]
        self.rng.shuffle(rules)
        return "\n".join(rules[:self.rng.randint(3, 5)])

    def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.8,
        model: str = None,
    ) -> List[str]:
        return [self.chat(msgs, temperature=temperature, model=model) for msgs in messages_list]

    def generate_with_temperatures(
        self,
        messages: List[Dict[str, str]],
        temperatures: List[float],
        model: str = None,
    ) -> List[str]:
        return [self.chat(messages, temperature=t, model=model) for t in temperatures]
