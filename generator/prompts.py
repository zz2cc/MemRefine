"""Prompt templates for memory generation and judge comparison."""

from typing import List, Optional


class PromptBuilder:
    """Builds prompts for generation and judging, including experience library injection."""

    SYSTEM_PROMPT_BASE = """You are an expert conversation analyst. Your task is to write a concise,
accurate memory document summarizing a conversation. The memory should:

1. Record all key facts: names, dates, numbers, locations, decisions, and action items.
2. Be structured with clear sections: Summary, Key Facts, Decisions, Action Items.
3. Never invent information not present in the original conversation.
4. Be self-contained — a reader should understand the conversation without seeing the original.
5. Use bullet points for clarity where appropriate."""

    GENERATION_USER_TEMPLATE = """{experience_section}Please read the following conversation and produce a memory document.

=== CONVERSATION ===
{dialogue}
=== END CONVERSATION ===

Memory Document:"""

    JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of conversation memory documents. You
compare high-scoring and low-scoring memory documents to extract general writing principles.

A memory document is a compressed summary of a dialogue — it should retain all key information
(names, numbers, decisions, action items) while removing redundant chat. The scoring function
rewards: information retention, factual fidelity to the source, and absence of fabrication.
It penalizes: missing key details, invented facts, and content that drifts from the original.

Your analysis should be THOROUGH and MULTI-LEVEL. Identify CONCRETE DIFFERENCES between the
high and low scoring documents. Pick the 3 most impactful differences and formulate a
transferable writing rule for each."""

    JUDGE_USER_TEMPLATE = """Compare the following two sets of memory documents for the same conversation.

=== ORIGINAL CONVERSATION ===
{dialogue}
=== END CONVERSATION ===

=== HIGH-SCORING MEMORIES (score: {high_score:.3f}) ===
{high_memories}
=== END HIGH-SCORING ===

=== LOW-SCORING MEMORIES (score: {low_score:.3f}) ===
{low_memories}
=== END LOW-SCORING ===

Step 1 — Identify CONCRETE WORD-LEVEL differences:
- Which specific facts/numbers/names from the dialogue appear in high-score but are MISSING in low-score?
- What words or phrases did low-score INVENT that never appear in the dialogue?
- Where does high-score retain the dialogue's own phrasing vs low-score's loose paraphrasing?
- Which document is more information-dense (more dialogue facts per word)?

Step 2 — For each difference, derive a general writing rule. Output exactly 3 rules — the most impactful ones.
Categorize with [Tag]:

[COMPLETENESS] — rules about capturing ALL key information from the source
[ACCURACY] — rules about avoiding fabrication, hallucination, distortion
[STRUCTURE] — rules about organization that preserves information density
[STYLE] — rules about word choice, conciseness, and fidelity to source phrasing

Format each rule as:
[Tag] 请务必[具体做法] 或 [Tag] 注意[避免事项]

Rules should capture WRITING PATTERNS specific to this type of content — concrete enough
to directly guide the writer, but focused on technique rather than specific entities.
A good rule teaches HOW to write, not WHAT to write.

Good examples:
[ACCURACY] 注意不得编造对话中未出现的金额数字或订单号，所有数据必须与原文严格一致
[COMPLETENESS] 请务必记录所有参与方的全名和身份，不可只写模糊称谓
[STYLE] 注意保留原文中的情绪暗示词和互动细节，不要用中性词替换

Now output your analysis and rules:"""

    @staticmethod
    def build_generation_messages(
        dialogue: str,
        experience_entries: Optional[List[str]] = None,
        top_n: int = 20,
    ) -> list:
        """Build messages for memory generation, with optional experience injection."""
        # Build experience section
        experience_section = ""
        if experience_entries:
            recent = experience_entries[-top_n:]
            rules = "\n".join(f"  {i+1}. {rule}" for i, rule in enumerate(recent))
            experience_section = (
                "IMPORTANT WRITING RULES (learned from previous iterations):\n"
                f"{rules}\n\n"
            )

        user_prompt = PromptBuilder.GENERATION_USER_TEMPLATE.format(
            experience_section=experience_section,
            dialogue=dialogue,
        )

        return [
            {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT_BASE},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def build_judge_messages(
        dialogue: str,
        high_memories: List[str],
        low_memories: List[str],
        high_score: float,
        low_score: float,
    ) -> list:
        """Build messages for the judge LLM to compare memories and extract rules."""
        high_text = "\n\n---\n\n".join(f"[Memory {i+1}]\n{m}" for i, m in enumerate(high_memories))
        low_text = "\n\n---\n\n".join(f"[Memory {i+1}]\n{m}" for i, m in enumerate(low_memories))

        user_prompt = PromptBuilder.JUDGE_USER_TEMPLATE.format(
            dialogue=dialogue,
            high_memories=high_text,
            low_memories=low_text,
            high_score=high_score,
            low_score=low_score,
        )

        return [
            {"role": "system", "content": PromptBuilder.JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def parse_judge_output(output: str) -> List[str]:
        """Parse judge LLM output into a list of experience rules.

        Handles multiple formats:
          [ACCURACY] 请务必[具体做法]
          [COMPLETENESS] 注意[避免事项]
          - 请务必[rule]
          1. 注意[rule]
        """
        import re

        rules = []
        for line in output.split("\n"):
            raw = line.strip()
            if not raw or len(raw) < 5:
                continue

            # Pattern 1: [Tag] prefix format (new style)
            tag_match = re.match(r'\[(ACCURACY|COMPLETENESS|STRUCTURE|STYLE)\]\s*(.+)', raw)
            if tag_match:
                tag = tag_match.group(1)
                rule_body = tag_match.group(2).strip()
                # Clean up trailing punctuation
                rule_body = rule_body.rstrip("。，.!?,;；")
                if len(rule_body) >= 5:
                    rules.append(f"[{tag}] {rule_body}")
                continue

            # Pattern 2: Lines starting with 请务必 or 注意 (old style, still supported)
            if raw.startswith("请务必") or raw.startswith("注意"):
                for prefix in ["请务必", "注意"]:
                    if raw.startswith(prefix):
                        rule_body = raw[len(prefix):].strip()
                        # Remove leading numbering/punctuation
                        while rule_body and (rule_body[0].isdigit() or rule_body[0] in ".、-: "):
                            rule_body = rule_body.lstrip("0123456789.、-: ")
                        rule_body = rule_body.rstrip("。，.!?,;；")
                        if rule_body and len(rule_body) >= 3:
                            rules.append(f"{prefix}{rule_body}")
                        break
                continue

            # Pattern 3: Numbered list items with rule-like content
            numbered_match = re.match(r'^(\d+)[.\)、]\s*(.+)', raw)
            if numbered_match:
                rule_body = numbered_match.group(2).strip()
                rule_body = rule_body.rstrip("。，.!?,;；")
                if rule_body and len(rule_body) >= 5:
                    rules.append(rule_body)
                continue

        # Deduplicate within this batch
        seen = set()
        unique_rules = []
        for r in rules:
            if r not in seen:
                seen.add(r)
                unique_rules.append(r)

        return unique_rules
