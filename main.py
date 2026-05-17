#!/usr/bin/env python
"""Iterative Memory Optimization via Self-Comparison.

Two modes:
  python main.py --mode test                # run on 5 built-in test dialogues
  python main.py --mode user --text "..."   # run on a single user-provided dialogue
  python main.py --mode user --file doc.txt # run on a single user dialogue from file
"""

import argparse, os, sys, time, json

import _ssl_patch
_ssl_patch.apply()

def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ[key.strip()] = val.strip()  # force override, not setdefault
_load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from utils import set_seed, format_duration
from data.dialog_data import load_custom_dataset
from evaluator.ner import EntityExtractor
from generator.llm_client import LLMClient
from optimizer.judge import JudgeLLM
from optimizer.iterator import IterationOptimizer
from visualizer.plotting import plot_summary_dashboard, print_summary_table


def parse_args():
    p = argparse.ArgumentParser(description="Iterative Memory Optimization via Self-Comparison")
    p.add_argument("--mode", type=str, default="test", choices=["test", "user"],
                   help="test=5 built-in dialogues | user=single dialogue from --text/--file")
    p.add_argument("--text", type=str, default=None,
                   help="[user mode] Direct dialogue text input")
    p.add_argument("--file", type=str, default=None,
                   help="[user mode] Path to a .txt or .json file with a single dialogue")
    p.add_argument("--rounds", type=int, default=config.num_rounds)
    p.add_argument("--candidates", type=int, default=config.candidates_per_round)
    p.add_argument("--model", type=str, default=config.generator_model)
    p.add_argument("--judge-model", type=str, default=config.judge_model)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Test evaluators only, no LLM calls")
    p.add_argument("--mock-llm", action="store_true",
                   help="Use mock LLM instead of real API")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=config.seed)
    return p.parse_args()


def build_llm_clients(args):
    """Return (gen_llm, judge_llm) based on args."""
    use_mock = args.mock_llm or (not config.llm_api_key or config.llm_api_key == "sk-your-key-here")
    if use_mock:
        from generator.mock_llm import MockLLMClient
        print("  [LLM] Mock mode")
        mock = MockLLMClient(seed=args.seed)
        return mock, JudgeLLM(llm_client=mock)

    judge_key = config.judge_api_key or config.llm_api_key
    print(f"  [LLM] generator={config.generator_model}, judge={config.judge_model}")
    gen = LLMClient(api_key=config.llm_api_key, base_url=config.llm_base_url, model=config.generator_model)
    judge = JudgeLLM(llm_client=LLMClient(api_key=judge_key, base_url=config.llm_base_url, model=config.judge_model))
    return gen, judge


def run_pipeline(dialogues, output_dir, args):
    """Core pipeline: given a list of {"id":..., "text":...}, run optimization."""
    os.makedirs(output_dir, exist_ok=True)
    config.output_dir = output_dir

    gen_llm, judge_llm = build_llm_clients(args)
    entity_extractor = EntityExtractor()

    optimizer = IterationOptimizer(
        entity_extractor=entity_extractor,
        gen_llm=gen_llm,
        judge_llm=judge_llm,
    )

    start = time.time()
    results = optimizer.run_dataset(
        dialogues=dialogues,
        rounds=args.rounds,
        save_path=os.path.join(output_dir, "results.json"),
    )
    elapsed = time.time() - start
    print(f"\nTotal time: {format_duration(elapsed)}")

    print_summary_table(results)

    if not args.no_plots:
        print(f"\n{'='*60}")
        print("Generating visualizations")
        print("=" * 60)
        plot_summary_dashboard(results, output_dir=output_dir)

    # Print per-dialogue rules
    print(f"\n{'='*60}")
    print("EXPERIENCE RULES")
    print("=" * 60)
    for r in results:
        rules = r.get("experience_rules", [])
        if rules:
            print(f"\n  [{r['dialogue_id']}] ({len(rules)} rules):")
            for i, rule in enumerate(rules[:5]):
                print(f"    {i+1}. {rule}")
            if len(rules) > 5:
                print(f"    ... and {len(rules)-5} more (see {output_dir}/results_experiences.json)")

    return results


# ── Test mode ──────────────────────────────────────────────────
TEST_DATA_PATH = "data/samples/chinese_translated.json"

def cmd_test(args):
    print("\n" + "=" * 60)
    print("TEST MODE — 5 built-in dialogues")
    print("=" * 60)
    if not os.path.exists(TEST_DATA_PATH):
        print(f"ERROR: Test data not found at {TEST_DATA_PATH}")
        print("Run the translation step first or check the path.")
        return

    dataset = load_custom_dataset(TEST_DATA_PATH)
    dialogues = [{"id": d["id"], "text": d["text"]} for d in dataset.dialogues]
    print(f"  {len(dialogues)} dialogues loaded")

    run_pipeline(dialogues, "output/test", args)


# ── User mode ──────────────────────────────────────────────────
def cmd_user(args):
    print("\n" + "=" * 60)
    print("USER MODE — single dialogue")
    print("=" * 60)

    # Get dialogue text
    if args.text:
        text = args.text
        did = "user_input"
    elif args.file:
        path = args.file
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                text = data.get("dialogue") or data.get("text", "")
                did = data.get("id", "user_file")
            elif isinstance(data, list):
                text = data[0].get("dialogue") or data[0].get("text", "")
                did = data[0].get("id", "user_file")
            else:
                text = str(data)
                did = "user_file"
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            did = os.path.splitext(os.path.basename(path))[0]
    else:
        print("ERROR: --mode user requires --text or --file")
        return

    if not text.strip():
        print("ERROR: Empty dialogue text")
        return

    print(f"  Dialogue: {len(text)} chars")
    dialogues = [{"id": did, "text": text}]

    run_pipeline(dialogues, "output/user", args)


# ── Dry run ────────────────────────────────────────────────────
def cmd_dry_run(args):
    print("\n" + "=" * 60)
    print("DRY RUN — AutoMemo evaluator test (no LLM)")
    print("=" * 60)

    from data.dialog_data import create_sample_dataset
    from evaluator.automemo import AutoMemoScorer
    from evaluator.ner import EntityExtractor, compute_entity_f1

    dataset = create_sample_dataset(n=2)
    scorer = AutoMemoScorer()
    ner = EntityExtractor()

    for d in dataset:
        dialogue = d["text"]
        memory = dialogue[:200]

        print(f"\n--- {d['id']} ---")
        print(f"  Dialogue: {len(dialogue)} chars, Memory: {len(memory)} chars")
        entities_d = ner.extract_entities(dialogue)
        entities_m = ner.extract_entities(memory)
        print(f"  Entity F1: {compute_entity_f1(entities_d, entities_m):.4f}")
        scores = scorer.score(dialogue, memory)
        print(f"  Retention: {scores['retention']:.4f}")
        print(f"  Consistency: {scores['consistency']:.4f}")
        print(f"  Composite: {scores['composite']:.4f}")

    print("\n[Dry run] AutoMemo evaluator OK.")


# ── Main ───────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    if args.api_key:
        config.llm_api_key = args.api_key
    if args.base_url:
        config.llm_base_url = args.base_url
    config.num_rounds = args.rounds
    config.candidates_per_round = args.candidates
    config.generator_model = args.model
    config.judge_model = args.judge_model

    if args.dry_run:
        cmd_dry_run(args)
    elif args.mode == "user":
        cmd_user(args)
    else:
        cmd_test(args)


if __name__ == "__main__":
    main()
