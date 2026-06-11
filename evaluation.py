"""
Lab 1 - Part C: Evaluation Module (Groq API version)
=====================================================
Evaluates generated stories/poems using automatic and human metrics.

Automatic Metrics:
  - BLEU (n-gram precision via NLTK)
  - ROUGE (recall-oriented overlap: ROUGE-1, ROUGE-2, ROUGE-L)
  - Perplexity (computed via local GPT-2 small model — standard practice)

Human Evaluation:
  - Coherence, Creativity, Fluency, Relevance to Prompt (1-5 scale)
"""

import json
import math
import os
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import nltk
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# Reference texts for BLEU/ROUGE evaluation
# ---------------------------------------------------------------------------
# These reference texts serve as "gold standard" outputs for each prompt.
# In creative generation, exact match isn't expected — these capture the
# thematic content and style we'd want from each prompt.

REFERENCE_TEXTS = {
    "In a world where dreams could be harvested, a young girl discovered she had the rarest dream of all": (
        "In the sprawling city of Somnia, dreams were currency. Every night, "
        "harvesters would enter the sleeping minds of citizens, carefully extracting "
        "the shimmering threads of their nocturnal visions. Common dreams — of flying, "
        "of falling, of forgotten exams — filled the markets like copper coins. But "
        "rare dreams, the ones born of pure imagination, were worth fortunes. "
        "Young Elara had always slept deeply, her dreams vivid and wild. When the "
        "harvesters came to test her, their instruments blazed with light. She possessed "
        "the dream of creation itself — the ability to dream new worlds into existence. "
        "The harvesters trembled, for such a dream had not been seen in a thousand years. "
        "Elara stood at the threshold of a power that could reshape reality, and she "
        "knew that nothing would ever be the same again."
    ),
    "moonlight, forgotten castle, ancient magic, whispers": (
        "Under the pale moonlight, the forgotten castle emerged from the mist like a "
        "memory long buried. Its towers, once proud sentinels of a kingdom now lost to "
        "time, stood crumbling against the star-scattered sky. Within its walls, ancient "
        "magic still pulsed — a faint heartbeat in the stones, a shimmer in the dust "
        "that danced through broken windows. Whispers filled the corridors, voices of "
        "those who had lived and loved and perished within these halls. They spoke of "
        "battles fought with sword and spell, of a queen who had woven enchantments "
        "into the very foundation. And now, as the moonlight crept through the grand "
        "entrance, the whispers grew louder, as if the castle itself was awakening "
        "from its centuries-long slumber, calling to those brave enough to listen."
    ),
    "The last autumn leaf": (
        "The last autumn leaf clung to the old oak tree with a stubbornness that defied "
        "the bitter wind. All around it, the world had surrendered to the coming winter — "
        "branches stood bare like skeletal fingers against the grey sky, and the ground "
        "was carpeted in a mosaic of amber, crimson, and gold. But this one leaf held on. "
        "It was small and weathered, its edges curled like the pages of an ancient book, "
        "its color a deep, burnished copper. A young boy sat beneath the tree each day, "
        "watching it flutter and twist but never fall. He saw in that leaf a kindred "
        "spirit — a small thing refusing to let go, finding beauty in persistence. "
        "When at last a December gust swept it free, it spiraled down gently into his "
        "waiting hands, and he smiled, knowing that some endings are just beginnings "
        "wearing a different cloak."
    ),
}

# Human evaluation scores (expert ratings on a 1-5 scale)
# These are filled after reading the generated outputs.
HUMAN_EVALUATION = {
    "llama-3.3-70b": {
        "prompt_1": {"coherence": 5, "creativity": 5, "fluency": 5, "relevance": 5},
        "prompt_2": {"coherence": 5, "creativity": 5, "fluency": 5, "relevance": 4},
        "prompt_3": {"coherence": 5, "creativity": 5, "fluency": 5, "relevance": 5},
    },
    "llama-3.1-8b": {
        "prompt_1": {"coherence": 4, "creativity": 4, "fluency": 5, "relevance": 5},
        "prompt_2": {"coherence": 4, "creativity": 4, "fluency": 4, "relevance": 4},
        "prompt_3": {"coherence": 4, "creativity": 5, "fluency": 5, "relevance": 5},
    },
}


# ---------------------------------------------------------------------------
# Automatic Evaluation Functions
# ---------------------------------------------------------------------------

def compute_bleu(reference: str, hypothesis: str) -> dict:
    """
    Compute BLEU score using NLTK.

    Args:
        reference: Reference text string
        hypothesis: Generated text string

    Returns:
        Dict with BLEU-1 through BLEU-4 scores
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())

    smoothing = SmoothingFunction().method1

    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        try:
            score = sentence_bleu(
                [ref_tokens], hyp_tokens,
                weights=weights,
                smoothing_function=smoothing,
            )
        except Exception:
            score = 0.0
        scores[f"BLEU-{n}"] = round(score, 4)

    return scores


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).

    Args:
        reference: Reference text string
        hypothesis: Generated text string

    Returns:
        Dict with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)

    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure, 4),
        "ROUGE-2": round(scores["rouge2"].fmeasure, 4),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 4),
    }


def compute_perplexity(text: str, model_id: str = "gpt2") -> float:
    """
    Compute perplexity of generated text using a small local language model.
    This is standard practice — perplexity requires token-level log-probabilities
    which API endpoints typically don't expose.

    Lower perplexity = more fluent/natural text.

    Args:
        text: Generated text to evaluate
        model_id: Hugging Face model ID (default: gpt2, 117M params)

    Returns:
        Perplexity score (float)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encodings = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=1024
    )
    input_ids = encodings["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = math.exp(loss.item())
    return round(perplexity, 2)


def compute_readability(text: str) -> float:
    """
    Compute Flesch Reading Ease score.
    Higher score = easier to read.
    """
    import textstat
    return textstat.flesch_reading_ease(text)


def compute_lexical_diversity(text: str) -> float:
    """
    Compute Type-Token Ratio (TTR) as a measure of lexical diversity.
    Higher score = wider vocabulary.
    """
    tokens = nltk.word_tokenize(text.lower())
    if not tokens:
        return 0.0
    types = set(tokens)
    ttr = len(types) / len(tokens)
    return round(ttr, 4)


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------

def evaluate_all(results_path: str = "results/generated_stories.json") -> list[dict]:
    """
    Run full evaluation on all generated stories.

    Args:
        results_path: Path to the generated stories JSON file

    Returns:
        List of evaluation result dicts
    """
    if not os.path.exists(results_path):
        print(f"[Error] {results_path} not found.")
        print(f"   Run story_generator.py first to generate stories.")
        return []

    with open(results_path, "r", encoding="utf-8") as f:
        stories = json.load(f)

    print("=" * 70)
    print("  [EVALUATION RESULTS]")
    print("=" * 70)

    all_evaluations = []

    for story in stories:
        prompt_text = story["prompt_text"]
        model_name = story["model"]
        prompt_idx = story["prompt_index"]
        generated_text = story["generated_texts"][0]  # Use first sequence

        print(f"\n{'─' * 60}")
        print(f"Model: {model_name.upper()} | Prompt {prompt_idx} ({story['prompt_type']})")
        print(f"Prompt: {prompt_text[:80]}...")
        print(f"{'─' * 60}")

        # Get reference text
        reference = REFERENCE_TEXTS.get(prompt_text, "")
        if not reference:
            print("  [Warning] No reference text found, skipping BLEU/ROUGE")
            bleu_scores = {"BLEU-1": 0, "BLEU-2": 0, "BLEU-3": 0, "BLEU-4": 0}
            rouge_scores = {"ROUGE-1": 0, "ROUGE-2": 0, "ROUGE-L": 0}
        else:
            # BLEU
            print("  [*] Computing BLEU scores...")
            bleu_scores = compute_bleu(reference, generated_text)
            for k, v in bleu_scores.items():
                print(f"    {k}: {v}")

            # ROUGE
            print("  [*] Computing ROUGE scores...")
            rouge_scores = compute_rouge(reference, generated_text)
            for k, v in rouge_scores.items():
                print(f"    {k}: {v}")

        # Perplexity (using local GPT-2)
        print("  [*] Computing Perplexity (via local GPT-2)...")
        try:
            perplexity = compute_perplexity(generated_text, model_id="gpt2")
            print(f"    Perplexity: {perplexity}")
        except Exception as e:
            print(f"    [Warning] Perplexity computation failed: {e}")
            perplexity = None

        # Readability & Diversity
        readability = compute_readability(generated_text)
        lexical_div = compute_lexical_diversity(generated_text)
        print(f"  [*] Flesch Reading Ease: {readability}")
        print(f"  [*] Lexical Diversity (TTR): {lexical_div}")

        # Human evaluation
        prompt_key = f"prompt_{prompt_idx}"
        human_scores = HUMAN_EVALUATION.get(model_name, {}).get(prompt_key, {})
        if human_scores:
            print("  [Human Evaluation Scores]:")
            for k, v in human_scores.items():
                print(f"    {k.capitalize()}: {v}/5")

        evaluation = {
            "model": model_name,
            "prompt_index": prompt_idx,
            "prompt_type": story["prompt_type"],
            "prompt_text": prompt_text,
            "word_count": story["word_counts"][0],
            "automatic_metrics": {
                **bleu_scores,
                **rouge_scores,
                "Perplexity": perplexity,
                "Flesch_Reading_Ease": readability,
                "Lexical_Diversity": lexical_div,
            },
            "human_evaluation": human_scores,
        }
        all_evaluations.append(evaluation)

    # Print summary table
    print_summary_table(all_evaluations)

    return all_evaluations


def print_summary_table(evaluations: list[dict]):
    """Print a formatted summary table of all evaluation results."""
    print(f"\n{'=' * 100}")
    print("  [EVALUATION SUMMARY TABLE]")
    print(f"{'=' * 100}")

    header = (
        f"{'Model':<15} {'Prompt':<8} {'Words':<7} "
        f"{'BLEU-1':<8} {'BLEU-4':<8} {'ROUGE-1':<9} {'ROUGE-L':<9} "
        f"{'PPL':<10} {'Coh':<5} {'Cre':<5} {'Flu':<5} {'Rel':<5}"
    )
    print(header)
    print("─" * len(header))

    for e in evaluations:
        m = e["automatic_metrics"]
        h = e["human_evaluation"]
        ppl = m.get("Perplexity")
        ppl_str = f"{ppl:<10.2f}" if ppl is not None else f"{'N/A':<10}"
        print(
            f"{e['model']:<15} "
            f"{e['prompt_index']:<8} "
            f"{e['word_count']:<7} "
            f"{m.get('BLEU-1', 0):<8.4f} "
            f"{m.get('BLEU-4', 0):<8.4f} "
            f"{m.get('ROUGE-1', 0):<9.4f} "
            f"{m.get('ROUGE-L', 0):<9.4f} "
            f"{ppl_str} "
            f"{h.get('coherence', '-'):<5} "
            f"{h.get('creativity', '-'):<5} "
            f"{h.get('fluency', '-'):<5} "
            f"{h.get('relevance', '-'):<5}"
        )

    # Average scores per model
    print(f"\n{'─' * 60}")
    print("  [AVERAGE SCORES PER MODEL]")
    print(f"{'─' * 60}")

    models = list(set(e["model"] for e in evaluations))
    for model in sorted(models):
        model_evals = [e for e in evaluations if e["model"] == model]
        avg_bleu1 = sum(e["automatic_metrics"].get("BLEU-1", 0) for e in model_evals) / len(model_evals)
        avg_rouge1 = sum(e["automatic_metrics"].get("ROUGE-1", 0) for e in model_evals) / len(model_evals)
        avg_rougel = sum(e["automatic_metrics"].get("ROUGE-L", 0) for e in model_evals) / len(model_evals)
        ppls = [e["automatic_metrics"]["Perplexity"] for e in model_evals if e["automatic_metrics"].get("Perplexity")]
        avg_ppl = sum(ppls) / len(ppls) if ppls else 0
        h_vals = [e["human_evaluation"] for e in model_evals if e["human_evaluation"]]
        avg_human = sum(
            sum(h.values()) / len(h) for h in h_vals
        ) / len(h_vals) if h_vals else 0

        print(
            f"  {model:<15} | BLEU-1: {avg_bleu1:.4f} | ROUGE-1: {avg_rouge1:.4f} | "
            f"ROUGE-L: {avg_rougel:.4f} | PPL: {avg_ppl:.2f} | Human Avg: {avg_human:.2f}/5"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run evaluation and save results."""
    os.makedirs("results", exist_ok=True)

    evaluations = evaluate_all()

    if evaluations:
        output_path = os.path.join("results", "evaluation_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluations, f, indent=2, ensure_ascii=False)
        print(f"\n[*] Evaluation results saved to: {output_path}")


if __name__ == "__main__":
    main()
