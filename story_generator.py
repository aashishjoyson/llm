"""
Lab 1 - Part B: Story/Poem Generation System (Groq API)
========================================================
AI-Based Story/Poem Generation using Pre-Trained Language Models via Groq.

Groq provides ultra-fast inference on LPU (Language Processing Unit) hardware
for open-source transformer models.

Supported Models (via Groq):
  - LLaMA 3.3 70B Versatile  (Meta, decoder-only, 70B params)
  - Gemma 2 9B IT            (Google, decoder-only, 9B params)
  - Mixtral 8x7B             (Mistral AI, MoE decoder-only, 46.7B params)

Features:
  - Accept sentence, phrase, or keyword input
  - Generate coherent short stories/poems (~200-500 words)
  - Configurable: temperature, top-k, top-p, max_length, num_sequences
  - Blazing fast inference via Groq LPU
"""

import json
import os
import time
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "llama-3.3-70b": {
        "model_id": "llama-3.3-70b-versatile",
        "description": "LLaMA 3.3 70B Versatile (Meta) — Decoder-only, 70B params",
    },
    "llama-3.1-8b": {
        "model_id": "llama-3.1-8b-instant",
        "description": "LLaMA 3.1 8B Instant (Meta) — Decoder-only, 8B params",
    },
}

DEFAULT_PARAMS = {
    "temperature": 0.8,
    "top_p": 0.92,
    "max_tokens": 1024,
    "num_sequences": 1,
}

# Three diverse test prompts (sentence / keywords / phrase)
TEST_PROMPTS = [
    {
        "type": "sentence",
        "text": "In a world where dreams could be harvested, a young girl discovered she had the rarest dream of all",
    },
    {
        "type": "keywords",
        "text": "moonlight, forgotten castle, ancient magic, whispers",
    },
    {
        "type": "phrase",
        "text": "The last autumn leaf",
    },
]

# ---------------------------------------------------------------------------
# System prompt for creative writing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a masterful creative writer and poet. When given a prompt, \
you produce vivid, imaginative, and emotionally resonant stories or poems. \
Your writing should be between 200 and 500 words, with rich imagery, compelling \
narrative flow, and a satisfying conclusion. Use literary devices such as metaphor, \
simile, personification, and sensory details to bring the story to life. \
Do not include any preamble, commentary, or notes — only output the story or poem itself."""


# ---------------------------------------------------------------------------
# StoryGenerator class
# ---------------------------------------------------------------------------

class StoryGenerator:
    """Generates stories/poems using a pre-trained language model via Groq API."""

    def __init__(self, model_name: str = "llama-3.3-70b", api_key: str = None):
        """
        Initialize the Groq client and select a model.

        Args:
            model_name: Key from SUPPORTED_MODELS
            api_key: Groq API key (falls back to GROQ_API_KEY env var)
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Supported: {list(SUPPORTED_MODELS.keys())}"
            )

        config = SUPPORTED_MODELS[model_name]
        self.model_name = model_name
        self.model_id = config["model_id"]
        self.description = config["description"]

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key or key == "your_groq_api_key_here":
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY in your .env file "
                "or pass api_key= to StoryGenerator. "
                "Get a free key at: https://console.groq.com/keys"
            )

        self.client = Groq(api_key=key)
        print(f"[*] Initialized: {self.description}")
        print(f"   Model ID: {self.model_id}")
        print(f"   API: Groq LPU\n")

    def _build_user_prompt(self, prompt: str) -> str:
        """
        Build the user prompt based on input type.
        Adds creative framing for keywords.
        """
        prompt = prompt.strip()

        # Detect if input is keywords (comma-separated, short)
        if "," in prompt and len(prompt.split()) < 10:
            return (
                f"Write a creative story or poem inspired by these elements: "
                f"{prompt}. Weave them together into a compelling narrative "
                f"with vivid imagery and emotion."
            )
        else:
            return (
                f"Continue and expand this into a full creative story or poem "
                f"(200-500 words):\n\n{prompt}"
            )

    def generate(
        self,
        prompt: str,
        temperature: float = DEFAULT_PARAMS["temperature"],
        top_p: float = DEFAULT_PARAMS["top_p"],
        max_tokens: int = DEFAULT_PARAMS["max_tokens"],
        num_sequences: int = DEFAULT_PARAMS["num_sequences"],
        seed: int = 42,
    ) -> list[str]:
        """
        Generate story/poem from a prompt via Groq API.

        Args:
            prompt: Input text (sentence, phrase, or keywords)
            temperature: Controls randomness (0.1=focused, 2.0=wild)
            top_p: Nucleus sampling (cumulative probability threshold)
            max_tokens: Maximum number of tokens to generate
            num_sequences: Number of different sequences to generate
            seed: Random seed for reproducibility

        Returns:
            List of generated text strings
        """
        user_prompt = self._build_user_prompt(prompt)
        generated_texts = []

        for i in range(num_sequences):
            current_seed = seed + i  # Vary seed per sequence

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=current_seed,
                stream=False,
            )

            text = response.choices[0].message.content.strip()
            generated_texts.append(text)

            # Brief pause between sequences to respect rate limits
            if num_sequences > 1 and i < num_sequences - 1:
                time.sleep(0.5)

        return generated_texts

    def batch_generate(
        self, prompts: list[dict], **params
    ) -> list[dict]:
        """
        Generate stories for multiple prompts.

        Args:
            prompts: List of dicts with 'type' and 'text' keys
            **params: Generation parameters passed to generate()

        Returns:
            List of result dicts with prompt info and generated texts
        """
        results = []
        for i, prompt_info in enumerate(prompts, 1):
            prompt_text = prompt_info["text"]
            prompt_type = prompt_info["type"]
            print(f"--- Prompt {i} ({prompt_type}) ---")
            print(f"  Input: {prompt_text}")
            print(f"  Generating with {self.model_id}...")

            start = time.time()
            generated = self.generate(prompt_text, **params)
            elapsed = time.time() - start

            for j, text in enumerate(generated):
                word_count = len(text.split())
                print(f"  Sequence {j + 1}: {word_count} words ({elapsed:.1f}s)")

            results.append(
                {
                    "prompt_index": i,
                    "prompt_type": prompt_type,
                    "prompt_text": prompt_text,
                    "model": self.model_name,
                    "model_id": self.model_id,
                    "model_description": self.description,
                    "parameters": {
                        "temperature": params.get("temperature", DEFAULT_PARAMS["temperature"]),
                        "top_p": params.get("top_p", DEFAULT_PARAMS["top_p"]),
                        "max_tokens": params.get("max_tokens", DEFAULT_PARAMS["max_tokens"]),
                        "num_sequences": params.get("num_sequences", DEFAULT_PARAMS["num_sequences"]),
                    },
                    "generated_texts": generated,
                    "word_counts": [len(t.split()) for t in generated],
                    "generation_time_seconds": round(elapsed, 2),
                }
            )
            print()

        return results


# ---------------------------------------------------------------------------
# Main — run generation with all models on all test prompts
# ---------------------------------------------------------------------------

def main():
    """Run story generation with all test prompts and save results."""
    os.makedirs("results", exist_ok=True)
    all_results = []

    for model_name in SUPPORTED_MODELS:
        print("=" * 70)
        print(f"  Model: {model_name.upper()}")
        print("=" * 70)

        try:
            generator = StoryGenerator(model_name)
            results = generator.batch_generate(
                TEST_PROMPTS,
                temperature=0.8,
                top_p=0.92,
                max_tokens=1024,
                num_sequences=1,
            )
            all_results.extend(results)

            # Print generated stories
            for r in results:
                print(f"\n{'─' * 60}")
                print(f"Prompt ({r['prompt_type']}): {r['prompt_text']}")
                print(f"Model: {r['model_id']} | Time: {r['generation_time_seconds']}s")
                print(f"{'─' * 60}")
                for idx, text in enumerate(r["generated_texts"]):
                    print(f"\n[Generated Text {idx + 1}] ({r['word_counts'][idx]} words):\n")
                    print(text)
                print()

        except Exception as e:
            print(f"  [Error] with {model_name}: {e}\n")
            continue

        # Small delay between models to respect Groq rate limits
        time.sleep(2)

    # Save results
    if all_results:
        output_path = os.path.join("results", "generated_stories.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 70}")
        print(f"  [*] All results saved to: {output_path}")
        print(f"  Total generations: {len(all_results)}")
        print(f"{'=' * 70}")
    else:
        print("\n[Error] No results generated. Check your API key in .env file.")


if __name__ == "__main__":
    main()
