"""
Lab 1 - Part D: Streamlit Application (Groq API)
==================================================
Interactive Story/Poem Generation Web Application powered by Groq LPU.

Features:
  - Model selection (LLaMA 3.3, Gemma 2, Mixtral)
  - Parameter tuning (temperature, top-p, max tokens, num sequences)
  - Real-time ultra-fast story/poem generation
  - Automatic evaluation metrics (BLEU, ROUGE, Perplexity)
  - Human evaluation scoring
  - Download generated text

Run locally: streamlit run streamlit_app.py
"""

import json
import math
import os
import time

import nltk
import streamlit as st
import torch
import textstat
import nltk
from rouge_score import rouge_scorer

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="StoryForge AI — Story & Poem Generator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Premium Dark Theme
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    /* ── Import Google Fonts ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

    /* ── Root variables ──────────────────────────────── */
    :root {
        --bg-primary: #0a0e17;
        --bg-card: rgba(15, 23, 42, 0.8);
        --bg-card-hover: rgba(20, 30, 55, 0.9);
        --border-glow: rgba(99, 102, 241, 0.3);
        --accent-indigo: #818cf8;
        --accent-violet: #a78bfa;
        --accent-emerald: #34d399;
        --accent-amber: #fbbf24;
        --accent-rose: #fb7185;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }

    /* ── Global ──────────────────────────────────────── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ──────────────────────────────────────── */
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #a78bfa, #f0abfc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        animation: shimmer 3s ease-in-out infinite alternate;
    }

    @keyframes shimmer {
        0% { opacity: 0.85; }
        100% { opacity: 1; }
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* ── Story output card ───────────────────────────── */
    .story-card {
        background: var(--bg-card);
        border: 1px solid var(--border-glow);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
        line-height: 1.85;
        font-size: 1.05rem;
        color: var(--text-primary);
        font-family: 'Playfair Display', serif;
        white-space: pre-wrap;
    }

    .story-card:hover {
        border-color: var(--accent-indigo);
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.15);
        transform: translateY(-1px);
    }

    /* ── Metric cards ────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }

    .metric-card {
        background: var(--bg-card);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        flex: 1;
        min-width: 120px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent-indigo);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
    }

    .metric-label {
        font-size: 0.72rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ── Section headers ─────────────────────────────── */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--accent-violet);
        margin: 1.5rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Sidebar styling ─────────────────────────────── */
    .sidebar-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--accent-indigo);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-glow);
    }

    /* ── Info badge ───────────────────────────────────── */
    .info-badge {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin: 0.5rem 0;
    }

    /* ── Speed badge ─────────────────────────────────── */
    .speed-badge {
        background: rgba(52, 211, 153, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        font-size: 0.8rem;
        color: var(--accent-emerald);
        text-align: center;
        margin: 0.5rem 0;
    }

    /* ── Divider ─────────────────────────────────────── */
    .glow-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-indigo), transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* ── Status pill ─────────────────────────────────── */
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .status-success {
        background: rgba(52, 211, 153, 0.15);
        color: var(--accent-emerald);
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

MODELS = {
    "LLaMA 3.3 70B Versatile (Meta)": {
        "id": "llama-3.3-70b-versatile",
        "params": "70B",
        "description": "Best for creative, long-form writing",
        "source": "groq",
    },
    "LLaMA 3.1 8B Instant (Meta)": {
        "id": "llama-3.1-8b-instant",
        "params": "8B",
        "description": "Fast, high-quality, lightweight",
        "source": "groq",
    },
    "GPT-2 Small (OpenAI)": {
        "id": "gpt2",
        "params": "117M",
        "description": "Classic local decoder model",
        "source": "local",
    },
}

EXAMPLE_PROMPTS = [
    {
        "prompt": "In a world where dreams could be harvested, a young girl discovered she had the rarest dream of all",
        "reference": (
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
        )
    },
    {
        "prompt": "moonlight, forgotten castle, ancient magic, whispers",
        "reference": (
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
        )
    },
    {
        "prompt": "The last autumn leaf",
        "reference": (
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
        )
    },
]

SYSTEM_PROMPT = """You are a masterful creative writer and poet. When given a prompt, \
you produce vivid, imaginative, and emotionally resonant stories or poems. \
Your writing should be between 200 and 500 words, with rich imagery, compelling \
narrative flow, and a satisfying conclusion. Use literary devices such as metaphor, \
simile, personification, and sensory details to bring the story to life. \
Do not include any preamble, commentary, or notes — only output the story or poem itself."""


# ---------------------------------------------------------------------------
# Generation Logic
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_local_gpt2():
    """Load GPT-2 for local generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model

def generate_story_local_gpt2(prompt: str, temperature: float, top_p: float, max_tokens: int, num_sequences: int) -> tuple[list[str], float]:
    """Generate story via local GPT-2."""
    from transformers import set_seed
    tokenizer, model = load_local_gpt2()
    
    if "," in prompt and len(prompt.split()) < 10:
        prepared = f"Once upon a time, in a land painted with {prompt}, there existed a tale unlike any other. "
    else:
        prepared = prompt

    inputs = tokenizer(prepared, return_tensors="pt", truncation=True)
    
    start = time.time()
    set_seed(42)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_sequences,
            do_sample=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.get("attention_mask")
        )
    
    texts = []
    for out in outputs:
        texts.append(tokenizer.decode(out, skip_special_tokens=True).strip())
    
    elapsed = time.time() - start
    return texts, elapsed

def generate_story_groq(
    prompt: str,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    num_sequences: int,
    api_key: str,
) -> tuple[list[str], float]:
    """Generate story via Groq API. Returns (texts, elapsed_seconds)."""
    from groq import Groq

    client = Groq(api_key=api_key)

    # Build user prompt
    if "," in prompt and len(prompt.split()) < 10:
        user_prompt = (
            f"Write a creative story or poem inspired by these elements: "
            f"{prompt}. Weave them together into a compelling narrative "
            f"with vivid imagery and emotion."
        )
    else:
        user_prompt = (
            f"Continue and expand this into a full creative story or poem "
            f"(200-500 words):\n\n{prompt}"
        )

    texts = []
    start = time.time()

    for i in range(num_sequences):
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=42 + i,
            stream=False,
        )
        texts.append(response.choices[0].message.content.strip())
        if num_sequences > 1 and i < num_sequences - 1:
            time.sleep(0.3)

    elapsed = time.time() - start
    return texts, elapsed


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

def compute_perplexity(text: str, model_id: str = "gpt2") -> float:
    """Compute perplexity using local GPT-2."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(encodings["input_ids"], labels=encodings["input_ids"])
        loss = outputs.loss

    return round(math.exp(loss.item()), 2)

def compute_readability(text: str) -> float:
    return textstat.flesch_reading_ease(text)

def compute_lexical_diversity(text: str) -> float:
    tokens = nltk.word_tokenize(text.lower())
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)

def compute_metrics(generated: str, reference: str = "") -> dict:
    """Compute BLEU, ROUGE, and Perplexity for a generated text."""
    metrics = {}

    # BLEU & ROUGE (only if reference provided)
    if reference:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        ref_tok = nltk.word_tokenize(reference.lower())
        hyp_tok = nltk.word_tokenize(generated.lower())
        smooth = SmoothingFunction().method1

        for n in range(1, 5):
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            try:
                score = sentence_bleu(
                    [ref_tok], hyp_tok, weights=weights, smoothing_function=smooth
                )
            except Exception:
                score = 0.0
            metrics[f"BLEU-{n}"] = round(score, 4)

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(reference, generated)
        metrics["ROUGE-1"] = round(scores["rouge1"].fmeasure, 4)
        metrics["ROUGE-2"] = round(scores["rouge2"].fmeasure, 4)
        metrics["ROUGE-L"] = round(scores["rougeL"].fmeasure, 4)

    # Add other metrics
    metrics["Perplexity"] = compute_perplexity(generated)
    metrics["Readability"] = compute_readability(generated)
    metrics["Lexical Diversity"] = compute_lexical_diversity(generated)

    return metrics


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

# Header
st.markdown('<h1 class="hero-title">✨ StoryForge AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    'Craft enchanting stories and poems powered by Groq LPU ⚡'
    '</p>',
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sidebar-header">🔑 API Configuration</div>',
        unsafe_allow_html=True,
    )

    # Check if API key is provided by the server/environment
    server_api_key = os.getenv("GROQ_API_KEY", "")
    try:
        if "GROQ_API_KEY" in st.secrets:
            server_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    if server_api_key and server_api_key != "your_groq_api_key_here":
        st.markdown(
            '<span class="status-pill status-success">✓ Server API Key Configured</span>',
            unsafe_allow_html=True,
        )
        api_key = server_api_key
    else:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free key at https://console.groq.com/keys",
        )
        if api_key:
            st.markdown(
                '<span class="status-pill status-success">✓ Custom API Key Set</span>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-header">🤖 Model Selection</div>',
        unsafe_allow_html=True,
    )

    model_choice = st.selectbox(
        "Select Model",
        list(MODELS.keys()),
        help="LLaMA 3.3 70B gives the best creative output. Gemma 2 is the fastest.",
        label_visibility="collapsed",
    )

    model_info = MODELS[model_choice]
    st.markdown(
        f'<div class="info-badge">'
        f'<strong>{model_info["id"]}</strong><br>'
        f'{model_info["params"]} parameters<br>'
        f'{model_info["description"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-header">🎛️ Generation Parameters</div>',
        unsafe_allow_html=True,
    )

    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.05,
        help="Controls randomness. Lower = focused, Higher = creative.",
    )

    top_p = st.slider(
        "🎲 Top-p (Nucleus Sampling)",
        min_value=0.1,
        max_value=1.0,
        value=0.92,
        step=0.01,
        help="Cumulative probability cutoff for sampling.",
    )

    max_tokens = st.slider(
        "📏 Max Tokens",
        min_value=100,
        max_value=2048,
        value=1024,
        step=64,
        help="Maximum number of tokens to generate.",
    )

    num_sequences = st.slider(
        "📝 Number of Sequences",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        help="Generate multiple different versions.",
    )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="speed-badge">⚡ Powered by Groq LPU — Ultra-fast inference</div>',
        unsafe_allow_html=True,
    )

# ── Main area ────────────────────────────────────────────────────────────────

if "input_prompt_val" not in st.session_state:
    st.session_state["input_prompt_val"] = ""
if "reference_text_val" not in st.session_state:
    st.session_state["reference_text_val"] = ""

# Prompt input
st.markdown(
    '<div class="section-header">📝 Enter Your Prompt</div>', unsafe_allow_html=True
)

col_input, col_examples = st.columns([3, 1])

with col_examples:
    st.markdown("**✨ Quick Prompts**")
    for idx, ex_dict in enumerate(EXAMPLE_PROMPTS):
        ex = ex_dict["prompt"]
        short = ex[:35] + "..." if len(ex) > 35 else ex
        if st.button(f"📌 {short}", key=f"ex_{idx}", use_container_width=True):
            st.session_state["input_prompt_val"] = ex
            st.session_state["reference_text_val"] = ex_dict["reference"]
            st.rerun()

with col_input:
    user_prompt = st.text_area(
        "Write a sentence, phrase, or keywords to generate a story/poem",
        value=st.session_state["input_prompt_val"],
        height=110,
        placeholder="e.g., In a world where dreams could be harvested, a young girl discovered she had the rarest dream of all...",
        label_visibility="collapsed",
    )
    # Update session state if user types manually
    st.session_state["input_prompt_val"] = user_prompt

# Optional reference text
with st.expander("📖 Reference text (optional — for BLEU/ROUGE evaluation)", expanded=True):
    reference_text = st.text_area(
        "Paste a reference story/poem for comparison",
        value=st.session_state["reference_text_val"],
        height=200,
        placeholder="Leave empty to compute only Perplexity",
        label_visibility="collapsed",
    )
    st.session_state["reference_text_val"] = reference_text

# Generate button
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

generate_clicked = st.button(
    "✨ Generate Story / Poem",
    use_container_width=True,
    type="primary",
)

# ── Generation Logic ─────────────────────────────────────────────────────────

if generate_clicked:
    if model_info["source"] == "groq" and (not api_key or api_key == "your_groq_api_key_here"):
        st.error("🔑 Please enter your Groq API key in the sidebar to use Groq models.")
    elif not user_prompt.strip():
        st.warning("⚠️ Please enter a prompt before generating.")
    else:
        with st.spinner(f"🪄 Generating with **{model_info['id']}**..."):
            try:
                if model_info["source"] == "groq":
                    generated_texts, elapsed = generate_story_groq(
                        prompt=user_prompt.strip(),
                        model_id=model_info["id"],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        num_sequences=num_sequences,
                        api_key=api_key,
                    )
                else:
                    generated_texts, elapsed = generate_story_local_gpt2(
                        prompt=user_prompt.strip(),
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        num_sequences=num_sequences,
                    )
                
                st.session_state["generated_texts"] = generated_texts
                st.session_state["generation_time"] = elapsed
                st.session_state["current_prompt"] = user_prompt
                st.session_state["current_model"] = model_choice
            except Exception as e:
                st.error(f"❌ Generation failed: {e}")

# ── Display Results ──────────────────────────────────────────────────────────

if "generated_texts" in st.session_state:
    generated_texts = st.session_state["generated_texts"]
    gen_time = st.session_state.get("generation_time", 0)
    current_model = st.session_state.get("current_model", "")

    st.markdown(
        '<div class="section-header">📖 Generated Output</div>',
        unsafe_allow_html=True,
    )

    # Speed badge
    st.markdown(
        f'<div class="speed-badge">'
        f'⚡ Generated in {gen_time:.2f}s via Groq LPU'
        f'</div>',
        unsafe_allow_html=True,
    )

    for i, text in enumerate(generated_texts):
        word_count = len(text.split())
        if num_sequences > 1:
            st.markdown(f"**Sequence {i + 1}** — {word_count} words")

        st.markdown(
            f'<div class="story-card">{text}</div>', unsafe_allow_html=True
        )

        st.download_button(
            label=f"⬇️ Download {'Sequence ' + str(i+1) if num_sequences > 1 else 'Story/Poem'}",
            data=text,
            file_name=f"generated_story_{i+1}.txt",
            mime="text/plain",
            key=f"dl_{i}",
        )

    # ── Evaluation Metrics ───────────────────────────────────────────────────

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">📊 Evaluation Metrics</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("📐 Computing metrics..."):
        ref = reference_text.strip() if reference_text else ""
        metrics = compute_metrics(generated_texts[0], ref)

    # Metric display
    cols = st.columns(5)
    cols[0].metric("Perplexity", f"{metrics.get('Perplexity', 0):.2f}")
    cols[1].metric("Readability", f"{metrics.get('Readability', 0):.1f}")
    cols[2].metric("Lex. Diversity", f"{metrics.get('Lexical Diversity', 0):.3f}")
    cols[3].metric("Word Count", len(generated_texts[0].split()))
    cols[4].metric("Model", MODELS[current_model]["id"] if current_model else "—")

    if ref:
        bleu_cols = st.columns(4)
        for col, n in zip(bleu_cols, range(1, 5)):
            key = f"BLEU-{n}"
            col.metric(key, f"{metrics.get(key, 0):.4f}")

        rouge_cols = st.columns(3)
        for col, key in zip(rouge_cols, ["ROUGE-1", "ROUGE-2", "ROUGE-L"]):
            col.metric(key, f"{metrics.get(key, 0):.4f}")



    if ref:
        with st.expander("📋 Full Metric Details (JSON)"):
            st.json(metrics)

    # ── Human Evaluation ─────────────────────────────────────────────────────

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">👤 Human Evaluation</div>',
        unsafe_allow_html=True,
    )

    hcol1, hcol2, hcol3, hcol4 = st.columns(4)
    with hcol1:
        coherence = st.slider("Coherence", 1, 5, 4, key="h_coh",
                              help="Does the story flow logically?")
    with hcol2:
        creativity = st.slider("Creativity", 1, 5, 4, key="h_cre",
                               help="Is the writing original and imaginative?")
    with hcol3:
        fluency = st.slider("Fluency", 1, 5, 4, key="h_flu",
                            help="Is the language natural and grammatical?")
    with hcol4:
        relevance = st.slider("Relevance", 1, 5, 4, key="h_rel",
                              help="Does it address the prompt?")

    avg_human = (coherence + creativity + fluency + relevance) / 4
    st.markdown(
        f'<div class="info-badge" style="text-align: center; font-size: 1rem;">'
        f'<strong>Average Human Score: {avg_human:.2f} / 5.00</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
<div style="text-align: center; color: #475569; font-size: 0.8rem; padding: 1rem 0;">
    <p style="margin: 0.2rem 0;"><strong>StoryForge AI</strong> — Lab Exercise 1</p>
    <p style="margin: 0.2rem 0;">Principles of Large Language Models | MSc AIML</p>
    <p style="margin: 0.2rem 0;">Built with Hugging Face Transformers, Groq API & Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
