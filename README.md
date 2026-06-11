# StoryForge AI — Story & Poem Generator 📝✨

An interactive AI-based creative writing application built as part of **Lab 1: Principles of Large Language Models**. The system generates short stories and poems based on user prompts (sentences, keywords, or phrases) and rigorously evaluates the output using automatic and human-centric NLP metrics.

The system utilizes the ultra-fast **Groq LPU API** to run massive open-source models (like LLaMA 3.3 70B) in seconds, and also includes local **GPT-2** evaluation and generation fallbacks using Hugging Face Transformers.

---

## 🌟 Features

- **Ultra-Fast Inference**: Generates 300+ words in under 3 seconds using Groq API.
- **Multiple State-of-the-Art Models**:
  - `LLaMA 3.3 70B Versatile` (Best for creative, rich narrative)
  - `LLaMA 3.1 8B Instant` (Fast, lightweight alternative)
  - `GPT-2 Small` (Local fallback generation & Perplexity evaluation)
- **Granular Controls**: Adjust Temperature, Top-p, Max Tokens, and Number of Sequences.
- **Automatic NLP Evaluation**:
  - **BLEU & ROUGE**: Measures n-gram precision and recall-oriented overlap against reference text.
  - **Perplexity**: Evaluates the linguistic fluency and predictability using a local GPT-2 (lower is better).
  - **Flesch Reading Ease**: Grades the readability/complexity of the text.
  - **Lexical Diversity (TTR)**: Measures the richness of vocabulary used.
- **Interactive UI**: A premium dark-themed Streamlit web application.

---

## 🛠️ Project Structure

```text
StoryForge AI/
│
├── streamlit_app.py        # Part D: Main interactive web application
├── story_generator.py      # Part B: Core generation engine script
├── evaluation.py           # Part C: Automatic metrics pipeline
├── comparative_study.md    # Part A: Theoretical model comparison report
│
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (API Keys)
│
└── results/
    ├── generated_stories.json   # Output cache from story_generator.py
    └── evaluation_results.json  # Output cache from evaluation.py
```

---

## 🚀 Getting Started

### 1. Prerequisites
You need **Python 3.8+** installed on your machine.
You will also need a **Groq API Key**. You can get one for free at [console.groq.com/keys](https://console.groq.com/keys).

### 2. Installation
Clone the repository (or download the folder), open your terminal, and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a file named `.env` in the root directory (if it doesn't already exist) and add your Groq API key:

```env
GROQ_API_KEY=gsk_your_api_key_here
```

---

## 💻 Usage

There are two ways to use this project: as a set of backend scripts, or via the interactive web app.

### 1. Web Application (Recommended)
To launch the beautiful, interactive user interface:
```bash
streamlit run streamlit_app.py
```
This will open the app in your browser (usually at `http://localhost:8501`). From there, you can type prompts, adjust generation sliders, and view the evaluation metrics in real time.

### 2. Backend Scripts
To run the raw generation pipeline across multiple models and test prompts:
```bash
python story_generator.py
```
*(This will save the output to `results/generated_stories.json`)*

To run the full evaluation pipeline (BLEU, ROUGE, PPL, Readability) on those generated stories:
```bash
python evaluation.py
```
*(This will print a summary table to the console and save data to `results/evaluation_results.json`)*

---

## 📐 About the Metrics

- **BLEU / ROUGE**: Requires you to paste a "Reference Text" in the app. It compares the AI's output against the reference to see how much of the vocabulary and phrasing overlaps.
- **Perplexity**: Does *not* require a reference text. The app quietly passes the generated text through a local GPT-2 model. If GPT-2 finds the language predictable and fluent, the score is lower (which is better!).
- **Readability / Lexical Diversity**: Uses standard linguistic formulas (`textstat` and `nltk`) to judge sentence length, syllable count, and unique word variation.
