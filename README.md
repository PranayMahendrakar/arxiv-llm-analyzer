# 🔬 arXiv LLM Analyzer

> **Autonomous Research Paper Analyzer** — Downloads the latest papers from arXiv, runs local LLM analysis using DistilGPT2 (HuggingFace Transformers), and publishes an AI-generated literature review to GitHub Pages. **Zero API keys. 100% open-source.**

[![Analyze Papers](https://github.com/PranayMahendrakar/arxiv-llm-analyzer/actions/workflows/analyze.yml/badge.svg)](https://github.com/PranayMahendrakar/arxiv-llm-analyzer/actions/workflows/analyze.yml)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://pranaymahendrakar.github.io/arxiv-llm-analyzer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🚀 Live Dashboard

**→ [https://pranaymahendrakar.github.io/arxiv-llm-analyzer/](https://pranaymahendrakar.github.io/arxiv-llm-analyzer/)**

The dashboard auto-updates daily and shows:
- 📄 **Paper summary** — title, abstract, authors, date
- 🔬 **Key methods** — AI-extracted methodologies
- 📊 **Datasets used** — benchmarks and evaluation sets
- 🚀 **Future work suggestions** — LLM-generated research directions

---

## 🏗️ Pipeline Architecture

```
┌─────────────────┐
│  GitHub Action  │  ← Runs daily at 06:00 UTC  (or manually triggered)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  arXiv API      │  ← Fetches latest N papers matching your query
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DistilGPT2     │  ← Local inference via HuggingFace Transformers (CPU)
│  (82 MB model)  │    Extracts: methods · datasets · future work
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HTML Report    │  ← Saved to docs/index.html
│  + papers.json  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GitHub Pages   │  ← Auto-deployed, publicly accessible
└─────────────────┘
```

---

## 📂 Project Structure

```
arxiv-llm-analyzer/
├── .github/
│   └── workflows/
│       └── analyze.yml        # GitHub Actions pipeline
├── docs/
│   ├── index.html             # GitHub Pages dashboard (auto-generated)
│   └── papers.json            # Raw analysis data (auto-generated)
├── analyze_papers.py          # Main analysis script
├── requirements.txt           # Python dependencies
└── README.md
```

---

## ⚙️ How It Works

### 1. GitHub Actions Trigger
The workflow runs automatically every day at **06:00 UTC** via a `cron` schedule.
You can also trigger it manually from the [Actions tab](../../actions) with a custom query.

### 2. arXiv Paper Download
Uses the [arXiv API](https://arxiv.org/help/api/index) (no key required) to fetch the
most recently submitted papers matching your search query (default: *"large language models"*).

### 3. Local LLM Analysis — DistilGPT2
Uses **DistilGPT2** (~82 MB) loaded via HuggingFace `transformers` + `pipeline`.
Runs entirely on CPU — no GPU or API key needed.

For each paper it generates:
| Field | Prompt Strategy |
|-------|----------------|
| 🔬 Key Methods | `Paper: {title}\nAbstract: {summary}\nKey methods used:` |
| 📊 Datasets | `Paper: {title}\nAbstract: {summary}\nDatasets mentioned:` |
| 🚀 Future Work | `Paper: {title}\nAbstract: {summary}\nFuture research:` |

### 4. Report Generation
Builds a responsive, dark-themed HTML dashboard and a structured `papers.json` file,
then commits them back to the repository.

### 5. GitHub Pages Deploy
The `docs/` folder is deployed automatically via `actions/deploy-pages`.

---

## 🔧 Customization

### Change the arXiv query
Edit the workflow dispatch input or set the env variable:
```yaml
# In .github/workflows/analyze.yml
env:
  ARXIV_QUERY: "transformer architecture"   # ← change this
  MAX_RESULTS: "8"
```

Or trigger manually from Actions → "Run workflow" with your query.

### Swap the model
Edit `analyze_papers.py`:
```python
MODEL_NAME = "distilgpt2"       # default (~82 MB, CPU-safe)
# MODEL_NAME = "microsoft/phi-2"  # better quality but ~2.7 GB
# MODEL_NAME = "facebook/opt-125m"  # fast alternative
```

---

## 🛠️ Local Development

```bash
git clone https://github.com/PranayMahendrakar/arxiv-llm-analyzer.git
cd arxiv-llm-analyzer

pip install -r requirements.txt

ARXIV_QUERY="neural networks" MAX_RESULTS=3 python analyze_papers.py
# → generates docs/index.html and docs/papers.json
```

Open `docs/index.html` in your browser to preview the dashboard.

---

## 🤖 Model Options (No API Key Required)

| Model | Size | Framework | Notes |
|-------|------|-----------|-------|
| **DistilGPT2** ✅ | ~82 MB | HuggingFace | Default — GitHub Actions safe |
| facebook/opt-125m | ~125 MB | HuggingFace | Slightly better quality |
| microsoft/phi-2 | ~2.7 GB | HuggingFace | Excellent quality, needs self-hosted runner |
| mistral-7b (GGUF) | ~4 GB | llama.cpp | Best quality, self-hosted only |

---

## 📄 License

MIT © [Pranay M Mahendrakar](https://github.com/PranayMahendrakar)
