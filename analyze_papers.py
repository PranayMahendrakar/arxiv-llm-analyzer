#!/usr/bin/env python3
"""
Autonomous Research Paper Analyzer
Downloads papers from arXiv, analyzes them using a local LLM (DistilGPT2 via HuggingFace),
and generates a structured HTML report for GitHub Pages.
"""

import os
import json
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ─── Configuration ────────────────────────────────────────────────────────────
ARXIV_SEARCH_URL = "https://export.arxiv.org/api/query"
QUERY            = os.environ.get("ARXIV_QUERY", "large language models")
MAX_RESULTS      = int(os.environ.get("MAX_RESULTS", "5"))
MODEL_NAME       = "distilgpt2"          # ~82 MB — fits GitHub Actions RAM
OUTPUT_DIR       = Path("docs")
DATA_FILE        = OUTPUT_DIR / "papers.json"
REPORT_FILE      = OUTPUT_DIR / "index.html"


# ─── 1. Fetch papers from arXiv ───────────────────────────────────────────────
def fetch_arxiv_papers(query: str, max_results: int) -> list:
    print(f"[arXiv] Fetching top {max_results} papers for: '{query}'")
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        title   = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        link    = entry.find("atom:id", ns).text.strip()
        pub     = entry.find("atom:published", ns).text.strip()[:10]
        categories = [c.get("term") for c in entry.findall("atom:category", ns)]
        papers.append({
            "title":      title,
            "summary":    summary[:800],
            "authors":    authors[:5],
            "link":       link,
            "published":  pub,
            "categories": categories,
        })
    print(f"[arXiv] Retrieved {len(papers)} papers.")
    return papers


# ─── 2. Load local LLM ────────────────────────────────────────────────────────
def load_model():
    print(f"[LLM] Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        pad_token_id=tokenizer.eos_token_id,
    )
    print("[LLM] Model ready.")
    return gen


# ─── 3. Analyse a single paper ────────────────────────────────────────────────
def analyse_paper(gen, paper: dict) -> dict:
    title   = paper["title"]
    summary = paper["summary"][:400]

    prompts = {
        "key_methods": (
            f"Paper: {title}\nAbstract: {summary}\nKey methods used:"
        ),
        "datasets": (
            f"Paper: {title}\nAbstract: {summary}\nDatasets or benchmarks mentioned:"
        ),
        "future_work": (
            f"Paper: {title}\nAbstract: {summary}\nFuture research directions:"
        ),
    }

    results = {}
    for field, prompt in prompts.items():
        out = gen(
            prompt,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1,
        )
        generated = out[0]["generated_text"][len(prompt):].strip()
        sentence  = generated.split(".")[0].strip()
        results[field] = (sentence[:120] + "…") if len(sentence) > 120 else sentence + "."
    return results


# ─── 4. Build HTML dashboard ──────────────────────────────────────────────────
CARD_TPL = """
  <div class="card">
    <div class="card-header">
      <h2><a href="{link}" target="_blank" rel="noopener">{title}</a></h2>
      <span class="meta">📅 {published} &nbsp;|&nbsp; 👥 {authors} &nbsp;|&nbsp; 🏷️ {categories}</span>
    </div>
    <div class="card-body">
      <section>
        <h3>📄 Abstract</h3>
        <p>{summary}</p>
      </section>
      <section>
        <h3>🔬 Key Methods <span class="badge ai">AI-generated</span></h3>
        <p>{key_methods}</p>
      </section>
      <section>
        <h3>📊 Datasets Used <span class="badge ai">AI-generated</span></h3>
        <p>{datasets}</p>
      </section>
      <section>
        <h3>🚀 Future Work <span class="badge ai">AI-generated</span></h3>
        <p>{future_work}</p>
      </section>
    </div>
  </div>
"""

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>arXiv LLM Analyzer — AI Literature Review</title>
  <style>
    :root {
      --bg:#0d1117;--surface:#161b22;--border:#30363d;
      --accent:#58a6ff;--text:#c9d1d9;--muted:#8b949e;
    }
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.6}
    header{background:var(--surface);border-bottom:1px solid var(--border);padding:1.5rem 2rem;display:flex;align-items:center;gap:1rem}
    header h1{font-size:1.5rem;color:var(--accent)}
    header p{color:var(--muted);font-size:.9rem}
    .pipeline{display:flex;gap:.5rem;flex-wrap:wrap;padding:1rem 2rem;background:var(--surface);border-bottom:1px solid var(--border);align-items:center;font-size:.85rem;color:var(--muted)}
    .pipeline .step{background:var(--bg);border:1px solid var(--border);border-radius:20px;padding:.3rem .8rem}
    .pipeline .arrow{color:var(--accent)}
    .stats{display:flex;gap:1.5rem;padding:1rem 2rem;flex-wrap:wrap}
    .stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.8rem 1.2rem;flex:1;min-width:140px}
    .stat .num{font-size:1.8rem;font-weight:700;color:var(--accent)}
    .stat .label{font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
    main{max-width:960px;margin:0 auto;padding:1.5rem 2rem;display:flex;flex-direction:column;gap:1.5rem}
    .card{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden}
    .card-header{padding:1.2rem 1.5rem;border-bottom:1px solid var(--border)}
    .card-header h2{font-size:1.05rem}
    .card-header a{color:var(--accent);text-decoration:none}
    .card-header a:hover{text-decoration:underline}
    .meta{font-size:.8rem;color:var(--muted);display:block;margin-top:.4rem}
    .card-body{padding:1.2rem 1.5rem;display:grid;grid-template-columns:1fr 1fr;gap:1rem}
    .card-body section:first-child{grid-column:1/-1}
    section h3{font-size:.85rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:.4rem;display:flex;align-items:center;gap:.5rem}
    section p{font-size:.92rem}
    .badge{font-size:.7rem;background:#388bfd22;border:1px solid #388bfd;border-radius:20px;padding:.1rem .5rem;color:var(--accent);text-transform:none;letter-spacing:0}
    footer{text-align:center;padding:2rem;color:var(--muted);font-size:.8rem;border-top:1px solid var(--border);margin-top:2rem}
    @media(max-width:600px){.card-body{grid-template-columns:1fr}}
  </style>
</head>
<body>
<header>
  <div style="font-size:2rem">🔬</div>
  <div>
    <h1>arXiv LLM Analyzer</h1>
    <p>Autonomous AI-generated literature review — powered by DistilGPT2 &amp; GitHub Actions</p>
  </div>
</header>
<div class="pipeline">
  <span class="step">⚙️ GitHub Action</span><span class="arrow">→</span>
  <span class="step">📥 arXiv Download</span><span class="arrow">→</span>
  <span class="step">🧠 LLM Analysis</span><span class="arrow">→</span>
  <span class="step">📝 Summary Gen</span><span class="arrow">→</span>
  <span class="step">🌐 GitHub Pages</span>
</div>
"""

HTML_FOOT = """<footer>
  Generated by <a href="https://github.com/PranayMahendrakar/arxiv-llm-analyzer" style="color:#58a6ff">arxiv-llm-analyzer</a>
  using <strong>DistilGPT2</strong> (HuggingFace Transformers) — no API keys required.<br/>
  Updated: {updated_full}
</footer>
</body></html>
"""


def build_html(papers, analyses):
    cards = ""
    for p, a in zip(papers, analyses):
        cards += CARD_TPL.format(
            title       = p["title"],
            link        = p["link"],
            published   = p["published"],
            authors     = ", ".join(p["authors"]) or "Unknown",
            categories  = ", ".join(p["categories"][:3]),
            summary     = p["summary"],
            key_methods = a.get("key_methods", "N/A"),
            datasets    = a.get("datasets", "N/A"),
            future_work = a.get("future_work", "N/A"),
        )
    now = datetime.datetime.utcnow()
    stats = f"""
<div class="stats">
  <div class="stat"><div class="num">{len(papers)}</div><div class="label">Papers Analyzed</div></div>
  <div class="stat"><div class="num">{MODEL_NAME}</div><div class="label">Local LLM Model</div></div>
  <div class="stat"><div class="num">{QUERY[:20]}</div><div class="label">arXiv Query</div></div>
  <div class="stat"><div class="num">{now.strftime('%Y-%m-%d')}</div><div class="label">Last Updated</div></div>
</div>
<main>
{cards}
</main>
"""
    foot = HTML_FOOT.format(updated_full=now.strftime("%Y-%m-%d %H:%M UTC"))
    return HTML_HEAD + stats + foot


# ─── 5. Main ──────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    papers   = fetch_arxiv_papers(QUERY, MAX_RESULTS)
    gen      = load_model()
    analyses = []
    for i, paper in enumerate(papers, 1):
        print(f"[LLM] Paper {i}/{len(papers)}: {paper['title'][:60]}…")
        analyses.append(analyse_paper(gen, paper))

    data = [{"paper": p, "analysis": a} for p, a in zip(papers, analyses)]
    DATA_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[IO] JSON → {DATA_FILE}")

    html = build_html(papers, analyses)
    REPORT_FILE.write_text(html, encoding="utf-8")
    print(f"[IO] HTML → {REPORT_FILE}")
    print("[✓] Done!")


if __name__ == "__main__":
    main()
