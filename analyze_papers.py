#!/usr/bin/env python3
"""
Autonomous Research Paper Analyzer
Downloads papers from arXiv, analyzes them using DistilGPT2 (HuggingFace),
and generates a structured HTML report for GitHub Pages.
No API keys required.
"""

import os
import json
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Configuration
ARXIV_BASE_URL = "https://export.arxiv.org/api/query"
QUERY          = os.environ.get("ARXIV_QUERY", "large language models")
MAX_RESULTS    = int(os.environ.get("MAX_RESULTS", "5"))
MODEL_NAME     = "distilgpt2"  # ~82 MB, fits GitHub Actions RAM
OUTPUT_DIR     = Path("docs")
DATA_FILE      = OUTPUT_DIR / "papers.json"
REPORT_FILE    = OUTPUT_DIR / "index.html"


# 1. Fetch papers from arXiv
def fetch_arxiv_papers(query, max_results):
    print(f"[arXiv] Fetching top {max_results} papers for: {query!r}")
    params = urllib.parse.urlencode({
        "search_query": "ti:" + query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = ARXIV_BASE_URL + "?" + params
    print(f"[arXiv] URL: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "arxiv-llm-analyzer/1.0 (github.com/PranayMahendrakar)"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_data = resp.read().decode("utf-8")
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall("atom:entry", ns):
        title_el   = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el      = entry.find("atom:id", ns)
        pub_el     = entry.find("atom:published", ns)
        if title_el is None:
            continue
        title   = title_el.text.strip().replace("\n", " ")
        summary = (summary_el.text or "").strip().replace("\n", " ")
        authors = [a.find("atom:name", ns).text
                   for a in entry.findall("atom:author", ns)
                   if a.find("atom:name", ns) is not None]
        link       = (id_el.text or "").strip()
        pub        = (pub_el.text or "")[:10]
        categories = [c.get("term", "") for c in entry.findall("atom:category", ns)]
        papers.append({
            "title": title,
            "summary": summary[:800],
            "authors": authors[:5],
            "link": link,
            "published": pub,
            "categories": categories,
        })
    print(f"[arXiv] Retrieved {len(papers)} papers.")
    return papers


# 2. Load local LLM
def load_model():
    print(f"[LLM] Loading {MODEL_NAME} ...")
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


# 3. Analyse a single paper
def analyse_paper(gen, paper):
    title = paper["title"][:150]
    prompts = {
        "key_methods":  f"Paper: {title}.\nMethods used:",
        "datasets":     f"Paper: {title}.\nDatasets mentioned:",
        "future_work":  f"Paper: {title}.\nFuture research:",
    }
    results = {}
    for field, prompt in prompts.items():
        try:
            out = gen(
                prompt,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1,
            )
            generated = out[0]["generated_text"][len(prompt):].strip()
            sentence  = generated.split(".")[0].strip()
            results[field] = sentence[:120] + "..." if len(sentence) > 120 else sentence + "."
        except Exception as e:
            print(f"[LLM] Warning {field}: {e}")
            results[field] = "Analysis unavailable."
    return results


# 4. Build HTML dashboard
CARD = """
<div class="card">
  <div class="card-header">
    <h2><a href="{link}" target="_blank" rel="noopener">{title}</a></h2>
    <span class="meta">Published: {published} | Authors: {authors} | Tags: {categories}</span>
  </div>
  <div class="card-body">
    <section class="full"><h3>Abstract</h3><p>{summary}</p></section>
    <section><h3>Key Methods <span class="ai-badge">AI</span></h3><p>{key_methods}</p></section>
    <section><h3>Datasets Used <span class="ai-badge">AI</span></h3><p>{datasets}</p></section>
    <section class="full"><h3>Future Work <span class="ai-badge">AI</span></h3><p>{future_work}</p></section>
  </div>
</div>
"""

HTML_TMPL = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>arXiv LLM Analyzer</title>
  <style>
    :root{{--bg:#0d1117;--sur:#161b22;--bdr:#30363d;--acc:#58a6ff;--txt:#c9d1d9;--mut:#8b949e}}
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.6}}
    header{{background:var(--sur);border-bottom:1px solid var(--bdr);padding:1.5rem 2rem;display:flex;align-items:center;gap:1rem}}
    header h1{{font-size:1.4rem;color:var(--acc)}}
    header p{{color:var(--mut);font-size:.85rem}}
    .pipe{{display:flex;gap:.4rem;flex-wrap:wrap;padding:.75rem 2rem;background:var(--sur);border-bottom:1px solid var(--bdr);align-items:center;font-size:.8rem;color:var(--mut)}}
    .step{{background:var(--bg);border:1px solid var(--bdr);border-radius:20px;padding:.25rem .7rem}}
    .arr{{color:var(--acc);font-weight:bold}}
    .stats{{display:flex;gap:1rem;padding:.75rem 2rem;flex-wrap:wrap}}
    .stat{{background:var(--sur);border:1px solid var(--bdr);border-radius:8px;padding:.7rem 1rem;flex:1;min-width:130px}}
    .stat .n{{font-size:1.6rem;font-weight:700;color:var(--acc)}}
    .stat .l{{font-size:.75rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}}
    main{{max-width:960px;margin:0 auto;padding:1rem 2rem;display:flex;flex-direction:column;gap:1.2rem}}
    .card{{background:var(--sur);border:1px solid var(--bdr);border-radius:10px;overflow:hidden}}
    .card-header{{padding:1rem 1.25rem;border-bottom:1px solid var(--bdr)}}
    .card-header h2{{font-size:1rem}}
    .card-header a{{color:var(--acc);text-decoration:none}}
    .card-header a:hover{{text-decoration:underline}}
    .meta{{font-size:.75rem;color:var(--mut);display:block;margin-top:.3rem}}
    .card-body{{padding:1rem 1.25rem;display:grid;grid-template-columns:1fr 1fr;gap:.9rem}}
    .card-body .full{{grid-column:1/-1}}
    section h3{{font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);margin-bottom:.3rem}}
    section p{{font-size:.88rem}}
    .ai-badge{{font-size:.65rem;background:#388bfd22;border:1px solid #388bfd;border-radius:20px;padding:.05rem .4rem;color:var(--acc)}}
    footer{{text-align:center;padding:1.5rem;color:var(--mut);font-size:.75rem;border-top:1px solid var(--bdr);margin-top:1.5rem}}
    footer a{{color:var(--acc);text-decoration:none}}
    @media(max-width:600px){{.card-body{{grid-template-columns:1fr}}}}
  </style>
</head>
<body>
<header>
  <div style="font-size:2rem">&#128300;</div>
  <div>
    <h1>arXiv LLM Analyzer</h1>
    <p>Autonomous AI literature review | DistilGPT2 + HuggingFace + GitHub Actions | No API keys</p>
  </div>
</header>
<div class="pipe">
  <span class="step">GitHub Action</span><span class="arr">-&gt;</span>
  <span class="step">arXiv API</span><span class="arr">-&gt;</span>
  <span class="step">DistilGPT2</span><span class="arr">-&gt;</span>
  <span class="step">HTML Report</span><span class="arr">-&gt;</span>
  <span class="step">GitHub Pages</span>
</div>
<div class="stats">
  <div class="stat"><div class="n">{paper_count}</div><div class="l">Papers</div></div>
  <div class="stat"><div class="n">{model}</div><div class="l">LLM Model</div></div>
  <div class="stat"><div class="n">{query}</div><div class="l">Query</div></div>
  <div class="stat"><div class="n">{updated}</div><div class="l">Updated UTC</div></div>
</div>
<main>{cards}</main>
<footer>
  <a href="https://github.com/PranayMahendrakar/arxiv-llm-analyzer">PranayMahendrakar/arxiv-llm-analyzer</a>
  | Powered by DistilGPT2 (HuggingFace Transformers) | No API keys | {updated_full}
</footer>
</body></html>
"""


def build_html(papers, analyses):
    cards = ""
    for p, a in zip(papers, analyses):
        cards += CARD.format(
            title       = p["title"],
            link        = p["link"],
            published   = p["published"],
            authors     = ", ".join(p["authors"]) or "Unknown",
            categories  = ", ".join(p["categories"][:3]),
            summary     = p["summary"] or "No abstract.",
            key_methods = a.get("key_methods", "N/A"),
            datasets    = a.get("datasets", "N/A"),
            future_work = a.get("future_work", "N/A"),
        )
    now = datetime.datetime.utcnow()
    return HTML_TMPL.format(
        paper_count  = len(papers),
        model        = MODEL_NAME,
        query        = QUERY[:16],
        updated      = now.strftime("%Y-%m-%d"),
        updated_full = now.strftime("%Y-%m-%d %H:%M UTC"),
        cards        = cards,
    )


# 5. Main
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    papers   = fetch_arxiv_papers(QUERY, MAX_RESULTS)
    gen      = load_model()
    analyses = []
    for i, paper in enumerate(papers, 1):
        print(f"[LLM] Paper {i}/{len(papers)}: {paper['title'][:60]}...")
        analyses.append(analyse_paper(gen, paper))
    data = [{"paper": p, "analysis": a} for p, a in zip(papers, analyses)]
    DATA_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[IO] JSON -> {DATA_FILE}")
    html = build_html(papers, analyses)
    REPORT_FILE.write_text(html, encoding="utf-8")
    print(f"[IO] HTML -> {REPORT_FILE}")
    print("[OK] Pipeline complete!")


if __name__ == "__main__":
    main()
