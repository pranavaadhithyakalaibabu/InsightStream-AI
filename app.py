"""
Vercel entrypoint for InsightStream.

Vercel loads this module and expects a top-level ASGI app named `app`.
The original Streamlit UI remains in streamlit_app.py for local use.
Audio and video transcription stay there too: Whisper and ffmpeg are too
large for a Vercel Python function.
"""

import os
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agent import query_transcript, run_agent

app = FastAPI(title="InsightStream")

ALLOWED_MODELS = ("gemini-2.5-flash", "gemini-2.5-pro")
MAX_TRANSCRIPT_CHARS = 400_000


class AnalyzeRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=MAX_TRANSCRIPT_CHARS)
    api_key: str = ""
    model: str = "gemini-2.5-flash"


class QueryRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=MAX_TRANSCRIPT_CHARS)
    question: str = Field(min_length=1, max_length=2_000)
    api_key: str = ""
    model: str = "gemini-2.5-flash"


def clean_transcript(text: str) -> str:
    """Normalize transcript text so the LLM receives consistent formatting."""
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def _resolve(api_key: str, model: str) -> tuple[str, str]:
    key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail="Add a Gemini API key, or set GEMINI_API_KEY on the server.",
        )
    chosen = model if model in ALLOWED_MODELS else ALLOWED_MODELS[0]
    return key, chosen


def _public_error(exc: BaseException, key: str) -> str:
    text = str(exc)
    if key:
        text = text.replace(key, "[redacted]")
    return text


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/analyze")
def analyze(body: AnalyzeRequest):
    transcript = clean_transcript(body.transcript)
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is empty.")
    key, model = _resolve(body.api_key, body.model)
    try:
        features, pains, themes, error = run_agent(transcript, api_key=key, model=model)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {_public_error(exc, key)}",
        ) from exc
    if error and key:
        error = error.replace(key, "[redacted]")
    return {
        "feature_requests": features,
        "pain_points": pains,
        "core_themes": themes,
        "error": error,
    }


@app.post("/api/query")
def ask(body: QueryRequest):
    transcript = clean_transcript(body.transcript)
    question = body.question.strip()
    if not transcript or not question:
        raise HTTPException(status_code=400, detail="Transcript and question are required.")
    key, model = _resolve(body.api_key, body.model)
    try:
        answer = query_transcript(question, transcript, api_key=key, model=model)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {_public_error(exc, key)}",
        ) from exc
    if key:
        answer = answer.replace(key, "[redacted]")
    return {"answer": answer}


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(PAGE_HTML)


PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>InsightStream</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #10110f;
      --panel: #181a16;
      --line: #2c2f28;
      --text: #f4f1e8;
      --muted: #a7a396;
      --accent: #e2b15a;
      --good: #8fbf8a;
      --warn: #d7a15a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 16px/1.5 "Avenir Next", "Segoe UI", sans-serif;
    }
    main { max-width: 980px; margin: 0 auto; padding: 40px 20px 80px; }
    h1 { font-weight: 600; letter-spacing: -0.03em; margin: 0 0 6px; font-size: 2rem; }
    .lede { color: var(--muted); margin: 0 0 28px; }
    form, .results, .ask {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px;
      margin-bottom: 18px;
    }
    label { display: block; font-size: 0.82rem; color: var(--muted); margin-bottom: 6px; }
    input, select, textarea, button {
      font: inherit;
      color: var(--text);
      background: #10110f;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    input, select, textarea { width: 100%; padding: 10px 12px; }
    textarea { min-height: 220px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1.4fr 0.8fr; gap: 12px; margin-bottom: 12px; }
    .actions { display: flex; gap: 10px; align-items: center; margin-top: 12px; flex-wrap: wrap; }
    button {
      background: var(--accent);
      color: #1a1408;
      border: 0;
      padding: 10px 16px;
      font-weight: 650;
      cursor: pointer;
    }
    button.secondary { background: transparent; color: var(--text); border: 1px solid var(--line); }
    button:disabled { opacity: 0.55; cursor: wait; }
    .note { color: var(--muted); font-size: 0.85rem; }
    .status { min-height: 1.4em; margin: 8px 0 0; }
    .status.error { color: #e7a0a0; }
    .group { margin-top: 8px; }
    .group h2 { font-size: 1.05rem; margin: 18px 0 10px; }
    .card {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 10px;
      background: #121410;
    }
    .card h3 { margin: 0 0 6px; font-size: 1rem; }
    .meta { color: var(--accent); font-size: 0.8rem; margin-bottom: 6px; }
    blockquote {
      margin: 8px 0 0;
      padding-left: 10px;
      border-left: 2px solid var(--line);
      color: var(--muted);
    }
    .empty { color: var(--muted); }
    @media (max-width: 700px) { .row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>InsightStream</h1>
    <p class="lede">Paste a customer interview. Each insight comes back with a verbatim quote, a confidence score, and a theme — after an auditor checks that the claim follows from the transcript.</p>
    <form id="analyze-form">
      <div class="row">
        <div>
          <label for="api-key">Gemini API key</label>
          <input id="api-key" type="password" autocomplete="off" placeholder="Leave blank if GEMINI_API_KEY is set on the server" />
        </div>
        <div>
          <label for="model">Model</label>
          <select id="model">
            <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
            <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
          </select>
        </div>
      </div>
      <label for="transcript">Transcript</label>
      <textarea id="transcript" placeholder="Paste the interview here, or load a .txt file."></textarea>
      <div class="actions">
        <button type="submit" id="analyze-btn">Analyze transcript</button>
        <button type="button" class="secondary" id="file-btn">Load .txt</button>
        <input id="file" type="file" accept=".txt,text/plain" hidden />
        <span class="note">Analysis usually takes 15–60 seconds.</span>
      </div>
      <p class="status" id="status"></p>
      <p class="note">Audio and video transcription stays in the local app: <code>streamlit run streamlit_app.py</code>. This hosted version analyzes text.</p>
    </form>
    <section class="results" id="results" hidden>
      <div class="actions">
        <button type="button" class="secondary" id="export-btn">Download CSV</button>
      </div>
      <div id="output"></div>
    </section>
    <section class="ask" id="ask" hidden>
      <label for="question">Missing something?</label>
      <input id="question" placeholder="Ask a question about this transcript" />
      <div class="actions">
        <button type="button" id="ask-btn">Ask</button>
      </div>
      <p class="status" id="answer"></p>
    </section>
  </main>
  <script>
    const transcript = document.getElementById("transcript");
    const statusEl = document.getElementById("status");
    const results = document.getElementById("results");
    const output = document.getElementById("output");
    const askSection = document.getElementById("ask");
    const answerEl = document.getElementById("answer");
    let lastPayload = null;

    document.getElementById("file-btn").onclick = () => document.getElementById("file").click();
    document.getElementById("file").onchange = async (event) => {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      transcript.value = await file.text();
    };

    function payload() {
      return {
        transcript: transcript.value,
        api_key: document.getElementById("api-key").value,
        model: document.getElementById("model").value,
      };
    }

    function setStatus(message, isError) {
      statusEl.textContent = message || "";
      statusEl.className = isError ? "status error" : "status";
    }

    async function readError(response) {
      try {
        const data = await response.json();
        if (typeof data.detail === "string") return data.detail;
        if (Array.isArray(data.detail) && data.detail[0] && data.detail[0].msg) return data.detail[0].msg;
      } catch (err) {}
      return "Request failed (" + response.status + ").";
    }

    function card(title, lines, quote, meta) {
      const el = document.createElement("article");
      el.className = "card";
      const h = document.createElement("h3");
      h.textContent = title || "Insight";
      el.appendChild(h);
      if (meta) {
        const m = document.createElement("div");
        m.className = "meta";
        m.textContent = meta;
        el.appendChild(m);
      }
      lines.forEach((line) => {
        if (!line) return;
        const p = document.createElement("p");
        p.textContent = line;
        el.appendChild(p);
      });
      if (quote) {
        const q = document.createElement("blockquote");
        q.textContent = quote;
        el.appendChild(q);
      }
      return el;
    }

    function render(data) {
      output.replaceChildren();
      const groups = [
        ["Feature requests", data.feature_requests || [], (item) => card(
          item.user_need,
          [item.proposed_solution],
          item.supporting_quote,
          "Confidence " + item.confidence_score + (item.source_id ? " · " + item.source_id : "")
        )],
        ["Pain points", data.pain_points || [], (item) => card(
          item.issue_description,
          [item.business_impact],
          item.supporting_quote,
          "Severity " + item.confidence_score + (item.source_id ? " · " + item.source_id : "")
        )],
        ["Core themes", data.core_themes || [], (item) => card(
          item.theme_name,
          [item.description],
          "",
          "Importance " + item.strategic_importance
        )],
      ];
      groups.forEach(([title, items, build]) => {
        const wrap = document.createElement("div");
        wrap.className = "group";
        const h = document.createElement("h2");
        h.textContent = title;
        wrap.appendChild(h);
        if (!items.length) {
          const empty = document.createElement("p");
          empty.className = "empty";
          empty.textContent = "None returned.";
          wrap.appendChild(empty);
        } else {
          items.forEach((item) => wrap.appendChild(build(item)));
        }
        output.appendChild(wrap);
      });
      results.hidden = false;
      askSection.hidden = false;
    }

    function csvCell(value) {
      const text = String(value ?? "");
      if (/[",\\n]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
      return text;
    }

    document.getElementById("export-btn").onclick = () => {
      if (!lastPayload) return;
      const rows = [["type", "title", "detail", "quote", "score", "source"]];
      (lastPayload.feature_requests || []).forEach((item) => rows.push([
        "feature_request", item.user_need, item.proposed_solution, item.supporting_quote, item.confidence_score, item.source_id
      ]));
      (lastPayload.pain_points || []).forEach((item) => rows.push([
        "pain_point", item.issue_description, item.business_impact, item.supporting_quote, item.confidence_score, item.source_id
      ]));
      (lastPayload.core_themes || []).forEach((item) => rows.push([
        "theme", item.theme_name, item.description, "", item.strategic_importance, ""
      ]));
      const csv = rows.map((row) => row.map(csvCell).join(",")).join("\\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "insightstream.csv";
      a.click();
      URL.revokeObjectURL(url);
    };

    document.getElementById("analyze-form").onsubmit = async (event) => {
      event.preventDefault();
      const button = document.getElementById("analyze-btn");
      button.disabled = true;
      setStatus("Analyzing transcript…");
      try {
        const response = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload()),
        });
        if (!response.ok) throw new Error(await readError(response));
        const data = await response.json();
        lastPayload = data;
        render(data);
        setStatus(data.error ? data.error : "Analysis complete.", Boolean(data.error));
      } catch (err) {
        setStatus(err.message || "Analysis failed.", true);
      } finally {
        button.disabled = false;
      }
    };

    document.getElementById("ask-btn").onclick = async () => {
      const question = document.getElementById("question").value.trim();
      if (!question) return;
      const button = document.getElementById("ask-btn");
      button.disabled = true;
      answerEl.textContent = "Looking through the transcript…";
      answerEl.className = "status";
      try {
        const response = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...payload(), question }),
        });
        if (!response.ok) throw new Error(await readError(response));
        const data = await response.json();
        answerEl.textContent = data.answer || "No answer returned.";
      } catch (err) {
        answerEl.textContent = err.message || "Query failed.";
        answerEl.className = "status error";
      } finally {
        button.disabled = false;
      }
    };
  </script>
</body>
</html>
"""
