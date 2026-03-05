# InsightStream-AI
Autonomous AI User Research Analyst using LangGraph and Gemini 2.5 Flash.

# InsightStream-AI

**Autonomous AI User Research Analyst using LangGraph and Gemini 2.5 Flash.**

> Upload a customer interview transcript. Get validated, citation-backed feature requests and pain points in under 45 seconds. Zero hallucinations — every insight cites an exact verbatim quote, verified by an adversarial semantic entailment auditor.

---

## The Problem

Analyzing **1 hour** of qualitative user interviews requires **4 hours** of manual spreadsheet synthesis. Product teams analyze 2–3 "favorite" interviews and ignore the rest. Critical feedback never reaches the roadmap.

## What InsightStream Does

| Before (Manual) | After (InsightStream) |
|---|---|
| 4 hours per interview | 45 seconds |
| Subjective tagging | Verbatim quote citations |
| No verification | Adversarial semantic entailment |
| 2–3 interviews/quarter analyzed | All of them |
| $300–$600 in PM labor per interview | $0.02 in API costs |

---

## Architecture

InsightStream runs a **6-node LangGraph StateGraph** with parallel Map-Reduce extraction and an adversarial verification layer:

```
Upload → Router (conditional) → Extractor(s) [parallel] → Reducer → Formatter
       → Deduplicator (semantic) → Auditor (entailment gate) → Synthesizer → UI
```

**Key design decisions:**

- **Map-Reduce chunking** at 1,500-word boundaries with global context injection — prevents "Lost in the Middle" degradation for long interviews
- **Two-phase deduplication** — syntactic (quote-matching in Reducer) then semantic (LLM merge in Deduplicator) — minimizes API calls while catching paraphrased duplicates
- **Adversarial Auditor** — every insight is tested via `verify_entailment()`: "Does Statement A logically follow from Transcript Segment B? Yes/No." Fail = rejected. API error = rejected. The system never passes an unverified insight.
- **Anti-loud-user-bias** — themes weighted by unique `source_id` count, not raw mention frequency
- **Fail-closed** — every exception path defaults to "reject the insight," never "show it anyway"

---

## Features

- **Three input methods** — paste text, upload `.txt` files, or upload `.mp3`/`.mp4`/`.wav` for automatic Whisper transcription
- **Speaker Assignment Step-Gate** — review and swap Speaker 0/Speaker 1 labels before processing
- **Structured extraction** — feature requests and pain points with verbatim quotes, confidence scores (1–10), and speaker attribution
- **Core Theme synthesis** — 3–5 strategic themes weighted by participant breadth, not volume
- **Verification Center** — approve/reject checkboxes per insight with confidence scores
- **"Missing Something?" search** — RAG-style natural language query over the raw transcript
- **CSV export** — download approved insights for backlog ingestion

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit |
| **Orchestration** | LangGraph (StateGraph, conditional edges, parallel Send) |
| **AI / LLM** | Gemini 2.5 Flash (default) / Gemini 2.5 Pro |
| **Schema Enforcement** | Pydantic v2 + LangChain structured output |
| **Media Processing** | OpenAI Whisper + MoviePy + imageio-ffmpeg |
| **Data Export** | Pandas → CSV |

---

## Quick Start

### Prerequisites

- Python 3.11+ recommended (3.9+ supported)
- A Google Gemini API key — [get one here](https://aistudio.google.com/apikey)

### Installation

```bash
git clone https://github.com/pranavaadhithyakalaibabu/InsightStream-AI.git
cd InsightStream-AI

python -m venv .venv
source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

### Configuration

1. Open `http://localhost:8501` in your browser
2. Enter your Gemini API key in the Control Panel
3. Select Gemini 2.5 Flash (recommended) or Gemini 2.5 Pro

---

## Usage

**1. Upload** — Paste a transcript, upload `.txt` files, or upload audio/video for Whisper transcription.

**2. Review Speaker Labels** — For media uploads, check Speaker 0/1 labels. Click "Swap" if inverted.

**3. Analyze** — Click "Analyze transcript." The pipeline chunks, extracts, deduplicates, verifies, and synthesizes in one pass (15–45 seconds).

**4. Review** — Each insight shows confidence score, verbatim quote, and source attribution. Approve or reject in the Verification Center.

**5. Search** — Use "Missing Something?" to query the transcript for anything the pipeline missed.

**6. Export** — Download approved insights as CSV.

---

## Key Metrics

| Metric | Target | Method |
|--------|--------|--------|
| **Grounding Attribution** | 100% precision | Semantic entailment on every insight |
| **Latency** | < 45s (60-min transcript) | Parallel Map-Reduce |
| **Override Rate** | < 15% | Approve/reject tracking |
| **Coverage** | > 90% | Ground-truth eval harness |
| **Cost** | ~$0.02/interview | Gemini Flash pricing |

---

## Documentation

| Document | What It Covers |
|----------|---------------|
| [AI-Native PRD](docs/InsightStream_AI_Native_PRD.docx) | Golden dataset, model selection, failure modes, success metrics |
| [Model Card](docs/InsightStream_Model_Card.docx) | Mitchell et al. framework: biases, evals, ethical guardrails |
| [Data Flywheel](docs/InsightStream_Data_Flywheel.docx) | Buy vs. Bake, 3-step flywheel, unit economics, cold start |
| [Ethics Charter](docs/InsightStream_Ethics_Charter.docx) | PII protocol, bias mitigation, EU AI Act, participant disclosure |
| [UX Spec](docs/InsightStream_UX_Spec.docx) | Confidence-state UI, Verification Center, perceived latency |
| [Eval Harness](docs/InsightStream_Eval_Harness.docx) | 15 hard-mode test cases, auditor matrix, regression protocol |

---

## Project Structure

```
InsightStream-AI/
├── README.md
├── agent.py               # LangGraph pipeline (6 nodes + conditional routing)
├── app.py                 # Streamlit UI
├── requirements.txt
└── docs/
    ├── InsightStream_AI_Native_PRD.docx
    ├── InsightStream_Model_Card.docx
    ├── InsightStream_Data_Flywheel.docx
    ├── InsightStream_Ethics_Charter.docx
    ├── InsightStream_UX_Spec.docx
    └── InsightStream_Eval_Harness.docx
```

---

## How It Works

1. **Router** checks transcript length. Under 1,500 words → single pass. Over → parallel chunks with global context summary injected into each.

2. **Extractor** nodes call Gemini with Pydantic-enforced schema. The prompt ignores Speaker 0 (interviewer), extracts verbatim quotes, and applies a strict confidence rubric.

3. **Reducer** merges parallel outputs and deduplicates by normalized quote matching.

4. **Formatter** validates every item against Pydantic schemas. Invalid data is caught, not passed through.

5. **Deduplicator** merges semantically equivalent insights via LLM ("slow login" + "auth speed is poor" → one insight).

6. **Auditor** tests every insight: constructs a statement, locates the quote in the transcript, asks Gemini "Does A follow from B? Yes/No." Only "Yes" passes. Everything else is rejected.

7. **Synthesizer** clusters verified insights into 3–5 themes, weighted by unique participant count.

---

## Roadmap

- [ ] Jira and Productboard API integrations
- [ ] Real-time live transcription capture
- [ ] Secure telemetry opt-in for correction pairs → fine-tuning
- [ ] Automated NER-based PII masking
- [ ] Confidence-state UI (green/amber/gray cards)
- [ ] Multi-language support
- [ ] Batch processing with cross-interview synthesis

---

## License

Proprietary. All rights reserved.

---

Built by [Pranav Aadhithya Kalaibabu](https://github.com/pranavaadhithyakalaibabu)

