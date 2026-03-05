# InsightStream-AI

**AI-powered qualitative research assistant. Extracts validated feature requests and pain points from customer interview transcripts.**

Built with LangGraph, Gemini 2.5 Flash, and Pydantic. Every extracted insight must cite a verbatim quote from the transcript, verified by an adversarial semantic entailment check. Insights that fail verification are dropped before the user sees them.

---

## The Problem

Analyzing 1 hour of customer interview requires roughly 4 hours of manual spreadsheet synthesis. Most product teams analyze 2–3 interviews per quarter and skip the rest. The result: roadmap decisions built on incomplete data.

## What This Does

Upload a transcript. The system extracts structured insights in 15–45 seconds. Each insight includes a verbatim supporting quote, a confidence score (1–10), and speaker attribution. An adversarial Auditor node verifies every insight against the source text before it reaches the user. The user reviews, approves or rejects each insight, and exports to CSV.

---

## Architecture

A 6-node LangGraph StateGraph with one conditional edge for parallel fan-out:

```
START → Router (conditional)
         ├─ ≤ 1500 words → Extractor (single)
         └─ > 1500 words → Chunk → Extractor × N (parallel)
                                      ↓
                                   Reducer (merge + quote dedup)
                                      ↓
                                   Formatter (Pydantic validation)
                                      ↓
                                   Deduplicator (LLM semantic merge)
                                      ↓
                                   Auditor (semantic entailment gate)
                                      ↓
                                   Synthesizer (3–5 core themes)
                                      ↓
                                   END → Streamlit UI
```

### What Each Node Does

**Router** — Checks word count. Short transcripts go through in one pass. Long transcripts are split into ~1,500-word chunks. Before splitting, a separate LLM call generates a 2–3 sentence global context summary that gets injected into every chunk so parallel workers don't lose narrative coherence.

**Extractor** — Calls Gemini with a Pydantic-enforced schema (`ExtractedInsights`). The system prompt instructs the model to ignore Speaker 0 (interviewer), extract verbatim quotes, apply a strict confidence rubric, and tag each insight with a source_id.

**Reducer** — Merges parallel outputs. De-duplicates by normalizing quotes (lowercase, whitespace collapse) against a seen-set. Restores the full transcript to state so the Auditor can verify against the complete text, not a chunk fragment.

**Formatter** — Validates every raw extracted dict against `FeatureRequest` and `PainPoint` Pydantic models. Catches out-of-range confidence scores, missing fields, and type errors. Logs per-item errors without stopping the pipeline.

**Deduplicator** — Uses a second LLM call to merge semantically equivalent insights (e.g., "slow login" and "auth speed is poor" become one canonical insight). On failure, passes through the original lists unchanged.

**Auditor** — The critical node. For each insight, it constructs a statement from the user_need + proposed_solution (or issue + impact), locates the supporting quote in the transcript with an 800-character context window, and asks Gemini: "Does Statement A logically follow from Transcript Segment B? Answer: Yes/No." Only "Yes" passes. "No," "Partially," API errors, timeouts, and malformed responses all result in rejection.

**Synthesizer** — Clusters verified insights into 3–5 core themes. The prompt instructs the model to weight themes by unique participant count (source_id), not by how many times something was mentioned.

---

## What's Verified vs. What's Claimed

This project serves as a portfolio case study. Here's an honest breakdown:

| Claim | Status | Evidence |
|-------|--------|----------|
| 6-node LangGraph pipeline with parallel extraction | **Implemented** | `build_graph()` in agent.py, lines 580–602 |
| Adversarial Auditor with semantic entailment | **Implemented** | `verify_entailment()` + `auditor_node()` in agent.py |
| Fail-closed design (rejects on any error) | **Implemented** | `except Exception: return False` in verify_entailment |
| Pydantic schema enforcement on all LLM output | **Implemented** | 6 Pydantic models + `with_structured_output()` on every LLM call |
| Two-phase dedup (syntactic + semantic) | **Implemented** | Reducer (quote matching) + Deduplicator (LLM merge) |
| Anti-loud-user-bias via source_id weighting | **Prompt-level** | Synthesizer prompt instructs it, but no hard-coded counting in code |
| Verbatim quote extraction | **Prompt-level** | Extractor prompt says "extract exactly word-for-word." No code verifies the quote exists as an exact substring in the transcript |
| 100% precision (0 hallucinations) | **Observed in testing** | 0 false positives across 15 adversarial test pairs. Not a statistical guarantee at scale |
| < 45s latency | **Architecture supports it** | Parallel chunking achieves this on paid API tier. Free tier hits rate limits |
| Manual Override Rate < 15% | **Defined, not measured** | Target metric. No telemetry in current code to track it |
| Correction log / data moat | **Designed, not built** | Full schema and flywheel documented. No implementation in codebase |
| PII masking at ingest | **Roadmap** | No automated PII handling. User must manually redact before upload |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Orchestration | LangGraph (StateGraph, conditional edges, parallel Send) |
| AI / LLM | Gemini 2.5 Flash (default) / Gemini 2.5 Pro |
| Schema Enforcement | Pydantic v2 + LangChain structured output |
| Media Processing | OpenAI Whisper + MoviePy + imageio-ffmpeg |
| Data Export | Pandas → CSV |

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

Open `http://localhost:8501`. Enter your Gemini API key in the Control Panel. Select Gemini 2.5 Flash.

---

## Usage

1. **Upload** — Paste text, upload `.txt` files, or upload `.mp3`/`.mp4`/`.wav` for Whisper transcription
2. **Review speakers** — For media uploads, check Speaker 0/1 labels. Click "Swap" if the diarization inverted them
3. **Analyze** — Click "Analyze transcript." Pipeline runs in 15–45 seconds (paid tier)
4. **Review** — Each insight shows confidence score, verbatim quote, and source. Approve or reject via checkboxes
5. **Search** — "Missing Something?" bar runs a RAG query against the full transcript
6. **Export** — Download approved insights as CSV

---

## Project Structure

```
InsightStream-AI/
├── README.md
├── agent.py               # LangGraph pipeline: 6 nodes, 4 prompts, 658 lines
├── app.py                 # Streamlit UI: 3 input methods, Verification Center, 554 lines
├── requirements.txt       # 11 dependencies
└── docs/
    ├── InsightStream_AI_Native_PRD.docx        # Aakash Gupta framework PRD
    ├── InsightStream_Model_Card.docx           # Mitchell et al. (2018) Model Card
    ├── InsightStream_Data_Flywheel.docx        # Buy vs. Bake, flywheel, unit economics
    ├── InsightStream_Ethics_Charter.docx       # PII, bias, EU AI Act, consent template
    ├── InsightStream_UX_Spec.docx              # Confidence-state UI, latency, export flow
    └── InsightStream_Eval_Harness.docx         # 15 test cases, auditor matrix, regression
```

---

## Documentation Suite

| Document | What It Covers |
|----------|---------------|
| **AI-Native PRD** | Golden evaluation dataset (5 test cases), model selection rationale, 5 failure modes with mitigations, success metrics |
| **Model Card** | Gemini 2.5 Flash specs (Sparse MoE, 1M context, Jan 2025 cutoff), 5 bias categories, eval results, 5 caveats, EU AI Act classification |
| **Data Flywheel** | Buy vs. Bake analysis with 3 trigger conditions, 3-step data flywheel, 5 proprietary data assets, unit economics ($0.02/interview), cold start strategy |
| **Ethics Charter** | Zero Data Retention architecture, PII handling (current gaps + NER roadmap), cultural bias analysis, participant disclosure template, 10-row risk matrix |
| **UX Spec** | Confidence-state card design (green/amber/gray), 45-second wait vibe map, Verification Center interaction model, Jira integration spec, 10 AI feedback signals |
| **Eval Harness** | 17-column Golden Dataset schema, 15 hard-mode test cases, auditor confusion matrix, 1–5 grading rubric, regression testing protocol |

---

## Metrics

**North Star:** Research Throughput Multiplier — hours of interview content analyzed per PM per week. Baseline: 2–3 hrs (1–2 interviews manually). Target: 15–20 hrs (6–8 interviews via InsightStream). Measurable post-deployment via upload volume telemetry.

| Metric | Target | Current Status |
|--------|--------|---------------|
| Research Throughput | 15–20 hrs/PM/week | Not yet measured. Requires production user base and telemetry |
| Precision (Auditor) | 100% | 100% observed across 15 test pairs (0 false positives) |
| Recall (Auditor) | ≥ 75% | 75–89% in testing. Rate limiting caused false negatives on free tier |
| Latency (60-min transcript) | < 45s | Achievable on paid API tier. Free tier exceeds due to 429 rate limits |
| Manual Override Rate | < 15% | Not yet measured. Requires production telemetry |
| Synthesis Coverage | > 90% | Not yet verified. Ground-truth tests blocked by API quota in testing |
| Cost per interview | ~$0.02 | Based on token-level calculation. Confirmed via API usage |

---

## Known Limitations

- **Extractor prompt says "Speaker 1" only.** In multi-speaker transcripts (Speaker 1, 2, 3), the prompt technically only asks for Speaker 1 insights. In practice, Gemini often extracts from all non-interviewer speakers, but this is LLM behavior, not a guarantee.
- **No quote verification.** The Extractor prompt instructs verbatim extraction, but no code checks that the supporting_quote exists as an exact substring in the transcript. A paraphrased quote can pass the Auditor's entailment check.
- **Diarization is heuristic.** Whisper speaker assignment alternates segments (even = Speaker 0, odd = Speaker 1). Real speaker identification is not performed. The Swap Speakers button is the safety net.
- **No PII handling.** Raw transcript text goes directly to the Gemini API. Users must manually redact before uploading.
- **Free-tier rate limits.** The Auditor makes one API call per insight. A transcript producing 10+ insights plus the extraction calls can exceed the 20 RPM free-tier limit.

---

## Roadmap

- [ ] Jira and Productboard API integrations
- [ ] Real-time live transcription capture
- [ ] Secure telemetry opt-in for correction pairs → model fine-tuning
- [ ] Automated NER-based PII masking at ingest
- [ ] Hard-coded source_id counting before Synthesizer (replace prompt-level anti-bias with deterministic logic)
- [ ] Quote existence verification (check supporting_quote is an exact substring of transcript)
- [ ] Multi-speaker prompt update (Speaker 1+ instead of Speaker 1 only)
- [ ] Confidence-state UI (green/amber/gray cards based on score)
- [ ] Retry logic on verify_entailment for 429 errors

---

Built by [Pranav Aadhithya Kalaibabu](https://github.com/pranavaadhithyakalaibabu)
