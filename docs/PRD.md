AI-Native PRD: InsightStream (v1.2)

Status: Approved | Owner: Pranav Aadhithya Kalaibabu - Product Manager | Framework: Aakash Gupta (Probabilistic/Deterministic Duality)

1. The Problem Hypothesis: "Synthesis Debt"

In professional research, the bottleneck is not data collection, but the compounding cost of qualitative analysis.

The Friction: Researchers spend 4 hours of analysis for every 1 hour of interview. Insights are often lost in spreadsheets because the "cost of extraction" exceeds the sprint velocity.

Personas: * UX Researcher: Needs evidence-based depth (grounded quotes).

Product Manager: Needs prioritized roadmap items and speed.

2. Technical Architecture (The Agentic Brain)

<img width="8192" height="718" alt="Raw Transcript Sanitization-2026-02-21-222510" src="https://github.com/user-attachments/assets/099f7b7f-3386-45fa-bd06-053a2acd2275" />

InsightStream uses a Sequential Agentic Workflow via LangGraph to maintain state and prevent context collapse.

2.1 The Pipeline

Sanitization Node (Deterministic): Regex-based cleaning of whitespace, non-UTF8 characters, and PII redaction.

Extractor Node (Probabilistic): Gemini 2.5 Flash ($temperature = 0$) identifies individual feature requests and pain points using a hardcoded 1–10 rubric.

Auditor Node (Deterministic): A verification script that matches every extracted insight against the source text. If no verbatim match is found, the insight is discarded.

Synthesizer Node (Probabilistic): Hierarchically clusters atomic insights into "Core Themes" across the entire dataset.

2.2 Model Justification

Model: Gemini 2.5 Flash.

Rationale: Optimized for Latency and COGS. Extraction is a high-token-volume, low-reasoning-density task. Flash provides 90%+ accuracy at 1/10th the cost of "Pro" models.

3. Requirements: Deterministic vs. Probabilistic

| Requirement Type | Feature | Specification |
| Deterministic | Data Cleaning | Regex scripts to strip carriage returns and normalize text for predictable token sequences. |
| Probabilistic | Pain Point Extraction | LLM-based identification of latent user needs based on tone and intent. |
| Deterministic | Grounding Mandate | System-level check: if !source_quote_exists: reject_insight. |
| Probabilistic | Theme Synthesis | Clustering N-interviews into a hierarchical summary (Themes $\rightarrow$ Features). |
| Deterministic | Schema Export | Validated CSV/Jira push of "Approved" insights only. |

4. User Journey & Workflow

<img width="4475" height="2930" alt="Raw Transcript Sanitization-2026-02-21-222919" src="https://github.com/user-attachments/assets/f4325318-d7be-40f0-b3ca-f99bacf54445" />

Ingest: User uploads raw text/transcript files.

Configure: User selects the "Rubric" (e.g., "Feature Requests" vs "Usability Friction").

Engine Run: The Agentic Workflow processes the data in parallel to prevent context window saturation.

The Verification Center: PM reviews extracted insights in a split-pane UI.

Human-in-the-Loop (HITL): User clicks "Approve," "Edit," or "Reject."

Sync: Approved insights are pushed to the product backlog.

5. Model Evaluation Metrics (Technical)

To ensure the system meets enterprise standards, we evaluate based on:

Extraction Precision: % of AI-extracted features that are factually present in the transcript. (Target: $>95\%$)

Recall: % of human-identified features captured by the AI. (Target: $>85\%$)

Hallucination Rate: Frequency of insights generated without a valid grounding quote. (Target: $0\%$)

Latency: End-to-end processing time for a 60-minute transcript. (Target: $<30s$)

6. Failure Handling & Edge Cases

Edge Case: Noisy Transcripts. * Handling: Sanitization node flags "Low Confidence" if word-error-rate (WER) markers are detected.

Edge Case: Contradictory Data.

Handling: If Interview A and Interview B provide opposing feedback, the Synthesizer node must create two distinct insights rather than "averaging" them.

Edge Case: Sample Bias. * Handling: The Synthesizer node must report a "Coverage Score" indicating if all uploaded transcripts contributed to the final themes, ensuring quieter participants aren't overshadowed by longer transcripts.

Failure Mode: API Timeout.

Handling: Exponential backoff (1s, 2s, 4s, 8s, 16s) with a local state save in LangGraph to resume processing.

7. Non-Functional Requirements

Privacy: Zero-retention on LLM provider side (Enterprise Tier).

Compliance: SOC2 Type II / GDPR-ready. The system utilizes an "Anonymization-at-Ingest" pattern where PII is redacted before tokens are sent to the LLM provider.

Scalability: System must handle up to 50 concurrent file uploads without performance degradation.

Cost Efficiency: Inference cost per 1-hour transcript must remain $<\$0.05$.

8. The Data Flywheel

<img width="6366" height="755" alt="Raw Transcript Sanitization-2026-02-21-222951" src="https://github.com/user-attachments/assets/155a561f-85f7-4e49-b9b0-1a691c466daa" />


The product's "Moat" is built via the Golden Dataset:

Input: User "Approved" and "Corrected" insights.

Output: A labeled dataset of industry-specific research.

Flywheel: These labels are used to fine-tune future iterations of the model, making InsightStream more accurate for the specific company's domain over time.

9. Non-Goals

Automated Roadmap Creation: We provide the building blocks; we do not replace the PM's strategic synthesis.

Real-time Transcription: We focus on the analysis of existing text data, not the live capture of audio.
