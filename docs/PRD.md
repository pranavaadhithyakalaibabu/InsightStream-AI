**AI-Native PRD: InsightStream (v2.0)**

**Status: Approved | Owner: Pranav Aadhithya Kalaibabu |**

**1. The Problem Hypothesis: "The Research Tax"**

Qualitative analysis is currently a tax researchers can't afford to pay.

The Friction: For every 1 hour of video, teams spend 4 hours in spreadsheets. Because the "cost per insight" is too high, critical user data sits rotting in folders instead of reaching the backlog.

The Goal: We are automating the grunt work of qualitative analysis. If the AI cannot provide a receipt (transcript link) for an insight, that insight does not exist to the system. We prioritize being right over being productive.

**2. Technical Architecture (The Logic)**

<img width="3666" height="6166" alt="Transcript Extraction-2026-02-23-030554" src="https://github.com/user-attachments/assets/2748fb99-0299-4369-9e87-59304bda4499" />

InsightStream uses a Sequential Workflow via LangGraph to maintain absolute control over the extraction and verification stages.

**2.1 The Pipeline**

Cleaning Node (Deterministic): Standardizes text and redacts PII. Ensures the model sees high-signal input only.

Diarized Filtering (Deterministic): Isolates "Participant" turns. We cut the interviewer's fluff to focus exclusively on the user's voice.

Map-Reduce Extraction (Probabilistic): Parallel chunking to ensure $O(1)$ latency. To prevent "Loss of Narrative" during chunking, we utilize a Global Context Injector (a recursive summary passed to every worker node). This ensures the extraction isn't lobotomized by the chunking process when a "Why" is separated from its "What" by 40 minutes of audio.

Semantic De-duplication (Probabilistic): Critical Cost Guardrail. Checks if "Login is slow" and "Auth takes too long" are the same point before auditing to minimize redundant token spend.

The Auditor (Trust Anchor): A zero-trust layer using Semantic Entailment. Every claim is verified against the source text; unproven claims are suppressed or flagged.

Weighted Synthesizer (Probabilistic): Clusters insights into themes. Fix: Insights are weighted by Participant Count, not Token Frequency, to prevent "Loud User" bias. Includes a Novelty Filter to prioritize weird contradictions over obvious consensus.

**3. Requirements: Deterministic vs. Probabilistic**



**4. User Journey & Workflow**

Ingest (MVP Scope): Support for .txt and media files (.mp4/.mp3). PDF support excluded for V1 to avoid formatting-edge-case bloat.

Speaker Assignment (Step-Gate): User views a "Transcript Preview" and confirms: "Speaker 0 is Interviewer." Includes a "Swap Speakers" toggle to fix diarization errors without re-processing.

Engine Run: Parallel extraction with integrated semantic de-duplication and global context injection.

Verification Center: Split-pane UI. Insights failing the Auditor check are flagged as "Unverified."

Force Verify (Override): Allows researchers to manually validate an "Unverified" insight if they know it occurred, bridging the AI trust gap.

Correction Log: Every edit, approval, or rejection is logged as the primary data moat.

**5. Evaluation Metrics (Technical)**

<img width="3840" height="2160" alt="llm-judge-flow-WHITE-4K" src="https://github.com/user-attachments/assets/4f69db89-c93d-478d-9761-576b8a293a4a" />

Grounding Attribution Rate: 100%. Every insight must link to a transcript timestamp or it is suppressed.

Synthesis Coverage: >90%. Percentage of validated atomic insights represented in final themes. Note: The system logs exactly which atomic insights were dropped during synthesis so they can be retrieved from the "Unverified" bucket if necessary.

Auditor Passing Rate: Monitoring metric to ensure the semantic entailment check isn't so strict it creates a "rejection loop" for valid but paraphrased data.

Manual Override Rate: % of sessions where "Force Verify" or "Speaker Swap" is utilized. (Target: $<15\%$). High rates indicate systemic model failure in the Auditor or Diarization nodes, identifying exactly where the agentic logic is becoming a productivity bottleneck.

Latency: <45s for 60-minute transcripts via parallelization.

**6. Failure Handling & Edge Cases**

Edge Case: Loud User Bias. Synthesizer must report a "Coverage Score." The UI will visually flag reports where a single participant dominates the insight generation to prevent over-indexing on one voice.

Edge Case: Boring Insights. The Synthesizer applies a "Novelty Filter" to highlight contradictions and specific technical friction over broad consensus.

Edge Case: Diarization Drift. Standard models (WhisperX/Pyannote) frequently "drift" in 60+ minute files. V1 implements a Chunk-Level Re-alignment or a UI "nudge" every 10 minutes of audio to prevent the Auditor from attributing Participant insights to the Interviewer.

Edge Case: Diarization Confusion. If the confidence score for speaker labeling falls below a set threshold, the UI must highlight speaker names (e.g., in yellow) during the Step-Gate to force manual verification before extraction.

Edge Case: The "Hallucination of Absence". To mitigate the risk of missed insights, the Verification Center includes a "Missing Something?" natural language search bar. This allows researchers to query the raw transcript directly to surface themes or pain points the extraction agent may have overlooked.

Failure Mode: Media Binaries. If FFmpeg/MoviePy is missing, the UI provides a clear non-technical error with a text-only fallback.

**7. Non-Functional Requirements**



**8. User Correction Log (The Moat)**

We ignore "flywheel" buzzwords. Our moat is the User Correction Log.

The Asset: A deterministic record of every time a researcher corrects, edits, or overrides an AI insight.

Storage & Access: To maintain the Stateless BYOK policy, these logs are stored in a local encrypted cache and can be exported by the user as a "Ground Truth" JSON/CSV file. For enterprise improvement, users may opt-in to a "Secure Telemetry" stream where anonymized correction pairs are transmitted to a private database for specialized model fine-tuning.

The Value: Our advantage isn't the model; it's the data. Every time a researcher corrects the AI, they are training our future proprietary advantage. This is the only way to outperform generic "Pro" models on niche domain jargon.

**9. Non-Goals**

Jira Integration: V1 focuses on clipboard/CSV export to fit researcher workflows.

Real-time Transcription: Focus is on analysis of existing data, not live capture.

PDF Formatting Logic: Stick to raw text and media to ensure stability at launch.
