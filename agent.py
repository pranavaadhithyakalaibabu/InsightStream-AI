"""
InsightStream Agent: LangGraph workflow for extracting and formatting
feature requests from customer interview transcripts.

State flows: START --(conditional: router returns List[Send])--> extractor(s) [parallel when chunked] -> reducer -> formatter -> auditor -> synthesizer -> END.
Chunked parallelism: transcripts > ~10 min are split and extracted in parallel; reducer de-duplicates.
"""

import operator
import os
import re
from typing import Annotated, List, Optional, Tuple, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic schema for structured output (per PRD)
# ---------------------------------------------------------------------------


class FeatureRequest(BaseModel):
    """Single feature request extracted from a transcript."""

    user_need: str = Field(description="A 1-sentence summary of the problem.")
    proposed_solution: str = Field(description="The actionable feature.")
    supporting_quote: str = Field(
        description="The exact text extracted from the raw transcript."
    )
    confidence_score: int = Field(
        ge=1,
        le=10,
        description="Integer 1-10 rating how explicitly the user requested this.",
    )
    source_id: str = Field(
        default="",
        description="Identifier for who said this (e.g. Speaker 1, Participant, or P1) for weighting themes by unique participants.",
    )


class PainPoint(BaseModel):
    """Single pain point extracted from a transcript."""

    issue_description: str = Field(description="Concise summary of the friction.")
    business_impact: str = Field(description="How it affects their workflow.")
    supporting_quote: str = Field(
        description="The exact text extracted from the raw transcript."
    )
    confidence_score: int = Field(
        ge=1,
        le=10,
        description="Integer 1-10 severity rating.",
    )
    source_id: str = Field(
        default="",
        description="Identifier for who said this (e.g. Speaker 1, Participant) for weighting themes by unique participants.",
    )


class ExtractedInsights(BaseModel):
    """Wrapper for LLM structured output: feature requests and pain points."""

    feature_requests: List[FeatureRequest] = Field(
        default_factory=list,
        description="List of identified feature requests.",
    )
    pain_points: List[PainPoint] = Field(
        default_factory=list,
        description="List of identified pain points.",
    )


class CoreTheme(BaseModel):
    """Synthesized overarching theme from multiple insights."""

    theme_name: str = Field(description="Short name for the theme.")
    description: str = Field(description="Summary of the overlapping feedback.")
    strategic_importance: int = Field(
        ge=1,
        le=10,
        description="1-10 scale based on frequency and severity.",
    )


class ThemeList(BaseModel):
    """Wrapper for LLM structured output: list of core themes."""

    themes: List[CoreTheme] = Field(
        default_factory=list,
        description="List of synthesized core themes.",
    )


# ---------------------------------------------------------------------------
# LangGraph state: TypedDict defines what flows between nodes
# ---------------------------------------------------------------------------
# Flow: router -> extractor(s) [parallel when chunked] -> reducer -> formatter -> auditor -> synthesizer -> END
# ---------------------------------------------------------------------------

# ~1500 words ≈ 10 minutes of speech; used for chunked parallelism.
CHUNK_WORDS_10MIN = 1500


def _reduce_extractor_outputs(current: list, update: list) -> list:
    """Reducer: append each parallel extractor result (list of one dict) into accumulator."""
    if not isinstance(update, list):
        update = [update] if update else []
    return current + update


class InsightStreamState(TypedDict, total=False):
    """State passed through the graph. Each node reads and optionally updates these keys."""

    # Set by the application before invoking the graph (from uploaded file).
    transcript: str

    # Set by router when chunking; used by reducer/auditor to restore full transcript.
    full_transcript: str

    # Set by router: brief summary of full transcript so chunk workers have overarching context.
    global_context: str

    # Optional: passed from app so extractor can build LLM when config is not forwarded to nodes.
    api_key: str
    model: str

    # Accumulator for parallel extractor runs; reducer appends. Consumed by reducer_node.
    extractor_outputs: Annotated[list, _reduce_extractor_outputs]

    # Written by reducer_node; consumed by formatter_node.
    extracted_raw: list[dict]
    extracted_pain_points_raw: list[dict]

    # Written by formatter_node; consumed by auditor_node then synthesizer_node.
    feature_requests: list[dict]
    pain_points: list[dict]

    # Written by synthesizer_node; final output shown in the UI.
    core_themes: list[dict]

    # Optional: hold the last error message for the UI to display.
    error: str


# ---------------------------------------------------------------------------
# LLM and prompts
# ---------------------------------------------------------------------------

EXTRACTOR_SYSTEM = """You are an expert User Research Analyst. Extract actionable product feature requests from the provided interview transcript(s). You MUST carefully read and extract data from ALL interviews provided in the text block, do not stop after the first one.

IGNORE all text from the Interviewer (Speaker 0). ONLY extract insights from the Participant (Speaker 1).

Extract BOTH feature requests AND distinct pain points. For FEATURE REQUESTS, use this strict rubric for confidence_score (1-10):
9 to 10 (Explicit): The user explicitly and directly asks for a specific feature or solution.
7 to 8 (Implicit High): The user describes a strong pain point that clearly implies a specific feature, but does not explicitly name it.
4 to 6 (Implicit Low): The user mentions a minor annoyance; a feature request can be inferred but requires guesswork.
1 to 3 (Vague): Vague feedback with no clear actionable product direction.

For PAIN POINTS, use this strict rubric for confidence_score (1-10 severity):
9 to 10: Critical workflow blocker—severely impacts ability to do their job.
7 to 8: Moderate friction or time-waster—noticeable impact on efficiency.
4 to 6: Minor annoyance—inconvenient but workable.
1 to 3: Vague complaint—no clear severity or impact.

Do not paraphrase any supporting_quote; extract it exactly word-for-word. For each item set source_id to the speaker label when present (e.g. Speaker 1, Speaker 2, or Participant); leave empty if unclear. Output both feature_requests and pain_points; either list may be empty if none are found."""


# ---------------------------------------------------------------------------
# Node: Router — decides single run vs chunked parallel; returns list of Send(extractor, state)
# ---------------------------------------------------------------------------


def _chunk_transcript(transcript: str, words_per_chunk: int = CHUNK_WORDS_10MIN) -> List[str]:
    """Split transcript into chunks of approximately words_per_chunk words."""
    words = transcript.split()
    if len(words) <= words_per_chunk:
        return [transcript] if transcript.strip() else []
    chunks: List[str] = []
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i : i + words_per_chunk]
        chunks.append(" ".join(chunk_words))
    return chunks


def _summarize_for_global_context(transcript: str, api_key: Optional[str] = None, model: str = "gemini-2.5-flash") -> str:
    """Quick Flash/Pro call: 2–3 sentence summary of the full transcript for chunk workers."""
    if not (transcript or "").strip():
        return ""
    try:
        key = api_key or os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model=model, api_key=key, temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("human", "Summarize this interview transcript in 2–3 sentences: the overarching goal and main topics. Be concise.\n\nTranscript:\n{transcript}"),
        ])
        response = (prompt | llm).invoke({"transcript": transcript[:30000]})
        return (response.content or "").strip()
    except Exception:
        return ""


def router_node(state: InsightStreamState, config: Optional[dict] = None) -> List[Send]:
    """
    Builds global_context (summary of full transcript), then if transcript is longer than ~10 min,
    chunks it and sends each chunk to extractor in parallel with global_context; otherwise sends full transcript once.
    """
    transcript = state.get("transcript") or ""
    configurable = (config or {}).get("configurable") or {}
    api_key = state.get("api_key") or configurable.get("api_key")
    model = state.get("model") or configurable.get("model") or "gemini-2.5-flash"

    global_context = _summarize_for_global_context(transcript, api_key=api_key, model=model)
    state_with_global = {**state, "global_context": global_context}

    words = len(transcript.split())
    if words <= CHUNK_WORDS_10MIN or not transcript.strip():
        return [Send("extractor", {**state_with_global, "transcript": transcript})]
    chunks = _chunk_transcript(transcript, CHUNK_WORDS_10MIN)
    state_with_full = {**state_with_global, "full_transcript": transcript}
    return [Send("extractor", {**state_with_full, "transcript": chunk}) for chunk in chunks]


# ---------------------------------------------------------------------------
# Node 1: Extractor — reads transcript (or chunk), identifies feature requests and pain points
# ---------------------------------------------------------------------------
# Incoming state: transcript (required), others may be unset.
# Outgoing state: extractor_outputs (one item for reducer to aggregate).
# ---------------------------------------------------------------------------


def extractor_node(state: InsightStreamState, config: Optional[dict] = None) -> dict:
    """
    Reads the transcript (or one chunk) from state and uses an LLM to identify feature requests
    and pain points. Returns one element for extractor_outputs so reducer can merge parallel runs.
    """
    transcript = state.get("transcript") or ""
    configurable = (config or {}).get("configurable") or {}
    api_key = state.get("api_key") or configurable.get("api_key")
    model = state.get("model") or configurable.get("model") or "gemini-2.5-flash"

    if not transcript.strip():
        return {"extractor_outputs": [{"extracted_raw": [], "extracted_pain_points_raw": []}]}

    global_context = state.get("global_context") or ""

    try:
        key = api_key or os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model=model, api_key=key, temperature=0)
        structured_llm = llm.with_structured_output(ExtractedInsights)
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACTOR_SYSTEM),
            ("human", "Overall interview context (use this to keep narrative coherence):\n{global_context}\n\nTranscript (segment or full):\n\n{transcript}"),
        ])
        chain = prompt | structured_llm
        result: ExtractedInsights = chain.invoke({"transcript": transcript, "global_context": global_context or "(none provided)"})
        raw_requests = [r.model_dump() for r in result.feature_requests]
        raw_pain_points = [p.model_dump() for p in result.pain_points]
        return {
            "extractor_outputs": [{"extracted_raw": raw_requests, "extracted_pain_points_raw": raw_pain_points}],
        }
    except Exception as e:
        return {"extractor_outputs": [{"extracted_raw": [], "extracted_pain_points_raw": []}], "error": f"Extractor failed: {e!s}"}


# ---------------------------------------------------------------------------
# Node: Reducer — merges parallel extractor outputs and de-duplicates by supporting_quote
# ---------------------------------------------------------------------------
# Incoming state: extractor_outputs (list of {extracted_raw, extracted_pain_points_raw}), full_transcript.
# Outgoing state: extracted_raw, extracted_pain_points_raw, transcript (restored if chunked).
# ---------------------------------------------------------------------------


def _normalize_quote_for_dedup(quote: str) -> str:
    """Normalize supporting_quote for de-duplication (lowercase, collapse whitespace)."""
    return re.sub(r"\s+", " ", (quote or "").lower().strip())


def reducer_node(state: InsightStreamState, config: Optional[dict] = None) -> dict:
    """
    Merges all extractor_outputs from parallel runs, de-duplicates by supporting_quote,
    and sets extracted_raw and extracted_pain_points_raw. Restores full_transcript to transcript for auditor.
    """
    outputs = state.get("extractor_outputs") or []
    full_transcript = state.get("full_transcript") or state.get("transcript") or ""

    all_raw: list[dict] = []
    all_pain: list[dict] = []
    for out in outputs:
        if isinstance(out, dict):
            all_raw.extend(out.get("extracted_raw") or [])
            all_pain.extend(out.get("extracted_pain_points_raw") or [])

    seen_quotes_fr: set[str] = set()
    dedup_raw: list[dict] = []
    for r in all_raw:
        q = _normalize_quote_for_dedup(r.get("supporting_quote", ""))
        if q and q in seen_quotes_fr:
            continue
        if q:
            seen_quotes_fr.add(q)
        dedup_raw.append(r)

    seen_quotes_pp: set[str] = set()
    dedup_pain: list[dict] = []
    for p in all_pain:
        q = _normalize_quote_for_dedup(p.get("supporting_quote", ""))
        if q and q in seen_quotes_pp:
            continue
        if q:
            seen_quotes_pp.add(q)
        dedup_pain.append(p)

    return {
        "extracted_raw": dedup_raw,
        "extracted_pain_points_raw": dedup_pain,
        "transcript": full_transcript,
        "error": state.get("error") or "",
    }


# ---------------------------------------------------------------------------
# Semantic de-duplication: merge insights that mean the same thing before Auditor
# ---------------------------------------------------------------------------

class MergedInsights(BaseModel):
    """Deduplicated feature requests and pain points (same shape as ExtractedInsights)."""

    feature_requests: List[FeatureRequest] = Field(default_factory=list)
    pain_points: List[PainPoint] = Field(default_factory=list)


DEDUP_SYSTEM = """You are a research analyst. Given a list of feature requests and pain points from interview transcripts, merge any that are SEMANTICALLY THE SAME (e.g. "The login is slow" and "Auth speed is poor" are the same; "Bulk export" and "Export to CSV" may be the same). For each merged item: keep the strongest supporting_quote and the highest confidence_score; combine user_need/issue_description and proposed_solution/business_impact into one clear statement. Preserve source_id (if multiple, keep the first or combine). Return only the merged, deduplicated lists. Do not add new items or drop items that have no duplicate."""


def deduplicator_node(state: InsightStreamState, config: Optional[dict] = None) -> dict:
    """
    Uses semantic similarity (via LLM) to merge duplicate insights before the expensive Auditor phase.
    Reduces token usage and keeps one canonical insight per theme.
    """
    feature_requests = state.get("feature_requests") or []
    pain_points = state.get("pain_points") or []
    configurable = (config or {}).get("configurable") or {}
    api_key = state.get("api_key") or configurable.get("api_key")
    model = state.get("model") or configurable.get("model") or "gemini-2.5-flash"

    if not feature_requests and not pain_points:
        return {"feature_requests": [], "pain_points": [], "error": state.get("error") or ""}

    try:
        key = api_key or os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model=model, api_key=key, temperature=0)
        structured_llm = llm.with_structured_output(MergedInsights)
        prompt = ChatPromptTemplate.from_messages([
            ("system", DEDUP_SYSTEM),
            ("human", "Feature requests:\n{feature_requests}\n\nPain points:\n{pain_points}"),
        ])
        fr_str = "\n---\n".join(
            f"user_need: {r.get('user_need', '')}\nproposed_solution: {r.get('proposed_solution', '')}\nsupporting_quote: {r.get('supporting_quote', '')}\nconfidence: {r.get('confidence_score')}\nsource_id: {r.get('source_id', '')}"
            for r in feature_requests
        )
        pp_str = "\n---\n".join(
            f"issue_description: {p.get('issue_description', '')}\nbusiness_impact: {p.get('business_impact', '')}\nsupporting_quote: {p.get('supporting_quote', '')}\nconfidence: {p.get('confidence_score')}\nsource_id: {p.get('source_id', '')}"
            for p in pain_points
        )
        result: MergedInsights = (prompt | structured_llm).invoke({
            "feature_requests": fr_str or "(none)",
            "pain_points": pp_str or "(none)",
        })
        merged_fr = [r.model_dump() for r in result.feature_requests]
        merged_pp = [p.model_dump() for p in result.pain_points]
        return {"feature_requests": merged_fr, "pain_points": merged_pp, "error": state.get("error") or ""}
    except Exception as e:
        return {"feature_requests": feature_requests, "pain_points": pain_points, "error": state.get("error") or f"Deduplicator: {e!s}"}


# ---------------------------------------------------------------------------
# Node 2: Formatter — validates extracted_raw and extracted_pain_points_raw into strict schema
# ---------------------------------------------------------------------------
# Incoming state: extracted_raw, extracted_pain_points_raw (from Extractor).
# Outgoing state: feature_requests, pain_points set to validated list of dicts.
# ---------------------------------------------------------------------------


def formatter_node(state: InsightStreamState, config: Optional[dict] = None) -> dict:
    """
    Validates extracted_raw and extracted_pain_points_raw against FeatureRequest and PainPoint
    schemas, and writes feature_requests and pain_points (strict JSON-ready dicts).
    """
    raw_list = state.get("extracted_raw") or []
    raw_pain = state.get("extracted_pain_points_raw") or []
    feature_requests: list[dict] = []
    pain_points: list[dict] = []
    errors: list[str] = []

    for i, item in enumerate(raw_list):
        try:
            if isinstance(item, dict):
                req = FeatureRequest(
                    user_need=item.get("user_need", ""),
                    proposed_solution=item.get("proposed_solution", ""),
                    supporting_quote=item.get("supporting_quote", ""),
                    confidence_score=int(item.get("confidence_score", 5)),
                    source_id=item.get("source_id", "") or "",
                )
                feature_requests.append(req.model_dump())
            else:
                errors.append(f"Feature {i}: expected dict, got {type(item).__name__}")
        except Exception as e:
            errors.append(f"Feature {i}: {e!s}")

    for i, item in enumerate(raw_pain):
        try:
            if isinstance(item, dict):
                pp = PainPoint(
                    issue_description=item.get("issue_description", ""),
                    business_impact=item.get("business_impact", ""),
                    supporting_quote=item.get("supporting_quote", ""),
                    confidence_score=int(item.get("confidence_score", 5)),
                    source_id=item.get("source_id", "") or "",
                )
                pain_points.append(pp.model_dump())
            else:
                errors.append(f"PainPoint {i}: expected dict, got {type(item).__name__}")
        except Exception as e:
            errors.append(f"PainPoint {i}: {e!s}")

    error_msg = "; ".join(errors) if errors else (state.get("error") or "")
    return {"feature_requests": feature_requests, "pain_points": pain_points, "error": error_msg}


# ---------------------------------------------------------------------------
# Semantic entailment: Gemini 2.5 Flash verifies if insight logically follows from segment
# ---------------------------------------------------------------------------


def _get_segment_around_quote(transcript: str, quote: str, window_chars: int = 800) -> str:
    """Find the transcript segment containing the supporting quote; return a window around it."""
    if not transcript or not quote or not quote.strip():
        return transcript[:window_chars] if transcript else ""
    quote_clean = re.sub(r"\s+", " ", quote.strip())
    transcript_norm = re.sub(r"\s+", " ", transcript)
    pos = transcript_norm.find(quote_clean)
    if pos == -1:
        # Fuzzy: use first 200 chars of quote to locate
        short = quote_clean[:200].strip()
        if short:
            pos = transcript_norm.find(short)
        if pos == -1:
            return transcript[:window_chars] if transcript else ""
    start = max(0, pos - window_chars // 2)
    end = min(len(transcript), pos + len(quote) + window_chars // 2)
    return transcript[start:end].strip()


def verify_entailment(
    statement_a: str,
    segment_b: str,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> bool:
    """
    Uses Gemini 2.5 Flash to verify if Statement A logically follows from Transcript Segment B.
    Returns True if the model answers Yes, False otherwise.
    """
    if not statement_a.strip() or not segment_b.strip():
        return False
    try:
        key = api_key or os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model=model, api_key=key, temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("human", "Does Statement A logically follow from Transcript Segment B? Answer: Yes/No.\n\nStatement A:\n{statement_a}\n\nTranscript Segment B:\n{segment_b}"),
        ])
        chain = prompt | llm
        response = chain.invoke({"statement_a": statement_a, "segment_b": segment_b})
        text = (response.content or "").strip().upper()
        return text.startswith("YES")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Node: Auditor — filters insights by semantic entailment (insight follows from transcript segment)
# ---------------------------------------------------------------------------
# Incoming state: feature_requests, pain_points, transcript (for segment lookup), api_key, model.
# Outgoing state: feature_requests, pain_points (filtered to only those that pass entailment).
# ---------------------------------------------------------------------------


def auditor_node(state: InsightStreamState, config: Optional[dict] = None) -> dict:
    """
    For each feature request and pain point, verifies that the insight logically follows
    from the source transcript segment using Gemini 2.5 Flash. Drops insights that fail.
    """
    feature_requests = state.get("feature_requests") or []
    pain_points = state.get("pain_points") or []
    transcript = state.get("transcript") or ""
    configurable = (config or {}).get("configurable") or {}
    api_key = state.get("api_key") or configurable.get("api_key")
    model = state.get("model") or configurable.get("model") or "gemini-2.5-flash"

    if not transcript.strip():
        return {"feature_requests": feature_requests, "pain_points": pain_points, "error": state.get("error") or ""}

    filtered_fr: list[dict] = []
    for req in feature_requests:
        statement_a = f"Need: {req.get('user_need', '')}. Solution: {req.get('proposed_solution', '')}."
        segment_b = _get_segment_around_quote(transcript, req.get("supporting_quote", ""))
        if verify_entailment(statement_a, segment_b, api_key=api_key, model=model):
            filtered_fr.append(req)
    filtered_pp: list[dict] = []
    for pp in pain_points:
        statement_a = f"Issue: {pp.get('issue_description', '')}. Impact: {pp.get('business_impact', '')}."
        segment_b = _get_segment_around_quote(transcript, pp.get("supporting_quote", ""))
        if verify_entailment(statement_a, segment_b, api_key=api_key, model=model):
            filtered_pp.append(pp)

    return {"feature_requests": filtered_fr, "pain_points": filtered_pp, "error": state.get("error") or ""}


# ---------------------------------------------------------------------------
# Node 3: Synthesizer — turns feature_requests + pain_points into core themes
# ---------------------------------------------------------------------------
# Incoming state: feature_requests, pain_points (from Formatter).
# Outgoing state: core_themes set to list of theme dicts.
# ---------------------------------------------------------------------------

SYNTHESIZER_SYSTEM = """You are a Head of Product. Review these extracted insights from multiple interviews and synthesize them into 3 to 5 overarching Core Themes that should drive the product roadmap.

CRITICAL: Avoid "loud user bias." Weight themes by the NUMBER OF UNIQUE PARTICIPANTS (source_id) who mentioned them, not by how many times something was said. One participant repeating the same point should not outweigh several different participants each mentioning a theme once.

For each theme provide: theme_name (short), description (summarizing the overlapping feedback), and strategic_importance (1-10 based on how many distinct participants raised it and severity, not raw mention count)."""


def synthesizer_node(state: InsightStreamState, config: Optional[dict] = None) -> dict:
    """
    Reads feature_requests and pain_points from state, calls LLM to synthesize
    3-5 core themes, and writes core_themes to state.
    """
    feature_requests = state.get("feature_requests") or []
    pain_points = state.get("pain_points") or []
    configurable = (config or {}).get("configurable") or {}
    api_key = state.get("api_key") or configurable.get("api_key")
    model = state.get("model") or configurable.get("model") or "gemini-2.5-flash"

    if not feature_requests and not pain_points:
        return {"core_themes": [], "error": state.get("error") or ""}

    try:
        key = api_key or os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model=model, api_key=key, temperature=0)
        structured_llm = llm.with_structured_output(ThemeList)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYNTHESIZER_SYSTEM),
            ("human", "Pain points:\n{pain_points}\n\nFeature requests:\n{feature_requests}"),
        ])
        pain_str = "\n".join(
            f"- [source_id: {p.get('source_id') or 'unknown'}] {p.get('issue_description', '')} (impact: {p.get('business_impact', '')})"
            for p in pain_points
        )
        req_str = "\n".join(
            f"- [source_id: {r.get('source_id') or 'unknown'}] {r.get('user_need', '')} (solution: {r.get('proposed_solution', '')})"
            for r in feature_requests
        )
        chain = prompt | structured_llm
        result: ThemeList = chain.invoke({"pain_points": pain_str, "feature_requests": req_str})
        core_themes = [t.model_dump() for t in result.themes]
        return {"core_themes": core_themes, "error": ""}
    except Exception as e:
        return {"core_themes": [], "error": f"Synthesizer failed: {e!s}"}


# ---------------------------------------------------------------------------
# Graph definition: START --(conditional: router_send)--> extractor(s) [parallel] -> reducer -> ...
# ---------------------------------------------------------------------------
# The "router" is a conditional edge function (returns List[Send]), not a node; nodes must return a state dict.
# ---------------------------------------------------------------------------

def build_graph():
    """
    Builds the LangGraph: START --(router_send)--> extractor(s) [parallel when chunked] -> reducer
    -> formatter -> auditor -> synthesizer -> END.
    """
    graph = StateGraph(InsightStreamState)

    graph.add_node("extractor", extractor_node)
    graph.add_node("reducer", reducer_node)
    graph.add_node("formatter", formatter_node)
    graph.add_node("deduplicator", deduplicator_node)
    graph.add_node("auditor", auditor_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_conditional_edges("__start__", router_node)
    graph.add_edge("extractor", "reducer")
    graph.add_edge("reducer", "formatter")
    graph.add_edge("formatter", "deduplicator")
    graph.add_edge("deduplicator", "auditor")
    graph.add_edge("auditor", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def query_transcript(
    question: str,
    transcript: str,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> str:
    """
    RAG-style query: answer the user's question based on the interview transcript.
    Uses a direct LLM call over the transcript (no retrieval chunking) for simplicity.
    """
    if not (question or "").strip() or not (transcript or "").strip():
        return ""
    try:
        key = api_key or os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model=model, api_key=key, temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("human", "Based only on the following interview transcript, answer this question. If the transcript does not contain relevant information, say so briefly.\n\nQuestion: {question}\n\nTranscript:\n{transcript}"),
        ])
        # Truncate very long transcripts to stay within context
        transcript_trim = transcript[:80000] if len(transcript) > 80000 else transcript
        response = (prompt | llm).invoke({"question": question.strip(), "transcript": transcript_trim})
        return (response.content or "").strip()
    except Exception as e:
        return f"Query failed: {e!s}"


def run_agent(transcript: str, api_key: Optional[str] = None, model: str = "gemini-2.5-flash") -> Tuple[List[dict], List[dict], List[dict], str]:
    """
    Run the InsightStream agent on a transcript.
    Returns (feature_requests list, pain_points list, core_themes list, error_message). error_message is empty on success.
    """
    compiled = build_graph()
    config = {"configurable": {"api_key": api_key, "model": model}}
    initial: InsightStreamState = {
        "transcript": transcript,
        "full_transcript": "",
        "global_context": "",
        "api_key": api_key or "",
        "model": model,
        "extractor_outputs": [],
        "extracted_raw": [],
        "extracted_pain_points_raw": [],
        "feature_requests": [],
        "pain_points": [],
        "core_themes": [],
        "error": "",
    }
    final_state = compiled.invoke(initial, config=config)
    return (
        final_state.get("feature_requests") or [],
        final_state.get("pain_points") or [],
        final_state.get("core_themes") or [],
        final_state.get("error") or "",
    )
