"""
InsightStream Agent: LangGraph workflow for extracting and formatting
feature requests from customer interview transcripts.

State flows: START -> extractor_node -> formatter_node -> END
Each node receives the full state, updates its designated keys, and passes state forward.
"""

import os
from typing import List, Optional, Tuple, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
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
# Flow: transcript -> extractor -> formatter -> synthesizer -> END
# ---------------------------------------------------------------------------


class InsightStreamState(TypedDict, total=False):
    """State passed through the graph. Each node reads and optionally updates these keys."""

    # Set by the application before invoking the graph (from uploaded file).
    transcript: str

    # Optional: passed from app so extractor can build LLM when config is not forwarded to nodes.
    api_key: str
    model: str

    # Written by extractor_node; consumed by formatter_node.
    extracted_raw: list[dict]
    extracted_pain_points_raw: list[dict]

    # Written by formatter_node; consumed by synthesizer_node.
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

Do not paraphrase any supporting_quote; extract it exactly word-for-word. Output both feature_requests and pain_points; either list may be empty if none are found."""


# ---------------------------------------------------------------------------
# Node 1: Extractor — reads transcript, identifies feature requests and pain points
# ---------------------------------------------------------------------------
# Incoming state: transcript (required), others may be unset.
# Outgoing state: extracted_raw, extracted_pain_points_raw set (and error cleared on success).
# ---------------------------------------------------------------------------


def extractor_node(state: InsightStreamState, config: Optional[dict] = None) -> dict:
    """
    Reads the transcript from state and uses an LLM to identify feature requests and pain points.
    Updates state with extracted_raw and extracted_pain_points_raw for the formatter node.
    """
    transcript = state.get("transcript") or ""
    configurable = (config or {}).get("configurable") or {}
    api_key = state.get("api_key") or configurable.get("api_key")
    model = state.get("model") or configurable.get("model") or "gemini-2.5-flash"

    if not transcript.strip():
        return {"extracted_raw": [], "extracted_pain_points_raw": [], "error": "Transcript is empty."}

    try:
        key = api_key or os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model=model, api_key=key, temperature=0)
        structured_llm = llm.with_structured_output(ExtractedInsights)
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACTOR_SYSTEM),
            ("human", "Transcript:\n\n{transcript}"),
        ])
        chain = prompt | structured_llm
        result: ExtractedInsights = chain.invoke({"transcript": transcript})
        raw_requests = [r.model_dump() for r in result.feature_requests]
        raw_pain_points = [p.model_dump() for p in result.pain_points]
        return {
            "extracted_raw": raw_requests,
            "extracted_pain_points_raw": raw_pain_points,
            "error": "",
        }
    except Exception as e:
        return {"extracted_raw": [], "extracted_pain_points_raw": [], "error": f"Extractor failed: {e!s}"}


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
                )
                pain_points.append(pp.model_dump())
            else:
                errors.append(f"PainPoint {i}: expected dict, got {type(item).__name__}")
        except Exception as e:
            errors.append(f"PainPoint {i}: {e!s}")

    error_msg = "; ".join(errors) if errors else (state.get("error") or "")
    return {"feature_requests": feature_requests, "pain_points": pain_points, "error": error_msg}


# ---------------------------------------------------------------------------
# Node 3: Synthesizer — turns feature_requests + pain_points into core themes
# ---------------------------------------------------------------------------
# Incoming state: feature_requests, pain_points (from Formatter).
# Outgoing state: core_themes set to list of theme dicts.
# ---------------------------------------------------------------------------

SYNTHESIZER_SYSTEM = """You are a Head of Product. Review these extracted insights from multiple interviews and synthesize them into 3 to 5 overarching Core Themes that should drive the product roadmap. For each theme provide: theme_name (short), description (summarizing the overlapping feedback), and strategic_importance (1-10 based on frequency and severity)."""


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
            f"- {p.get('issue_description', '')} (impact: {p.get('business_impact', '')})"
            for p in pain_points
        )
        req_str = "\n".join(
            f"- {r.get('user_need', '')} (solution: {r.get('proposed_solution', '')})"
            for r in feature_requests
        )
        chain = prompt | structured_llm
        result: ThemeList = chain.invoke({"pain_points": pain_str, "feature_requests": req_str})
        core_themes = [t.model_dump() for t in result.themes]
        return {"core_themes": core_themes, "error": ""}
    except Exception as e:
        return {"core_themes": [], "error": f"Synthesizer failed: {e!s}"}


# ---------------------------------------------------------------------------
# Graph definition: extractor -> formatter -> synthesizer -> END
# ---------------------------------------------------------------------------

def build_graph():
    """
    Builds the LangGraph: START -> extractor_node -> formatter_node -> synthesizer_node -> END.
    State is passed in full to each node; nodes return only the keys they update.
    """
    graph = StateGraph(InsightStreamState)

    graph.add_node("extractor", extractor_node)
    graph.add_node("formatter", formatter_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge("__start__", "extractor")
    graph.add_edge("extractor", "formatter")
    graph.add_edge("formatter", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_agent(transcript: str, api_key: Optional[str] = None, model: str = "gemini-2.5-flash") -> Tuple[List[dict], List[dict], List[dict], str]:
    """
    Run the InsightStream agent on a transcript.
    Returns (feature_requests list, pain_points list, core_themes list, error_message). error_message is empty on success.
    """
    compiled = build_graph()
    config = {"configurable": {"api_key": api_key, "model": model}}
    initial: InsightStreamState = {
        "transcript": transcript,
        "api_key": api_key or "",
        "model": model,
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
