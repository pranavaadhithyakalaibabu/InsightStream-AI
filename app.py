"""
InsightStream: Premium multimodal enterprise UI for the AI User Research Analyst.
Input via paste, document upload, or media (Whisper transcription).
Runs the LangGraph agent and displays structured feature requests in a card layout.
"""

import os
import re
import subprocess
import tempfile

import extra_streamlit_components as stx
import pandas as pd
import streamlit as st

from agent import run_agent

# Bundled ffmpeg path (imageio-ffmpeg); set once so Whisper and MoviePy use it
_FFMPEG_EXE = None


def _get_ffmpeg_exe():
    """Return path to ffmpeg (imageio-ffmpeg bundle or system). Used so Whisper finds ffmpeg."""
    global _FFMPEG_EXE
    if _FFMPEG_EXE is not None:
        return _FFMPEG_EXE
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            _FFMPEG_EXE = exe
            # Also put on PATH for moviepy and any other subprocess
            ffmpeg_dir = os.path.dirname(exe)
            path_sep = ";" if os.name == "nt" else ":"
            os.environ["PATH"] = ffmpeg_dir + path_sep + os.environ.get("PATH", "")
            return exe
    except Exception:
        pass
    return None


def _patch_whisper_ffmpeg():
    """Make Whisper use the bundled ffmpeg path (whisper uses hardcoded 'ffmpeg' in subprocess)."""
    exe = _get_ffmpeg_exe()
    if not exe:
        return
    try:
        import whisper.audio as whisper_audio
        from subprocess import CalledProcessError, run
        import numpy as np
        SAMPLE_RATE = getattr(whisper_audio, "SAMPLE_RATE", 16000)

        def load_audio_patched(file: str, sr: int = SAMPLE_RATE):
            cmd = [
                exe,
                "-nostdin", "-threads", "0",
                "-i", file,
                "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
                "-ar", str(sr), "-",
            ]
            try:
                out = run(cmd, capture_output=True, check=True).stdout
            except CalledProcessError as e:
                raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
            return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

        whisper_audio.load_audio = load_audio_patched
    except Exception:
        pass


def transcribe_media(media_file, model_size: str = "base") -> str:
    """
    Transcribe uploaded audio or video to text using OpenAI Whisper.
    For video (e.g. .mp4), extracts audio with moviepy first, then transcribes.
    Caller must ensure media_file is an upload with .name and .read()/.getvalue().
    """
    _get_ffmpeg_exe()
    _patch_whisper_ffmpeg()
    raw = media_file.getvalue() if hasattr(media_file, "getvalue") else media_file.read()
    suffix = os.path.splitext(getattr(media_file, "name", "audio.mp3"))[1].lower() or ".mp3"
    if suffix not in (".mp4", ".mp3", ".wav", ".m4a"):
        suffix = ".mp3"
    video_path = None
    audio_path = None
    video_clip = None
    audio_clip = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            video_path = tmp.name
        if suffix == ".mp4":
            from moviepy import VideoFileClip
            video_clip = VideoFileClip(video_path)
            fd, audio_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            video_clip.audio.write_audiofile(audio_path, logger=None)
            video_clip.close()
            video_clip = None
        else:
            audio_path = video_path
        import whisper
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path, language="en", verbose=False)
        return (result.get("text") or "").strip()
    finally:
        if video_clip is not None and hasattr(video_clip, "close"):
            video_clip.close()
        if audio_clip is not None and hasattr(audio_clip, "close"):
            audio_clip.close()
        for p in (video_path, audio_path):
            if p and p != video_path and os.path.isfile(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        if video_path and os.path.isfile(video_path):
            try:
                os.unlink(video_path)
            except OSError:
                pass


def clean_transcript(text: str) -> str:
    """Normalize transcript text so the LLM receives consistent formatting regardless of input method."""
    if not text:
        return ""
    # Normalize line endings: \r\n and \r -> single \n
    text = re.sub(r"\r\n|\r", "\n", text)
    # Collapse multiple newlines to a single newline
    text = re.sub(r"\n\n+", "\n", text)
    # Collapse multiple consecutive spaces to a single space
    text = re.sub(r" +", " ", text)
    return text.strip()


def _safe_key(name: str) -> str:
    """Sanitize item name for Streamlit widget keys (alphanumeric and underscore only)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


# ---------------------------------------------------------------------------
# Page config and layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="InsightStream",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium UI: hide default header and footer (standalone SaaS look)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* Hide default Streamlit branding and sidebar */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        section[data-testid="stSidebar"] { display: none !important; }
        div.block-container { padding-top: 2rem !important; }
        /* Minimalist underline tabs: transparent, bottom border only on active */
        [data-baseweb="tab-list"] { border-bottom: 1px solid #2A2A2A !important; gap: 0.5rem; background: transparent !important; }
        button[data-baseweb="tab"], [data-baseweb="tab"] {
            border: none !important;
            border-bottom: 2px solid transparent !important;
            border-radius: 0 !important;
            background: transparent !important;
            color: #6b7280 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"], [data-baseweb="tab"][aria-selected="true"] {
            color: #FAFAFA !important;
            border-bottom-color: #3B82F6 !important;
        }
        [data-theme="light"] button[data-baseweb="tab"][aria-selected="true"],
        [data-theme="light"] [data-baseweb="tab"][aria-selected="true"] { color: #0f172a !important; }
        /* Borderless inputs: subtle background only (dark) */
        [data-testid="stTextArea"] textarea, [data-testid="stTextArea"] div {
            border: none !important;
            border-radius: 8px !important;
            background-color: #161616 !important;
        }
        /* Unified label + text area block: remove gap, merge visually */
        div[data-testid="stWidgetLabel"] { margin-bottom: 0 !important; padding-bottom: 0 !important; }
        div:has(> [data-testid="stTextArea"]) { gap: 0 !important; }
        .element-container:has([data-testid="stTextArea"]) { gap: 0 !important; }
        .element-container:has([data-testid="stTextArea"]) div[data-testid="stWidgetLabel"] {
            margin-bottom: 0 !important; padding-bottom: 0 !important;
            background-color: #161616 !important;
            border-radius: 8px 8px 0 0 !important;
            padding-top: 0.5rem !important; padding-left: 0.75rem !important; padding-right: 0.75rem !important;
        }
        .element-container:has([data-testid="stTextArea"]) [data-testid="stTextArea"] {
            border-radius: 0 0 8px 8px !important;
            border-top: none !important;
        }
        [data-theme="light"] .element-container:has([data-testid="stTextArea"]) div[data-testid="stWidgetLabel"] {
            background-color: #f4f4f5 !important;
        }
        [data-testid="stFileUploader"], [data-testid="stFileUploader"] section {
            border: none !important;
            border-radius: 8px !important;
            background-color: #161616 !important;
        }
        [data-theme="light"] [data-testid="stTextArea"] textarea,
        [data-theme="light"] [data-testid="stTextArea"] div { background-color: #f4f4f5 !important; }
        [data-theme="light"] [data-testid="stFileUploader"],
        [data-theme="light"] [data-testid="stFileUploader"] section { background-color: #f4f4f5 !important; }
        /* Control panel: no boxes, blend into background */
        [data-testid="stExpander"] { border: none !important; background: transparent !important; }
        [data-testid="stTextInput"] input { border: none !important; background-color: #161616 !important; border-radius: 6px !important; }
        [data-theme="light"] [data-testid="stTextInput"] input { background-color: #f4f4f5 !important; }
        [data-testid="stMetric"], section[data-testid="stExpander"] {
            border: 1px solid #2A2A2A !important;
            border-radius: 8px !important;
            background-color: #111111 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state: initialize media transcript at top so it persists across reruns
if "media_transcript" not in st.session_state:
    st.session_state.media_transcript = ""

st.title("InsightStream — AI User Research Analyst")
st.caption("Paste text, upload a document, or upload media to extract structured feature requests.")

# ---------------------------------------------------------------------------
# Split-pane layout: main workspace (7) | control panel (3)
# ---------------------------------------------------------------------------

cookie_manager = stx.CookieManager()
_saved = cookie_manager.get(cookie="insightstream_api_key")
saved_api_key = (_saved or os.getenv("GEMINI_API_KEY", "")) if _saved else os.getenv("GEMINI_API_KEY", "")

main_workspace, control_panel = st.columns([7, 3], gap="large")

# ---------------------------------------------------------------------------
# Control Panel (right column): model and API key, no expander
# ---------------------------------------------------------------------------

with control_panel:
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = saved_api_key or ""
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Leave blank to use GEMINI_API_KEY from environment.",
        key="api_key_input",
    )
    model = st.selectbox(
        "Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="Model used by the Extractor node.",
        key="model_select",
    )

# ---------------------------------------------------------------------------
# Main Workspace (left column): tabs and input logic
# ---------------------------------------------------------------------------

with main_workspace:
    tab_paste, tab_doc, tab_media = st.tabs(["Paste Text", "Upload Document", "Upload Media"])

    pasted_text = ""
    media_file = None

    with tab_paste:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        pasted_text = st.text_area(
            "Paste transcript",
            height=200,
            placeholder="Paste your customer interview transcript here…",
            help="Direct text input for analysis.",
            key="paste_area",
        )
        st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
        pasted_text = (pasted_text or "").strip()

    with tab_doc:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload transcript (.txt)",
            type=["txt"],
            help="Raw customer interview transcript as plain text. Multiple files = one analysis per file.",
            key="doc_uploader",
            accept_multiple_files=True,
        )
        st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    with tab_media:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        media_file = st.file_uploader(
            "Upload audio or video",
            type=["mp4", "mp3", "wav"],
            help="Supported: .mp4, .mp3, .wav. Click Transcribe to convert to text.",
            key="media_uploader",
        )
        if media_file is not None:
            if st.button("Transcribe", type="primary", key="transcribe_btn"):
                with st.spinner("Transcribing audio…"):
                    try:
                        transcript_text = transcribe_media(media_file)
                        st.session_state.media_transcript = transcript_text
                        st.success("Transcription complete. Run **Analyze transcript** below to get insights.")
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        st.session_state.media_transcript = ""
            st.text_area(
                "Transcription",
                value=st.session_state.media_transcript,
                height=150,
                key="media_transcript_display",
                disabled=True,
            )
        # Do not clear media_transcript when media_file is None (e.g. on rerun); it would wipe the saved transcript
        st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    # Standardized input list: one item per paste, per uploaded file, or media (no concatenation)
    transcripts_to_process = []
    if pasted_text:
        transcripts_to_process.append({"name": "Pasted_Transcript", "content": clean_transcript(pasted_text)})
    if uploaded_files:
        for file in uploaded_files:
            try:
                raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
                try:
                    decoded = raw.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        decoded = raw.decode("cp1252")
                    except (UnicodeDecodeError, LookupError):
                        decoded = raw.decode("latin-1")
                transcripts_to_process.append({"name": file.name, "content": clean_transcript(decoded)})
            except Exception as e:
                st.error(f"Failed to read file {file.name}: {e}")
    # Explicit variable passing: Analyze uses text from session state, not a local variable
    media_transcript = st.session_state.get("media_transcript", "")
    if media_transcript and media_transcript.strip():
        transcripts_to_process.append({"name": "Media_Transcript", "content": clean_transcript(media_transcript)})

    # ---------------------------------------------------------------------------
    # Analysis block: run LangGraph per item when transcripts_to_process is populated
    # ---------------------------------------------------------------------------

    if transcripts_to_process:
        st.subheader("Transcript preview")
        with st.expander("Show transcript", expanded=False):
            for item in transcripts_to_process:
                st.markdown(f"**{item['name']}**")
                preview = item["content"][:800] + ("..." if len(item["content"]) > 800 else "")
                st.text(preview)
                st.markdown("---")

        if st.button("Analyze transcript", type="primary"):
            results = []
            for item in transcripts_to_process:
                with st.spinner(f"Processing {item['name']}…"):
                    try:
                        feature_requests, pain_points, core_themes, error = run_agent(
                            transcript=item["content"],
                            api_key=api_key.strip() or None,
                            model=model,
                        )
                    except Exception as e:
                        st.error(f"Gemini agent error ({item['name']}): {e}")
                        feature_requests = []
                        pain_points = []
                        core_themes = []
                        error = str(e)
                results.append({
                    "name": item["name"],
                    "feature_requests": feature_requests,
                    "pain_points": pain_points,
                    "core_themes": core_themes,
                    "error": error or "",
                })
            st.session_state["analysis_results"] = results
            st.toast("Analysis successfully completed!")

        if st.session_state.get("analysis_results"):
            for res in st.session_state["analysis_results"]:
                name = res["name"]
                safe = _safe_key(name)
                feature_requests = res["feature_requests"]
                pain_points = res["pain_points"]
                core_themes = res["core_themes"]
                error = res["error"]

                if error:
                    st.error(error)

                st.markdown(f"### 📄 Insights from: {name}")
                st.divider()

                if core_themes:
                    st.subheader("🌟 Synthesized Core Themes")
                    for t in core_themes:
                        with st.container():
                            st.markdown(f"**{t.get('theme_name', '—')}** · Strategic importance: **{t.get('strategic_importance', '—')}/10**")
                            st.markdown(t.get("description", "—"))
                            st.markdown("---")
                    st.divider()

                st.subheader("Identified Pain Points")
                if pain_points:
                    num_cards = len(pain_points)
                    cols = st.columns(2 if num_cards >= 2 else 1)
                    for idx, pp in enumerate(pain_points):
                        col = cols[idx % 2]
                        with col:
                            with st.container():
                                st.markdown("---")
                                st.markdown(f"**Pain point {idx + 1}**")
                                confidence = pp.get("confidence_score", 0)
                                st.metric("Confidence Score", f"{confidence}/10", delta=None)
                                st.markdown(f"**Issue:** {pp.get('issue_description', '—')}")
                                st.markdown(f"**Business impact:** {pp.get('business_impact', '—')}")
                                st.markdown(f"**Supporting quote:** _{pp.get('supporting_quote', '—')}_")
                                st.checkbox("Approve for Backlog", value=True, key=f"pp_{safe}_{idx}")
                else:
                    st.caption("No pain points identified.")
                st.divider()

                st.subheader("Feature Requests")
                if feature_requests:
                    num_cards = len(feature_requests)
                    cols = st.columns(2 if num_cards >= 2 else 1)
                    for idx, req in enumerate(feature_requests):
                        col = cols[idx % 2]
                        with col:
                            with st.container():
                                st.markdown("---")
                                st.markdown(f"**Feature request {idx + 1}**")
                                confidence = req.get("confidence_score", 0)
                                st.metric("Confidence Score", f"{confidence}/10", delta=None)
                                st.markdown(f"**User need:** {req.get('user_need', '—')}")
                                st.markdown(f"**Proposed solution:** {req.get('proposed_solution', '—')}")
                                st.markdown(f"**Supporting quote:** _{req.get('supporting_quote', '—')}_")
                                st.checkbox("Approve for Backlog", value=True, key=f"fr_{safe}_{idx}")
                else:
                    st.caption("No feature requests identified.")
                st.divider()

                st.subheader("Raw JSON Structure")
                with st.expander(f"Show JSON — {name}", expanded=False):
                    st.json({"feature_requests": feature_requests, "pain_points": pain_points})

                approved_rows = []
                for idx, pp in enumerate(pain_points):
                    if st.session_state.get(f"pp_{safe}_{idx}", True):
                        approved_rows.append({
                            "Category": "Pain Point",
                            "Core Issue/Need": pp.get("issue_description", ""),
                            "Impact/Solution": pp.get("business_impact", ""),
                            "Supporting Quote": pp.get("supporting_quote", ""),
                            "Confidence Score": pp.get("confidence_score", ""),
                        })
                for idx, req in enumerate(feature_requests):
                    if st.session_state.get(f"fr_{safe}_{idx}", True):
                        approved_rows.append({
                            "Category": "Feature Request",
                            "Core Issue/Need": req.get("user_need", ""),
                            "Impact/Solution": req.get("proposed_solution", ""),
                            "Supporting Quote": req.get("supporting_quote", ""),
                            "Confidence Score": req.get("confidence_score", ""),
                        })
                cols_export = ["Category", "Core Issue/Need", "Impact/Solution", "Supporting Quote", "Confidence Score"]
                df_export = pd.DataFrame(approved_rows, columns=cols_export) if approved_rows else pd.DataFrame(columns=cols_export)
                st.download_button(
                    "⬇️ Download Approved Insights (CSV)",
                    data=df_export.to_csv(index=False).encode("utf-8"),
                    file_name=f"Insights_{name}.csv",
                    mime="text/csv",
                    key=f"download_{safe}",
                )
                st.divider()