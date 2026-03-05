# Product: InsightStream (AI User Research Analyst)

## Goal
Build a local, interactive web application that allows a Product Manager to upload a raw transcript of a customer interview. The AI agent will analyze the text, extract core user pain points, and output structured, actionable feature requests.

## Architecture & Flow
1. User uploads a `.txt` file containing a customer interview transcript via the Streamlit UI.
2. The LangGraph agent is triggered. It contains the following nodes:
   - Node 1 (Extractor): Reads the transcript and identifies feature requests.
   - Node 2 (Formatter): Takes the extracted data and formats it into a strict JSON schema.
3. The UI displays the final structured user stories.

## Output Schema (Pydantic / JSON)
Every extracted feature request must contain:
- "user_need": A 1-sentence summary of the problem.
- "proposed_solution": The actionable feature.
- "supporting_quote": The exact text extracted from the raw transcript.
- "confidence_score": An integer (1-10) rating how explicitly the user requested this.