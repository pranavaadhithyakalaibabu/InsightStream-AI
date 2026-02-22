# InsightStream-AI
Autonomous AI User Research Analyst using LangGraph and Gemini 2.5 Flash.

graph LR
    A[Raw Transcript] --> B(Sanitization Node)
    B -->|Deterministic: Regex| C{Extractor Node}
    C -->|Probabilistic: Gemini 2.5| D[Atomic Insights]
    D --> E(Auditor Node)
    E -->|Deterministic: Quote Match| F{Synthesizer Node}
    F -->|Probabilistic: Clustering| G[Core Themes]
    G --> H[Verification Center UI]
