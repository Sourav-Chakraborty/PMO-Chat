LLMQueryPrompt="""
You are a helpful AI assistant.

You will receive:
1. CONTEXT retrieved from a vector database (may be empty or irrelevant)
2. A USER QUESTION

Rules:
- If the CONTEXT is relevant, use it.
- If the CONTEXT is empty, irrelevant, or unhelpful, IGNORE it completely.
- You are allowed to answer using your general world knowledge.
- Do NOT say "Based on the context" if you are not using it.

CONTEXT:
{context}

QUESTION:
{question}

"""

documentSummeryPrompt="""
You are given a JSON-like structure that contains multiple PDFs, where each PDF contains an array called pages, and each page contains extracted text, table data, or images.

Your task is to analyze the entire content of all PDFs and extract crisp, standalone, atomic information points suitable for embedding into a vector database for a PMO (Project Management Office) knowledge base chatbot.

Your Output Requirements:

- Produce concise, self-contained knowledge chunks.
- Each point must stand alone without depending on previous text.
- Avoid repetition; merge similar points.
- No long paragraphs—each output should be a clean, unambiguous statement.
- Focus on factual, policy, rule, guideline, and responsibility-based information.
- Ignore decorative text, repeated headers, and formatting noise.
- Do not hallucinate. Base output strictly on provided text.
- Ensure maximum usefulness for semantic retrieval.

### Output Format (JSON ARRAY):
[
  {{
    "title": "<short title>",
    "summary": "<crisp standalone explanation>",
    "category": "<optional section or grouping>"
  }}
]

### Input Structure Example:
{{
  "extracted_results": [
    {{
      "filename": "...",
      "extracted": {{
        "pages": [
          {{
            "page_number": 1,
            "text": "...",
            "tables": [...],
            "images": [...]
          }}
        ]
      }}
    }}
  ]
}}

### Input Data:
{context}

Return ONLY a valid JSON array.
"""

