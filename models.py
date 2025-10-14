import os
from typing import List, Dict
from openai import OpenAI

SYSTEM = (
    "You are AgroQA, a farm management assistant. Use ONLY the provided context unless common sense is trivial. "
    "Always include bracketed numeric citations like [1], [2] that map to the provided sources."
)

def build_answer_prompt(question: str, docs: List[Dict], mode: str = "short") -> list:
    context_blocks = []
    for i, d in enumerate(docs):
        src = d["meta"].get("source", "unknown")
        page = d["meta"].get("page", "?")
        context_blocks.append(f"[{i+1}] source={src} page={page}\n{d['text']}")
    context = "\n\n".join(context_blocks)

    style = (
        "Provide a concise 3-5 sentence answer with citations like [1], [2]."
        if mode == "short"
        else "Provide a detailed, practical answer with clear steps and numbered citations."
    )

    user_content = (
        f"CONTEXT (authoritative excerpts):\n{context}\n\n"
        f"QUESTION: {question}\n"
        f"STYLE: {style}\n"
        f"REQUIRED: Cite sources using [#] that match the CONTEXT blocks. Add actual quotes to output when possible along with the citations.\n"
        f"CONSTRAINTS: Respond in plain text only. Do NOT include code, code fences, pseudocode, or any mention of charts/plots/graphs/figures, "
        f"or plotting libraries (matplotlib/plotly/seaborn), and do not say things like 'see the chart below' or 'here's a plot'."
    )

    '''
    NEXT STEPS: 
    - add actual quotes to output when possible along with the citations x
    - stream output
    - create second prompt to give code for graphs only if applicable x
    - create third prompt for explanation of graphs if applicable
    '''

    return [
        {"role": "system", "content": SYSTEM + " Respond in plain text only. Never include code or references to charts/plots/graphs."},
        {"role": "user", "content": user_content},
    ]

def build_graph_prompt(question: str, docs: List[Dict], answer: str) -> list:
    context_blocks = []
    for i, d in enumerate(docs):
        src = d["meta"].get("source", "unknown")
        page = d["meta"].get("page", "?")
        context_blocks.append(f"[{i+1}] source={src} page={page}\n{d['text']}")
    context = "\n\n".join(context_blocks)

    user_content = (
        f"CONTEXT (authoritative excerpts):\n{context}\n\n"
        f"ORIGINAL QUESTION: {question}\n"
        f"ORIGINAL ANSWER: {answer}\n"
        "TASK: If (and only if) a simple graph helps communicate the answer using the CONTEXT, "
        "output valid Python code for ONE matplotlib chart (no imports, no plt.show()). "
        "If not applicable, output exactly 'N/A'. Output ONLY code or 'N/A'. "
        "However, if the ORIGINAL QUESTION contains 'chart', 'plot', 'graph', or 'matplotlib', you MUST output plotting code (do not output 'N/A'). Otherwise, return 'N/A' only when a chart would not help."
    )

    return [
        {"role": "system",
         "content": "You only output Python code for one matplotlib chart based on the provided context, or 'N/A'. "
                    "Do not include explanations or code fences."},
        {"role": "user", "content": user_content},
    ]

def answer(question: str, docs: List[Dict], mode: str = "short") -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    messages = build_answer_prompt(question, docs, mode)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    
    graph = build_graph_prompt(question, docs, answer)
    resp_graph = client.chat.completions.create(
        model=model,
        messages=graph,
        temperature=0.2,
    )
    answer_graph = resp_graph.choices[0].message.content.strip()
    if answer_graph.startswith("```"):
        answer_graph = answer_graph.strip("`")
        if answer_graph.startswith("python"):
            answer_graph = answer_graph[len("python"):].lstrip()

    return answer, answer_graph