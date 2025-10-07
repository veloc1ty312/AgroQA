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
        "Provide a concise 3–5 sentence answer with citations like [1], [2]."
        if mode == "short"
        else "Provide a detailed, practical answer with clear steps and numbered citations."
    )

    user_content = (
        f"CONTEXT (authoritative excerpts):\n{context}\n\n"
        f"QUESTION: {question}\n"
        f"STYLE: {style}\n"
        f"REQUIRED: Cite sources using [#] that match the CONTEXT blocks. Add actual quotes to output when possible along with the citations."
    )

    '''
    NEXT STEPS: 
    - add actual quotes to output when possible along with the citations x
    - stream output
    - create second prompt to give code for graphs only if applicable
    '''

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content},
    ]

def build_graph_prompt(question: str, docs: List[Dict]) -> list:
    context_blocks = []
    for i, d in enumerate(docs):
        src = d["meta"].get("source", "unknown")
        page = d["meta"].get("page", "?")
        context_blocks.append(f"[{i+1}] source={src} page={page}\n{d['text']}")
    context = "\n\n".join(context_blocks)

    user_content = (
        f"CONTEXT (authoritative excerpts):\n{context}\n\n"
        f"QUESTION: Only if applicable, create python code for graphs using real data (ONLY output code). If not applicable, output 'N/A'\n"
    )

    return [
        {"role": "system", "content": SYSTEM},
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
    graph = build_graph_prompt(question, docs)
    resp_graph = client.chat.completions.create(
        model=model,
        messages=graph,
        temperature=0.2,
    )
    return resp.choices[0].message.content, resp_graph.choices[0].message.content