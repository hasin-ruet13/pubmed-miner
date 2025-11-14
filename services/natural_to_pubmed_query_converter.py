from __future__ import annotations
import os
import requests
import json
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI

load_dotenv()

# ------- Unified NL → PubMed Converter (multi-LLM) ------- #

def natural_query_to_pubmed_query(user_input: str, llm_meta: dict) -> str:
    """
    Convert natural-language text into a PubMed Boolean query.
    Supports Gemini, OpenAI GPT, Claude, Groq, and Custom (OpenAI-compatible) models.

    llm_meta must contain:
        - model_choice (e.g., "Gemini", "GPT-4o", "Claude", "Groq", "Custom")
        - model_name   (the actual model name)
        - api_key      (user's API key)
        - api_url      (only for Custom LLM)
        - extra_headers (custom headers, optional)
        - timeout      (optional)
    """

    model_choice = llm_meta.get("model_choice")
    model_name   = llm_meta.get("model_name")
    api_key      = llm_meta.get("api_key")
    api_url      = llm_meta.get("api_url")
    timeout      = llm_meta.get("timeout", 60)
    extra_headers = llm_meta.get("extra_headers", {})

    if not api_key and "Custom" not in model_choice:
        raise ValueError("No API key provided for LLM.")

    # --- YOUR EXACT ORIGINAL PROMPT ---
    prompt = f"""Your task is to convert the natural-language text into a precise PubMed Boolean
query. You must follow a two-step process:

STEP 1 — Extract structured search concepts:
- ENTITY: A virus or protein term that must appear in the title.
- GENERAL ENTITY: A related biological entity if present.
- MECHANISTIC TERMS: Functional sequence- or protein-related concepts.

STEP 2 — Build the PubMed Boolean query:
- ENTITY must use the [Title] tag.
- GENERAL ENTITY, if present, must be untagged.
- All MECHANISTIC TERMS must end with [Text Word].
- Final structure:
  ((ENTITY [Title]) AND (GENERAL ENTITY)) AND ((TERM1[Text Word]) OR (TERM2[Text Word]) ...)

Output ONLY the PubMed query, nothing else.

NATURAL LANGUAGE INPUT:
{user_input}
"""

    # ---------------------------------------------------------------------
    # 1. GEMINI
    # ---------------------------------------------------------------------
    if "Gemini" in model_choice:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return resp.text.strip()

    # ---------------------------------------------------------------------
    # 2. OPENAI GPT-4o, GPT-4o-mini, etc.
    # ---------------------------------------------------------------------
    if "GPT" in model_choice or "OpenAI" in model_choice:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()

    # ---------------------------------------------------------------------
    # 3. CLAUDE (Anthropic)
    # ---------------------------------------------------------------------
    if "Claude" in model_choice or "Anthropic" in model_choice:
        headers = {
            "x-api-key": api_key,
            "content-type": "application/json"
        }
        body = {
            "model": model_name,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=timeout
        )
        data = resp.json()
        return data["content"][0]["text"].strip()

    # ---------------------------------------------------------------------
    # 4. GROQ / LLAMA
    # ---------------------------------------------------------------------
    if "Groq" in model_choice or "Llama" in model_choice:
        resp = requests.post(
            "https://api.groq.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            },
            timeout=timeout
        )
        out = resp.json()
        return out["choices"][0]["message"]["content"].strip()

    # ---------------------------------------------------------------------
    # 5. CUSTOM (OpenAI-compatible REST)
    # ---------------------------------------------------------------------
    if "Custom" in model_choice:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        headers.update(extra_headers)

        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }

        resp = requests.post(api_url, headers=headers, json=body, timeout=timeout)
        out = resp.json()
        return out["choices"][0]["message"]["content"].strip()

    raise ValueError(f"Unsupported model choice: {model_choice}")
