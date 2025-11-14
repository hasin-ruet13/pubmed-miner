from __future__ import annotations

import os, json, io, zipfile
from datetime import date
import calendar
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from typing import Dict

from services.pmc import get_last_fetch_source
from services.pubmed import (
    esearch_reviews, esearch_all, esummary, parse_pubdate_interval, overlaps
)
from pipeline.batch_analyze import fetch_all_fulltexts, analyze_texts
from pipeline.csv_export import flatten_to_rows
from services.natural_to_pubmed_query_converter import natural_query_to_pubmed_query
# Import prompts for editing
from llm.prompts import PROMPTS
from llm.validation_questions import VALIDATION_QUESTIONS


def _persist(key, value):
    st.session_state[key] = value
    return value


def _bucketize_papers(papers_dict):
    fetched, no_pmc, errors = [], [], []
    for pmid, info in papers_dict.items():
        row = {
            "PMID": str(pmid),
            "PMCID": info.get("pmcid") or "",
            "Title": info.get("title") or "",
            "Status": info.get("status"),
            "Error": info.get("error") or "",
        }
        if info.get("status") == "ok":
            fetched.append(row)
        elif info.get("status") == "no_pmc_fulltext":
            no_pmc.append(row)
        else:
            errors.append(row)
    return fetched, no_pmc, errors


def main():
    load_dotenv()
    st.set_page_config(page_title="PubMed → PMC → LLM (Batch Miner)", layout="wide")
    st.title("🧪 PubMed Miner")
    st.caption("Search PubMed articles, fetch PMC full text, run your LLM extractor, and download findings.")

    # ===== Sidebar Configuration =====
    with st.sidebar:
        # NCBI Configuration (needed for search)
        st.header("🔬 NCBI Configuration")
        st.caption("Required for PubMed search")
        
        ncbi_api_key = st.text_input(
            "NCBI API Key",
            value=os.getenv("NCBI_API_KEY", ""),
            type="password",
            help="Get from: https://www.ncbi.nlm.nih.gov/account/settings/"
        )
        
        # Strip whitespace and update environment
        if ncbi_api_key:
            ncbi_api_key = ncbi_api_key.strip()
            os.environ["NCBI_API_KEY"] = ncbi_api_key
            st.success("✅ NCBI API Key set")
        elif os.getenv("NCBI_API_KEY"):
            st.info("ℹ️ Using NCBI API Key from .env file")
        else:
            st.warning("⚠️ No API Key (limited to 3 requests/second)")
        
        # Test connection
        with st.expander("🔧 Test NCBI Connection", expanded=False):
            if st.button("Test Quick Search", key="test_ncbi"):
                try:
                    with st.spinner("Testing NCBI connection..."):
                        test_pmids = esearch_reviews("covid-19", mindate="2020/01/01", maxdate="2020/12/31", sort="relevance", cap=5)
                        # Note: Test always uses reviews for consistency
                        if test_pmids:
                            st.success(f"✅ NCBI working! Found {len(test_pmids)} test results")
                        else:
                            st.warning("⚠️ NCBI returned 0 results (may be normal for this query)")
                except Exception as e:
                    st.error(f"❌ NCBI connection failed:\n```\n{str(e)}\n```")
        
        st.divider()
        
        # LLM Configuration
        st.header("🤖 LLM Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "Select LLM Model",
            [
                "Gemini (Google)",
                "GPT-4o (OpenAI)",
                "Claude (Anthropic)",
                "Llama (Groq)",
                "Custom (Hackathon)",
            ],
            index=0,
            help="Choose which LLM to use for extraction"
        )
        
        # API Key input based on selection
        api_key_env_var = None
        api_key = ""
        model_name = ""
        custom_api_url = os.getenv("CUSTOM_LLM_URL", "")
        custom_headers_json = os.getenv("CUSTOM_LLM_HEADERS", "")
        custom_headers_dict: Dict[str, str] = {}
        custom_timeout = int(os.getenv("CUSTOM_LLM_TIMEOUT", "120") or 120)
        if "Gemini" in model_choice:
            api_key = st.text_input(
                "Gemini API Key",
                value=os.getenv("GEMINI_API_KEY", ""),
                type="password",
                help="Get from: https://ai.google.dev/"
            )
            api_key_env_var = "GEMINI_API_KEY"
            model_name = st.text_input("Model Name", value="gemini-2.5-flash-lite")
            
        elif "GPT-4o" in model_choice:
            api_key = st.text_input(
                "OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
                help="Get from: https://platform.openai.com/api-keys"
            )
            api_key_env_var = "OPENAI_API_KEY"
            model_name = st.text_input("Model Name", value="gpt-4o-2024-11-20")
            
        elif "Claude" in model_choice:
            api_key = st.text_input(
                "Anthropic API Key",
                value=os.getenv("ANTHROPIC_API_KEY", ""),
                type="password",
                help="Get from: https://console.anthropic.com/"
            )
            api_key_env_var = "ANTHROPIC_API_KEY"
            model_name = st.text_input("Model Name", value="claude-sonnet-4-20250514")
            
        elif "Groq" in model_choice or "Llama" in model_choice:  # Llama (Groq)
            api_key = st.text_input(
                "Groq API Key",
                value=os.getenv("GROQ_API_KEY", ""),
                type="password",
                help="Get from: https://console.groq.com/keys"
            )
            api_key_env_var = "GROQ_API_KEY"
            model_name = st.text_input("Model Name", value="llama-3.3-70b-versatile")
        
        elif "Custom" in model_choice:
            st.info("🎯 **Hackathon Models**: Select from the available models provided through the local proxy.")
            
            # Hackathon models from the capability matrix
            HACKATHON_MODELS = [
                "gpt35",
                "gpt35large",
                "gpt4",
                "gpt4large",
                "gpt4turbo",
                "gpt4o",
                "gpto1",
                "gpto1mini",
                "gpto3",
                "gpto3mini",
                "gpto4mini",
                "gpt41",
                "gpt41mini",
                "gpt41nano",
                "gpt5",
                "gpt5mini",
                "gpt5nano",
                "gemini25pro",
                "gemini25flash",
                "claudeopus4",
                "claudeopus41",
                "claudesonnet4",
                "claudesonnet45",
                "claudesonnet37",
                "claudesonnet35v2",
            ]
            
            # Model selector with auto-populate
            selected_model = st.selectbox(
                "Select Hackathon Model",
                options=[""] + HACKATHON_MODELS,
                index=0,
                help="Choose from the 25 models available through the local proxy. All models support text extraction."
            )
            
            # Auto-populate model name if selected, otherwise allow manual entry
            if selected_model:
                model_name = selected_model
                os.environ["CUSTOM_LLM_MODEL"] = model_name
            else:
                model_name = st.text_input(
                    "Model Name (or select from dropdown above)",
                    value=os.getenv("CUSTOM_LLM_MODEL", ""),
                    help="Enter model name manually if not in the dropdown"
                )
            
            custom_api_url = st.text_input(
                "Local Proxy API URL",
                value=custom_api_url,
                help="The proxy endpoint URL provided by hackathon organizers (e.g., http://localhost:8080/v1/completions or https://proxy.hackathon.local/v1/completions)"
            ).strip()
            if custom_api_url:
                os.environ["CUSTOM_LLM_URL"] = custom_api_url
            
            api_key = st.text_input(
                "API Key (optional)",
                value=os.getenv("CUSTOM_LLM_API_KEY", ""),
                type="password",
                help="Leave blank if the proxy does not require authentication."
            )
            api_key_env_var = "CUSTOM_LLM_API_KEY"
            custom_headers_json = st.text_area(
                "Extra HTTP headers (JSON)",
                value=custom_headers_json,
                help='Optional: JSON object of additional headers. Example: {"X-Org": "Team-42"}'
            )
            parsed_headers: Dict[str, str] = {}
            if custom_headers_json.strip():
                try:
                    parsed_headers = json.loads(custom_headers_json.strip())
                    if not isinstance(parsed_headers, dict):
                        raise ValueError("Headers JSON must be an object")
                except Exception:
                    st.error("Extra HTTP headers must be a valid JSON object of key-value pairs.")
                    parsed_headers = {}
            custom_headers_dict = parsed_headers
            custom_timeout = st.number_input(
                "Request timeout (seconds)",
                min_value=5,
                max_value=600,
                value=custom_timeout,
                help="Adjust if the endpoint is slow."
            )
            os.environ["CUSTOM_LLM_TIMEOUT"] = str(custom_timeout)
            if custom_headers_dict:
                os.environ["CUSTOM_LLM_HEADERS"] = json.dumps(custom_headers_dict)
        
        # Strip whitespace from API key and persist in environment for backend usage
        if api_key:
            api_key = api_key.strip()
            if api_key_env_var:
                os.environ[api_key_env_var] = api_key
        
        st.divider()
        
        # Extraction parameters
        st.header("⚙️ Extraction Settings")
        chunk_chars = st.slider("Max chars per chunk", 8000, 24000, 16000, 1000)
        overlap_chars = st.slider("Overlap per chunk", 200, 1500, 500, 50)
        delay_ms = st.slider("Delay between chunk calls (ms)", 0, 1500, 400, 50)
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.6, 0.05)
        
        st.divider()
        
        # Validation settings
        st.header("✅ Validation Settings")
        enable_validation = st.checkbox(
            "Enable Agentic Validation",
            value=False,
            help="Validate all extractions by asking multiple questions per extraction. Uses the same LLM backend."
        )
        
        if enable_validation:
            validation_threshold = st.slider(
                "Validation Accuracy Threshold (%)",
                50, 100, 85, 5,
                help="Minimum accuracy percentage to pass validation (default: 85%)"
            )
            validation_delay_ms = st.slider(
                "Validation Delay (ms)",
                0, 1000, 200, 50,
                help="Delay between validation calls to avoid rate limits"
            )
            st.info("ℹ️ Validation will ask 2-3 questions per extraction and check if accuracy ≥ threshold.")
        else:
            validation_threshold = 85.0
            validation_delay_ms = 200
        
        st.divider()
        st.caption("💡 Tip: Test with 1-2 papers first to verify API keys work")

    # Build LLM metadata for query converter (simplified version)
    llm_meta_query = {
        "model_choice": model_choice,
        "model_name": model_name,
        "api_key": api_key,
    }
    if "Custom" in model_choice:
        llm_meta_query["api_url"] = custom_api_url
        llm_meta_query["timeout"] = custom_timeout
        if custom_headers_dict:
            llm_meta_query["extra_headers"] = custom_headers_dict

    # ===== Prompt Editor Section =====
    with st.expander("📝 **Edit Extraction Prompt**", expanded=False):
        st.markdown("""
        ### Quick Start
        
        1. **Edit the prompt sections** below to customize what features are extracted
        2. **Modify PATTERN RECOGNITION GUIDE** to add/modify pattern descriptions (mutations, proteins, domains, etc.)
        3. **Modify INSTRUCTIONS** to change extraction priorities and behaviors
        4. Click **"💾 Save Changes"** and test on a sample paper
        
        ---
        
        ### What You Can Edit
        
        **PATTERN RECOGNITION GUIDE** (in editable section):
        - **Mutation Patterns**: Add mutation formats you encounter (e.g., `A226V`, `p.Ala226Val`, spelled mutations, arrow notation)
        - **Protein Patterns**: Modify how proteins are identified (names, abbreviations, complexes)
        - **Residue Number Patterns**: Adjust residue range detection patterns
        - **Amino Acid Position Patterns**: Customize position notation recognition
        - **Structural Domain Patterns**: Add domain/region pattern descriptions
        - **Motif Patterns**: Modify motif detection patterns
        - **Coverage Strategy**: Adjust how the LLM scans and extracts patterns
        
        **INSTRUCTIONS** (in editable section):
        - Extraction priorities and coverage requirements
        - Filtering criteria and quality thresholds
        - Mutation format conversion rules
        - Domain extraction requirements
        
        **SYSTEM / INSTRUCTION** (in editable section):
        - AI extractor role and identity
        - Specialization for different viruses or domains
        
        **DEFINITIONS** (in editable section):
        - Feature types and their definitions
        - Add new feature types (e.g., "• **NEW_FEATURE** (description)")
        
        **Locked** (for safety - cannot edit):
        - JSON schema and output format (ensures app works correctly)
        - Output rules (prevents breaking changes)
        - Few-shot examples (maintains consistency)
        - The `{TEXT}` placeholder (required for paper content)
        """)
        
        # Load current editable section
        current_editable = PROMPTS.analyst_prompt_editable
        
        # Single editor for editable section
        edited_section = st.text_area(
            "Editable Prompt Section",
            value=current_editable,
            height=500,
            help="This section contains: SYSTEM/INSTRUCTION, DEFINITIONS, PATTERN RECOGNITION GUIDE, and INSTRUCTIONS. Edit these to customize extraction behavior. JSON schema, output rules, and examples are locked for safety."
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("💾 Save Changes", key="save_prompt_changes"):
                PROMPTS.analyst_prompt_editable = edited_section
                # Clear any override to use the new editable section
                PROMPTS._analyst_prompt_override = ""
                st.success("✅ Prompt updated! Will be used for next extraction.")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Default", key="reset_prompt_default"):
                from llm.prompts import AnalystPrompts
                default_prompts = AnalystPrompts()
                PROMPTS.analyst_prompt_editable_part1 = default_prompts.analyst_prompt_editable_part1
                PROMPTS.analyst_prompt_editable_part2 = default_prompts.analyst_prompt_editable_part2
                PROMPTS._analyst_prompt_override = ""
                st.success("✅ Reset to default prompt.")
                st.rerun()
        
        # Optional: Show preview of full prompt (collapsed by default)
        with st.expander("👁️ Preview Full Prompt (Read-only)", expanded=False):
            st.markdown("**Full assembled prompt that will be sent to the LLM:**")
            full_preview = PROMPTS.analyst_prompt
            st.code(full_preview, language="text", line_numbers=False)
            st.caption("💡 This is what the LLM receives. The editable section is embedded in the middle.")

    # ===== Validation Question Editor Section =====
    with st.expander("✅ **Edit Validation Questions**", expanded=False):
        st.markdown("""
        ### Customize Validation Questions
        
        Enter 1-5 validation questions (one per line). These questions will be asked for each extraction.
        
        **Placeholders you can use:**
        - `{mutation}` - The mutation name (e.g., "A226V")
        - `{feature_name}` - The feature name (e.g., "RNA-binding domain")
        - `{protein}` - The protein name (e.g., "E1", "nsP3")
        - `{virus}` - The virus name (e.g., "Chikungunya virus")
        - `{position}` - Single position (e.g., "226")
        - `{residue_range}` - Residue range (e.g., "1-73")
        - `{effect_description}` - Effect/function description
        
        **Example:**
        ```
        Is there experimental evidence for a sequence feature in {protein} protein of {virus}?
        Are the residue positions correctly identified for this feature in {protein}?
        Is the biological function correctly described for this feature in {protein}?
        ```
        """)
        
        # Get current questions as a single string (one per line)
        current_questions = "\n".join(VALIDATION_QUESTIONS.questions)
        
        questions_text = st.text_area(
            "Validation Questions (1-5 questions, one per line)",
            value=current_questions,
            height=200,
            help="Enter 1-5 questions, one per line. Each question will be asked for every extraction.",
            key="val_questions_text"
        )
        
        # Count questions
        question_lines = [q.strip() for q in questions_text.split('\n') if q.strip()]
        num_questions = len(question_lines)
        
        if num_questions == 0:
            st.warning("⚠️ Please provide at least 1 question.")
        elif num_questions > 5:
            st.warning(f"⚠️ You have {num_questions} questions. Only the first 5 will be used.")
        else:
            st.info(f"✅ {num_questions} question(s) configured.")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("💾 Save Questions", key="save_val_questions"):
                VALIDATION_QUESTIONS.questions = questions_text
                st.success("✅ Validation questions updated!")
        
        with col2:
            if st.button("🔄 Reset to Default", key="reset_val_questions"):
                VALIDATION_QUESTIONS.reset_to_default()
                st.success("✅ Reset to default validation questions.")
                st.rerun()

    # ===== Search Section =====
    st.subheader("1) Enter your query")
    
    # Show NCBI API status in main area
    if not os.getenv("NCBI_API_KEY"):
        st.info("💡 **Tip:** Add your NCBI API key in the sidebar (👈) to increase rate limits from 3 to 10 requests/second.")
    
    # Query type selection
    query_type = st.radio(
        "Query Type",
        ["Natural Language Query", "PubMed Query"],
        index=0,
        horizontal=True,
        help="Choose 'Natural Language Query' to enter plain English, or 'PubMed Query' to enter a PubMed Boolean query directly."
    )
    
    # Toggle for review papers only
    reviews_only = st.checkbox(
        "🔍 Restrict to Review articles only",
        value=True,
        help="If checked, only search for Review articles. If unchecked, search all article types."
    )
    
    # Set placeholder based on query type
    if query_type == "Natural Language Query":
        placeholder = 'e.g., dengue virus mutations in the E protein'
        help_text = "Enter your search in plain English. The system will automatically convert it to a PubMed Boolean query. Examples: 'COVID-19 spike protein variants', 'influenza hemagglutinin mutations', 'dengue virus E protein structure'"
    else:
        placeholder = 'e.g., (dengue[MeSH Terms] OR dengue virus[Title]) AND (mutation[Text Word] OR variant[Text Word])'
        help_text = "Enter a PubMed Boolean query with field tags. Use AND, OR, NOT operators. Field tags: [Title], [MeSH Terms], [Text Word], [Abstract], etc."

    query = st.text_area(
        "Query", 
        height=100, 
        placeholder=placeholder,
        help=help_text
    )
    
    st.subheader("2) Choose publication date range & search")
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        mindate = st.text_input(
            "Start Date (MM/YYYY)", 
            value="01/2005",
            placeholder="MM/YYYY",
            help="Enter start date in MM/YYYY format (e.g., 01/2020)"
        )
    with colB:
        maxdate = st.text_input(
            "End Date (MM/YYYY)", 
            value="12/2025",
            placeholder="MM/YYYY",
            help="Enter end date in MM/YYYY format (e.g., 12/2025)"
        )
    with colC:
        sort = st.selectbox("Sort", ["relevance", "pub+date"], index=0)
    with colD:
        cap = st.slider("Max records", 0, 500, 100, 100)

    search_button_text = "🔎 Search PubMed (reviews)" if reviews_only else "🔎 Search PubMed (all articles)"
    go = st.button(search_button_text)

    # ===== Handle Search Button Click =====
    if go:
        # Basic query check
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            # Validate and convert MM/YYYY to YYYY/MM format for NCBI
            try:
                # Parse MM/YYYY input
                min_parts = mindate.strip().split("/")
                max_parts = maxdate.strip().split("/")
                
                if len(min_parts) != 2 or len(max_parts) != 2:
                    st.error("❌ Invalid date format. Please use MM/YYYY (e.g., 01/2020)")
                    st.stop()
                
                # Convert MM/YYYY to YYYY/MM for NCBI API
                mindate_formatted = f"{min_parts[1]}/{min_parts[0]}"
                maxdate_formatted = f"{max_parts[1]}/{max_parts[0]}"
                
                # Create date objects for overlap checking
                start_date = date(int(min_parts[1]), int(min_parts[0]), 1)
                # Last day of month for end date
                end_year, end_month = int(max_parts[1]), int(max_parts[0])
                last_day = calendar.monthrange(end_year, end_month)[1]
                end_date = date(end_year, end_month, last_day)
                
            except Exception as e:
                st.error(f"❌ Invalid date format: {e}. Please use MM/YYYY (e.g., 01/2020)")
                st.stop()
            
            raw_query = query.strip()

            # ---- NATURAL LANGUAGE BRANCH: convert but DO NOT search yet ----
            if query_type == "Natural Language Query":
                with st.spinner("🔄 Converting natural-language query to PubMed Boolean query…"):
                    try:
                        converted_query = natural_query_to_pubmed_query(raw_query, llm_meta_query)
                    except Exception as e:
                        st.warning(f"⚠️ Query conversion failed — using raw query. Error: {e}")
                        converted_query = raw_query

                # Store everything needed for later search
                st.session_state["pending_nlq"] = True
                st.session_state["nlq_payload"] = {
                    "converted_query": converted_query,
                    "mindate_formatted": mindate_formatted,
                    "maxdate_formatted": maxdate_formatted,
                    "start_date": start_date,
                    "end_date": end_date,
                    "reviews_only": reviews_only,
                    "sort": sort,
                    "cap": cap,
                }

                st.info("✅ Converted query generated. You can edit it below and run the PubMed search.")

            # ---- PUBMED QUERY BRANCH: run search immediately ----
            else:
                try:
                    search_type = "reviews" if reviews_only else "all articles"
                    with st.spinner(f"Searching PubMed ({search_type})…"):
                        if reviews_only:
                            pmids = esearch_reviews(raw_query, mindate=mindate_formatted, maxdate=maxdate_formatted, sort=sort, cap=cap)
                        else:
                            pmids = esearch_all(raw_query, mindate=mindate_formatted, maxdate=maxdate_formatted, sort=sort, cap=cap)
                        
                        if not pmids:
                            st.warning(
                                f"❌ **No results found for your query.**\n\n"
                                f"**Your query:** `{raw_query}`\n\n"
                                f"**Tips:**\n"
                                f"- Try broader search terms\n"
                                f"- Check spelling\n"
                                f"- Expand date range (currently {start_date} to {end_date})\n"
                                f"- Try a simpler query (e.g., just 'dengue mutation')"
                            )
                            for k in ["hits_df", "hits_pmids", "selected_pmids"]:
                                if k in st.session_state:
                                    del st.session_state[k]
                        else:
                            st.info(f"✅ Found {len(pmids)} PMIDs from NCBI. Fetching metadata...")
                            sums = esummary(pmids)
                            
                            rows = []
                            for pid in pmids:
                                meta = sums.get(pid, {})
                                pubdate_raw = meta.get("pubdate") or ""
                                if overlaps(parse_pubdate_interval(pubdate_raw), (start_date, end_date)):
                                    rows.append({
                                        "PMID": str(pid),
                                        "Title": meta.get("title") or "",
                                        "Journal": meta.get("source") or "",
                                        "PubDate": pubdate_raw,
                                        "PubMed Link": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                                    })
                            
                            df_hits = pd.DataFrame(rows)
                            if df_hits.empty:
                                for k in ["hits_df", "hits_pmids", "selected_pmids"]:
                                    if k in st.session_state:
                                        del st.session_state[k]
                                st.warning(
                                    f"⚠️ **Found {len(pmids)} results, but none match your date range.**\n\n"
                                    f"Date range: {start_date} to {end_date}\n\n"
                                    f"Try expanding the date range above."
                                )
                            else:
                                df_hits["PMID"] = df_hits["PMID"].astype(str)
                                _persist("hits_df", df_hits.to_dict("records"))
                                _persist("hits_pmids", df_hits["PMID"].tolist())
                                st.session_state["selected_pmids"] = []
                                st.success(f"✅ Found {len(df_hits)} results. See 'Results' below to select papers.")
                
                except Exception as e:
                    st.error(
                        f"❌ **Error searching PubMed:**\n\n"
                        f"```\n{str(e)}\n```\n\n"
                        f"**Troubleshooting:**\n"
                        f"1. Check your internet connection\n"
                        f"2. Verify NCBI API key in `.env` file\n"
                        f"3. Try again in a few seconds (rate limiting)\n"
                        f"4. Simplify your query"
                    )

    # ===== NATURAL LANGUAGE CONFIRMATION / EDIT & SEARCH =====
    if st.session_state.get("pending_nlq"):
        payload = st.session_state.get("nlq_payload") or {}
        converted_query = payload.get("converted_query", "")

        st.write("### 📘 Converted PubMed Query (editable)")
        edited_query = st.text_area(
            "Converted PubMed Query (you can edit before running search):",
            value=converted_query,
            height=120,
            key="nlq_query_edit"
        )

        col1, col2 = st.columns(2)
        with col1:
            run_nlq_search = st.button("✔ Use this query and search PubMed", key="btn_run_nlq")
        with col2:
            cancel_nlq = st.button("❌ Cancel", key="btn_cancel_nlq")

        if cancel_nlq:
            st.session_state["pending_nlq"] = False
            st.session_state["nlq_payload"] = None
            st.info("Natural-language query conversion cancelled.")

        elif run_nlq_search:
            # Unpack stored parameters
            raw_query = edited_query
            mindate_formatted = payload["mindate_formatted"]
            maxdate_formatted = payload["maxdate_formatted"]
            start_date = payload["start_date"]
            end_date = payload["end_date"]
            reviews_only = payload["reviews_only"]
            sort = payload["sort"]
            cap = payload["cap"]

            try:
                search_type = "reviews" if reviews_only else "all articles"
                with st.spinner(f"Searching PubMed ({search_type})…"):
                    if reviews_only:
                        pmids = esearch_reviews(raw_query, mindate=mindate_formatted, maxdate=maxdate_formatted, sort=sort, cap=cap)
                    else:
                        pmids = esearch_all(raw_query, mindate=mindate_formatted, maxdate=maxdate_formatted, sort=sort, cap=cap)
                    
                    if not pmids:
                        st.warning(
                            f"❌ **No results found for your query.**\n\n"
                            f"**Your query:** `{raw_query}`\n\n"
                            f"**Tips:**\n"
                            f"- Try broader search terms\n"
                            f"- Check spelling\n"
                            f"- Expand date range (currently {start_date} to {end_date})\n"
                            f"- Try a simpler query (e.g., just 'dengue mutation')"
                        )
                        for k in ["hits_df", "hits_pmids", "selected_pmids"]:
                            if k in st.session_state:
                                del st.session_state[k]
                    else:
                        st.info(f"✅ Found {len(pmids)} PMIDs from NCBI. Fetching metadata...")
                        sums = esummary(pmids)
                        
                        rows = []
                        for pid in pmids:
                            meta = sums.get(pid, {})
                            pubdate_raw = meta.get("pubdate") or ""
                            if overlaps(parse_pubdate_interval(pubdate_raw), (start_date, end_date)):
                                rows.append({
                                    "PMID": str(pid),
                                    "Title": meta.get("title") or "",
                                    "Journal": meta.get("source") or "",
                                    "PubDate": pubdate_raw,
                                    "PubMed Link": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                                })
                        
                        df_hits = pd.DataFrame(rows)
                        if df_hits.empty:
                            for k in ["hits_df", "hits_pmids", "selected_pmids"]:
                                if k in st.session_state:
                                    del st.session_state[k]
                            st.warning(
                                f"⚠️ **Found {len(pmids)} results, but none match your date range.**\n\n"
                                f"Date range: {start_date} to {end_date}\n\n"
                                f"Try expanding the date range above."
                            )
                        else:
                            df_hits["PMID"] = df_hits["PMID"].astype(str)
                            _persist("hits_df", df_hits.to_dict("records"))
                            _persist("hits_pmids", df_hits["PMID"].tolist())
                            st.session_state["selected_pmids"] = []
                            st.success(f"✅ Found {len(df_hits)} results. See 'Results' below to select papers.")
            except Exception as e:
                st.error(
                    f"❌ **Error searching PubMed:**\n\n"
                    f"```\n{str(e)}\n```\n\n"
                    f"**Troubleshooting:**\n"
                    f"1. Check your internet connection\n"
                    f"2. Verify NCBI API key in `.env` file\n"
                    f"3. Try again in a few seconds (rate limiting)\n"
                    f"4. Simplify your query"
                )
            finally:
                # Clear pending NLQ state so we don't keep showing the editor
                st.session_state["pending_nlq"] = False
                st.session_state["nlq_payload"] = None

    # ===== Results display (unchanged) =====
    if st.session_state.get("hits_df"):
        st.markdown("#### Results")
        df_hits = pd.DataFrame(st.session_state["hits_df"])
        pmid_options = [str(x) for x in st.session_state.get("hits_pmids", [])]
        
        st.dataframe(df_hits, width='stretch')
        st.download_button(
            "⬇️ Download matches (.csv)", 
            data=df_hits.to_csv(index=False).encode("utf-8"), 
            file_name="pubmed_review_matches.csv", 
            mime="text/csv"
        )
        
        st.markdown("##### Select PMIDs to process with LLM")
        if "selected_pmids" not in st.session_state:
            st.session_state["selected_pmids"] = []
        
        select_all = st.checkbox("Select all", value=False, key="select_all_hits")
        if select_all:
            st.session_state["selected_pmids"] = pmid_options.copy()
        
        st.session_state["selected_pmids"] = st.multiselect(
            "Choose PMIDs (multi-select supported)", 
            options=pmid_options,
            default=[p for p in st.session_state["selected_pmids"] if p in pmid_options], 
            key="pmid_multiselect"
        )
        
        st.caption(f"Selected: {len(st.session_state['selected_pmids'])} of {len(pmid_options)}")
        
        if st.session_state["selected_pmids"]:
            st.download_button(
                "⬇️ Download selected PMIDs (.txt)", 
                data=("\n".join(st.session_state["selected_pmids"]).encode("utf-8")), 
                file_name="selected_pmids.txt", 
                mime="text/plain"
            )

    # ===== Extraction section (modified to pass model info) =====
    st.subheader("3) Run extraction")
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        override_all = st.checkbox(
            "Process ALL results (ignore selection)", 
            value=False, 
            help="When checked, all results from Step 2 will be sent to the LLM."
        )
    with colB:
        clear_previous = st.checkbox(
            "Clear previous results before run", 
            value=True, 
            help="Prevents older findings from appearing alongside this run."
        )
    with colC:
        run_all = st.button(
            "🚀 Fetch PMC & Run LLM", 
            disabled=(not override_all and len(st.session_state.get("selected_pmids", [])) == 0)
        )

    if st.button("🗑️ Reset", key="reset_all_data"):
        for k in ["hits_df", "hits_pmids", "batch_papers", "batch_results", "llm_log", 
                  "selected_pmids", "select_all_hits", "pmid_multiselect"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    if run_all:
        hits = [str(x) for x in st.session_state.get("hits_pmids", [])]
        selected = [str(x) for x in st.session_state.get("selected_pmids", [])]
        pmids = hits if override_all else selected
        
        if not pmids:
            st.warning("No PMIDs selected. Pick at least one in Step 2 or check 'Process ALL results'.")
            st.stop()
        
        if clear_previous:
            st.session_state["batch_results"] = {}
            st.session_state["llm_log"] = []
        
        # Validate API key (strip whitespace first)
        api_key_local = api_key.strip() if api_key else ""
        if not api_key_local and "Custom" not in model_choice:
            st.error(f"⚠️ Please enter your {model_choice} API key in the sidebar!")
            st.stop()
        if "Custom" in model_choice and not custom_api_url:
            st.error("⚠️ Please provide the Custom LLM API URL in the sidebar before running extraction.")
            st.stop()
        
        st.info(f"🤖 Using **{model_choice}** (model: `{model_name}`)")
        
        with st.spinner(f"Fetching PMC full texts for {len(pmids)} PMIDs…"):
            papers = fetch_all_fulltexts(pmids, delay_ms=150)
            _persist("batch_papers", papers)
        
        fetched, no_pmc, errors = _bucketize_papers(papers)
        n_ok, n_no, n_err = len(fetched), len(no_pmc), len(errors)
        st.success(f"PMC texts: ✅ {n_ok} fetched | ⚠️ {n_no} no PMC | ❌ {n_err} errors")
        
        if n_ok:
            with st.expander("Show fetched list", expanded=False):
                st.dataframe(
                    pd.DataFrame(fetched), 
                    width='stretch', 
                    height=min(400, 40 + 28 * len(fetched))
                )
        if n_no:
            with st.expander("Show no-PMC list", expanded=False):
                st.dataframe(
                    pd.DataFrame(no_pmc), 
                    width='stretch', 
                    height=min(400, 40 + 28 * len(no_pmc))
                )
        if n_err:
            with st.expander("Show fetch errors", expanded=False):
                st.dataframe(
                    pd.DataFrame(errors), 
                    width='stretch', 
                    height=min(400, 40 + 28 * len(errors))
                )
        
        if n_ok == 0:
            st.stop()
        
        ok_pmids_this_run = [row["PMID"] for row in fetched]
        
        st.markdown("#### Full texts for this run (selected & fetched)")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pmid in ok_pmids_this_run:
                info = papers.get(pmid, {})
                title = info.get("title") or ""
                pmcid = info.get("pmcid") or ""
                text = (info.get("text") or info.get("fulltext") or info.get("content") or "")
                
                with st.expander(f"{pmid} – {title[:100]}"):
                    if pmcid:
                        st.markdown(f"**PMCID:** {pmcid}  |  [Open PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/)")
                    else:
                        st.markdown("**PMCID:** (not available)")
                    
                    source = (info.get("source") or get_last_fetch_source(pmid) or "unknown")
                    badge = {
                        "jats": "🟢 JATS (XML)", 
                        "html": "🔵 HTML", 
                        "none": "⚪ none"
                    }.get(source, "⚪ unknown")
                    st.markdown(f"**Source:** {badge}")
                    
                    if text:
                        st.text_area("Full text (preview)", value=text, height=300, key=f"ta_{pmid}")
                        st.download_button(
                            f"⬇️ Download full text ({pmid}).txt", 
                            data=text.encode("utf-8"), 
                            file_name=f"{pmid}_{(pmcid or 'NO_PMCID')}.txt", 
                            mime="text/plain", 
                            key=f"dl_txt_{pmid}"
                        )
                        
                        html = info.get("html") or info.get("raw_html")
                        if html:
                            st.download_button(
                                f"⬇️ Download HTML ({pmid}).html", 
                                data=html.encode("utf-8"), 
                                file_name=f"{pmid}_{(pmcid or 'NO_PMCID')}.html", 
                                mime="text/html", 
                                key=f"dl_html_{pmid}"
                            )
                    else:
                        st.info("No full text captured for this paper.")
                
                safe_name = f"{pmid}_{(pmcid or 'NO_PMCID')}".replace("/", "_")
                zf.writestr(f"{safe_name}.txt", text if text else "")
        
        st.download_button(
            "⬇️ Download all selected full texts (.zip)", 
            data=zip_buf.getvalue(), 
            file_name="selected_full_texts.zip", 
            mime="application/zip"
        )

        # ===== LLM extraction phase =====
        papers = st.session_state.get("batch_papers", {})
        ok_pmids_this_run = [pid for pid, info in papers.items() if info.get("status") == "ok"]
        
        if not ok_pmids_this_run:
            st.info("Nothing to analyze: no successfully fetched PMC full texts in this run.")
            st.stop()
        
        llm_log = st.session_state.get("llm_log", [])
        batch_results = st.session_state.get("batch_results", {})
        
        prog = st.progress(0, text="Starting LLM…")
        log_box = st.empty()
        st.markdown("#### Findings")
        table_box = st.empty()
        
        # Pass model selection to analyze_texts (ensure api_key is passed even if env is set)
        # Frontend API key takes priority over env var in backend
        # Also pass current prompt (may be edited by user)
        llm_meta = {
            "model_choice": model_choice,
            "model_name": model_name,
            "api_key": api_key_local,  # This will be used as PRIMARY in backend
            "analyst_prompt": PROMPTS.analyst_prompt,  # Current prompt (includes user edits)
        }
        
        if "Custom" in model_choice:
            if not custom_api_url:
                st.warning("Please provide the Custom LLM API URL before running extraction.")
            llm_meta["api_url"] = custom_api_url
            llm_meta["timeout"] = custom_timeout
            if custom_headers_dict:
                llm_meta["extra_headers"] = custom_headers_dict
            # Auto-detect OpenAI-compatible endpoints
            if custom_api_url and ("/v1" in custom_api_url or "/openai" in custom_api_url.lower()):
                llm_meta["openai_compatible"] = True

        total = len(ok_pmids_this_run)
        for i, pmid in enumerate(ok_pmids_this_run, start=1):
            title = papers[pmid].get("title") or ""
            pmcid = papers[pmid].get("pmcid") or ""
            log_line = f"[{i}/{total}] Analyzing PMID {pmid} ({pmcid}) – {title[:80]}"
            llm_log.append(log_line)
            _persist("llm_log", llm_log)
            log_box.code("\n".join(llm_log[-20:]), language="text")
            
            try:
                single_dict = analyze_texts(
                    {pmid: papers[pmid]},
                    chunk_chars=chunk_chars, 
                    overlap_chars=overlap_chars,
                    delay_ms=delay_ms, 
                    min_confidence=min_conf, 
                    require_mut_quote=True,
                    llm_meta=llm_meta,  # pass model config
                    enable_validation=enable_validation,
                    validation_accuracy_threshold=validation_threshold,
                    validation_delay_ms=validation_delay_ms,
                )
                batch_results.update(single_dict)
                _persist("batch_results", batch_results)
                
                # Show validation status if enabled
                if enable_validation and pmid in single_dict:
                    val_result = single_dict[pmid].get("validation")
                    if val_result and "error" not in val_result:
                        accuracy = val_result.get("overall_accuracy", 0.0)
                        passed = val_result.get("meets_threshold", False)
                        status_icon = "✅" if passed else "⚠️"
                        log_line_val = f"   ↳ Validation: {status_icon} Accuracy: {accuracy:.1f}% (threshold: {validation_threshold}%)"
                        llm_log.append(log_line_val)
                        _persist("llm_log", llm_log)
                        log_box.code("\n".join(llm_log[-20:]), language="text")
                
                out_df_partial = flatten_to_rows(batch_results)
                table_box.dataframe(out_df_partial, width='stretch')
            except Exception as e:
                err_line = f"   ↳ ERROR on PMID {pmid}: {e}"
                llm_log.append(err_line)
                _persist("llm_log", llm_log)
                log_box.code("\n".join(llm_log[-20:]), language="text")
            
            prog.progress(int(i * 100 / total), text=f"LLM progress: {i}/{total}")
        
        st.success("LLM extraction complete ✅")
        
        # Show validation summary if enabled
        if enable_validation:
            batch_results = st.session_state.get("batch_results", {})
            validation_summary = []
            for pmid, entry in batch_results.items():
                if entry.get("status") == "ok" and "validation" in entry:
                    val = entry.get("validation", {})
                    if "error" not in val:
                        validation_summary.append({
                            "PMID": pmid,
                            "Title": entry.get("title", "")[:80],
                            "Accuracy": f"{val.get('overall_accuracy', 0.0):.1f}%",
                            "Status": "✅ Pass" if val.get("meets_threshold") else "⚠️ Fail",
                            "Extractions": val.get("total_extractions", 0),
                            "Validations": f"{val.get('passed_validations', 0)}/{val.get('total_validations', 0)}"
                        })
            
            if validation_summary:
                st.markdown("#### Validation Summary")
                val_df = pd.DataFrame(validation_summary)
                st.dataframe(val_df, width='stretch')
                
                # Show detailed validation results with questions
                st.markdown("#### Detailed Validation Results")
                for pmid, entry in batch_results.items():
                    if entry.get("status") == "ok" and "validation" in entry:
                        val = entry.get("validation", {})
                        if "error" not in val and "extraction_validations" in val:
                            with st.expander(f"📄 {pmid} - {entry.get('title', '')[:60]}... (Accuracy: {val.get('overall_accuracy', 0.0):.1f}%)", expanded=False):
                                extraction_validations = val.get("extraction_validations", [])
                                st.write(f"**Total Extractions:** {len(extraction_validations)}")
                                st.write(f"**Overall Accuracy:** {val.get('overall_accuracy', 0.0):.1f}%")
                                st.write(f"**Status:** {'✅ Passed' if val.get('meets_threshold') else '⚠️ Failed'} (threshold: {validation_threshold}%)")
                                st.divider()
                                
                                for idx, ext_val in enumerate(extraction_validations, 1):
                                    extraction = ext_val.get("extraction", {})
                                    validation = ext_val.get("validation", {})
                                    questions = validation.get("questions", [])
                                    answers = validation.get("answers", [])
                                    scores = validation.get("scores", {})
                                    extraction_info = validation.get("extraction_info", {})
                                    
                                    # Show extraction summary with row info
                                    mutation = ext_val.get("mutation") or extraction.get("mutation", "") or extraction_info.get("mutation", "")
                                    protein = ext_val.get("protein") or extraction.get("protein", "") or extraction_info.get("protein", "")
                                    virus = ext_val.get("virus") or extraction.get("virus", "") or extraction_info.get("virus", "")
                                    position = ext_val.get("position") or extraction.get("position") or extraction_info.get("position")
                                    
                                    ext_title = f"**Extraction {idx} (Row {ext_val.get('extraction_index', idx)}):** "
                                    if mutation:
                                        ext_title += f"Mutation {mutation}"
                                    elif protein:
                                        ext_title += f"Feature in {protein}"
                                    else:
                                        ext_title += "Feature"
                                    
                                    if position:
                                        ext_title += f" @ position {position}"
                                    
                                    if virus:
                                        ext_title += f" ({virus})"
                                    
                                    st.markdown(ext_title)
                                    
                                    # Show row info in a compact format
                                    row_info_parts = []
                                    if mutation:
                                        row_info_parts.append(f"Mutation: {mutation}")
                                    if protein:
                                        row_info_parts.append(f"Protein: {protein}")
                                    if position:
                                        row_info_parts.append(f"Position: {position}")
                                    if row_info_parts:
                                        st.caption(" | ".join(row_info_parts))
                                    
                                    # Show questions and answers
                                    if questions and answers:
                                        st.markdown("**Questions Asked:**")
                                        for q_idx, question in enumerate(questions, 1):
                                            # Find corresponding answer (question number is 1-indexed)
                                            answer_obj = next(
                                                (a for a in answers if isinstance(a, dict) and a.get("question") == q_idx),
                                                {"answer": "UNCERTAIN", "rationale": "No answer found"}
                                            )
                                            answer = answer_obj.get("answer", "UNCERTAIN")
                                            rationale = answer_obj.get("rationale", "")
                                            
                                            # Color code based on answer
                                            if answer.upper() == "YES":
                                                status_icon = "✅"
                                            elif answer.upper() == "NO":
                                                status_icon = "❌"
                                            else:
                                                status_icon = "⚠️"
                                            
                                            st.markdown(f"{status_icon} **Q{q_idx}:** {question}")
                                            st.markdown(f"   **Answer:** {answer}")
                                            if rationale:
                                                st.markdown(f"   **Rationale:** {rationale}")
                                        
                                        # Show extraction accuracy
                                        ext_accuracy = scores.get("accuracy", 0.0)
                                        st.markdown(f"**Extraction Accuracy:** {ext_accuracy:.1f}% ({scores.get('passed', 0)}/{scores.get('total', 0)} questions passed)")
                                    
                                    st.markdown("---")
        
        out_df = flatten_to_rows(st.session_state.get("batch_results", {}))
        table_box.dataframe(out_df, width='stretch')
        
        colD, colE = st.columns([1, 1])
        with colD:
            st.download_button(
                "⬇️ Download findings (.csv)", 
                data=out_df.to_csv(index=False).encode("utf-8"), 
                file_name="pubmed_mutation_findings.csv", 
                mime="text/csv"
            )
        with colE:
            st.download_button(
                "⬇️ Download raw JSON (.json)", 
                data=json.dumps(st.session_state["batch_results"], ensure_ascii=True, indent=2).encode("utf-8"), 
                file_name="pubmed_batch_results.json", 
                mime="application/json"
            )


if __name__ == "__main__":
    main()
