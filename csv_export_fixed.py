# ============================================================
# pipeline/csv_export.py
# Final CSV Exporter (Strict Column Alignment)
# - mutation_standard : short literal form (e.g., "M2-S31N", "A226V")
# - mutation_hgvs     : full HGVS (e.g., "p.Ser31Asn", "p.Ala226Val")
# ============================================================

from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional
import pandas as pd

# ---------------- Amino Acid Lookup ----------------
AA1_TO_AA3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val", "*": "Ter"
}

AA3_TO_AA1 = {v: k for k, v in AA1_TO_AA3.items()}

# ---------------- Regex for Mutations ----------------
# Pattern 1: Prefix format (e.g., "M2-S31N", "PA-I38T")
MUT_PREFIX_RE = re.compile(
    r'\b(?P<prefix>[A-Z][A-Z0-9]{0,7})[\-–—](?P<old>[A-Z])(?P<pos>\d+)(?P<new>[A-Z*])\b'
)

# Pattern 2: HGVS format (e.g., "p.Ser31Asn", "p.Ala226Val")
MUT_HGVS_RE = re.compile(
    r'\bp\.(?P<old>[A-Z][a-z]{2})(?P<pos>\d+)(?P<new>[A-Z][a-z]{2}|Ter)\b'
)

# Pattern 3: Simple format (e.g., "A226V", "K128E") - but avoid false positives
MUT_SIMPLE_RE = re.compile(
    r'\b(?P<old>[A-Z])(?P<pos>\d{2,4})(?P<new>[A-Z*])\b'
)

# Patterns to EXCLUDE (figure references, gene names, etc.)
EXCLUDE_PATTERNS = [
    r'\bFig\.\s*[A-Z]\d+[A-Z]\b',  # Fig. S2I
    r'\bTable\s*[A-Z]?\d+[A-Z]\b',  # Table 3A
    r'\b[A-Z]{2,}[0-9]+[A-Z]\b',  # Gene names like HSPA1L, CCR5
    r'\b[A-Z]{2,}-[NT]\d+\b',  # Construct names like HSPA1L-T1, NA-N1
    r'\b[A-Z]{2,}-FL\b',  # Full-length constructs
]

# Compile exclude patterns
EXCLUDE_RE = re.compile('|'.join(EXCLUDE_PATTERNS), re.IGNORECASE)

# ---------------- Utility Functions ----------------
def _is_valid_mutation(match_text: str, context: str) -> bool:
    """Check if matched text is a valid mutation (not a figure/gene/construct reference)."""
    # Check against exclusion patterns
    if EXCLUDE_RE.search(match_text):
        return False
    
    # Check if in methods/primers section (common false positive areas)
    context_lower = context.lower()
    if any(keyword in context_lower for keyword in ['primer', 'forward sequence', 'reverse sequence', 'construct']):
        # Allow if explicitly mentions "mutation" or "mutant"
        if not any(word in context_lower for word in ['mutation', 'mutant', 'substitution', 'variant']):
            return False
    
    return True

def _merge_quotes(quotes: List[str]) -> str:
    """Merge and deduplicate quotes."""
    uniq, seen = [], set()
    for q in quotes:
        if not q:
            continue
        t = q.strip()
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        uniq.append(t)
    uniq.sort(key=len)
    return " | ".join(uniq[:3])

def _aa3_to_aa1(aa3: str) -> Optional[str]:
    """Convert three-letter amino acid code to single letter."""
    return AA3_TO_AA1.get(aa3, None)

def _aa1_to_aa3(aa1: str) -> Optional[str]:
    """Convert single-letter amino acid code to three letters."""
    return AA1_TO_AA3.get(aa1, None)

def _to_hgvs(old_aa: str, position: str, new_aa: str) -> str:
    """
    Convert mutation to HGVS format.
    Args:
        old_aa: Single-letter code (e.g., "S")
        position: Position number (e.g., "31")
        new_aa: Single-letter code (e.g., "N")
    Returns:
        HGVS format (e.g., "p.Ser31Asn")
    """
    old_3 = _aa1_to_aa3(old_aa)
    new_3 = _aa1_to_aa3(new_aa)
    
    if not old_3 or not new_3:
        return ""
    
    return f"p.{old_3}{position}{new_3}"

def _to_standard(prefix: str, old_aa: str, position: str, new_aa: str) -> str:
    """
    Convert mutation to standard short format.
    Args:
        prefix: Protein prefix (e.g., "M2", "PA") or empty string
        old_aa: Single-letter code (e.g., "S")
        position: Position number (e.g., "31")
        new_aa: Single-letter code (e.g., "N")
    Returns:
        Standard format (e.g., "M2-S31N" or "A226V")
    """
    if prefix:
        return f"{prefix}-{old_aa}{position}{new_aa}"
    else:
        return f"{old_aa}{position}{new_aa}"

def _hgvs_to_standard(hgvs: str, protein: str = "") -> str:
    """Convert HGVS format to standard short format."""
    match = MUT_HGVS_RE.search(hgvs)
    if not match:
        return ""
    
    old_3 = match.group('old')
    pos = match.group('pos')
    new_3 = match.group('new')
    
    old_1 = _aa3_to_aa1(old_3)
    new_1 = _aa3_to_aa1(new_3)
    
    if not old_1 or not new_1:
        return ""
    
    return _to_standard(protein, old_1, pos, new_1)

def _confidence(feature: dict, quote: str) -> float:
    """Calculate confidence score for a feature."""
    score = 0.4 if quote else 0.0
    if feature.get("protein"): 
        score += 0.15
    if feature.get("mutation_standard") and feature.get("mutation_hgvs"): 
        score += 0.15
    elif feature.get("mutation_standard") or feature.get("mutation_hgvs"):
        score += 0.1
    if feature.get("virus"): 
        score += 0.1
    if feature.get("evidence_quotes"): 
        score += 0.1
    return min(score, 1.0)

def _dedup_key(row: dict) -> Tuple[str, str, str]:
    """Generate deduplication key for a row."""
    return (
        str(row.get("pmid", "")).strip(),
        str(row.get("protein", "")).upper().strip(),
        (row.get("mutation_standard") or "").upper().strip(),
    )

def _extract_mutations_from_quote(quote: str, base_row: dict) -> List[dict]:
    """
    Extract all valid mutations from a quote.
    Returns list of mutation dictionaries with both standard and HGVS forms.
    """
    if not quote:
        return []
    
    mutations = []
    
    # Extract prefix format (M2-S31N, PA-I38T)
    for match in MUT_PREFIX_RE.finditer(quote):
        match_text = match.group(0)
        if not _is_valid_mutation(match_text, quote):
            continue
        
        prefix = match.group('prefix')
        old = match.group('old')
        pos = match.group('pos')
        new = match.group('new')
        
        standard = _to_standard(prefix, old, pos, new)
        hgvs = _to_hgvs(old, pos, new)
        
        if standard and hgvs:
            mutations.append({
                'protein': prefix,
                'mutation_standard': standard,
                'mutation_hgvs': hgvs,
                'position': int(pos) if pos.isdigit() else pos
            })
    
    # Extract HGVS format (p.Ser31Asn, p.Ala226Val)
    for match in MUT_HGVS_RE.finditer(quote):
        match_text = match.group(0)
        if not _is_valid_mutation(match_text, quote):
            continue
        
        old_3 = match.group('old')
        pos = match.group('pos')
        new_3 = match.group('new')
        
        old_1 = _aa3_to_aa1(old_3)
        new_1 = _aa3_to_aa1(new_3)
        
        if not old_1 or not new_1:
            continue
        
        protein = base_row.get('protein', '')
        standard = _to_standard(protein, old_1, pos, new_1)
        hgvs = match.group(0)
        
        if standard and hgvs:
            mutations.append({
                'protein': protein,
                'mutation_standard': standard,
                'mutation_hgvs': hgvs,
                'position': int(pos) if pos.isdigit() else pos
            })
    
    # Extract simple format (A226V, K128E) - be more conservative
    for match in MUT_SIMPLE_RE.finditer(quote):
        match_text = match.group(0)
        if not _is_valid_mutation(match_text, quote):
            continue
        
        # Only accept if context suggests it's a mutation
        context_window = quote[max(0, match.start()-50):min(len(quote), match.end()+50)].lower()
        if not any(keyword in context_window for keyword in ['mutation', 'mutant', 'substitution', 'variant', 'change']):
            continue
        
        old = match.group('old')
        pos = match.group('pos')
        new = match.group('new')
        
        # Skip if position is too short (likely false positive)
        if len(pos) < 2:
            continue
        
        protein = base_row.get('protein', '')
        standard = _to_standard('', old, pos, new)
        hgvs = _to_hgvs(old, pos, new)
        
        if standard and hgvs:
            mutations.append({
                'protein': protein,
                'mutation_standard': standard,
                'mutation_hgvs': hgvs,
                'position': int(pos) if pos.isdigit() else pos
            })
    
    return mutations

def _spawn_from_quote(base_row: dict, quote: str) -> List[dict]:
    """
    Detect all mutations in a quote and generate rows.
    Each mutation gets both standard and HGVS forms.
    """
    mutations = _extract_mutations_from_quote(quote, base_row)
    
    if not mutations:
        return [base_row]
    
    rows = []
    for mut in mutations:
        row = dict(base_row)
        row.update(mut)
        rows.append(row)
    
    return rows

# ---------------- Main Function ----------------
def flatten_to_rows(batch: Dict[str, Dict]) -> pd.DataFrame:
    """
    Convert LLM batch to a clean DataFrame.
    Ensures both mutation_standard (short) and mutation_hgvs (HGVS) are populated.
    """
    out = {}
    
    for pmid, entry in (batch or {}).items():
        if not isinstance(entry, dict) or entry.get("status") != "ok":
            continue
        
        pmcid = entry.get("pmcid")
        title = entry.get("title")
        feats = (entry.get("result", {}) or {}).get("sequence_features", []) or []
        
        for f in feats:
            quotes = [q for q in (f.get("evidence_quotes") or []) if isinstance(q, str)]
            quote = _merge_quotes(quotes)
            
            base = {
                "pmid": pmid,
                "pmcid": pmcid,
                "title": title,
                "virus": f.get("virus") or "",
                "protein": f.get("protein") or "",
                "mutation_standard": "",
                "mutation_hgvs": "",
                "position": f.get("position"),
                "target_type": "mutation",
                "region_range": "",
                "confidence": 0.0,
                "quote": quote
            }
            
            # Get mutation info from feature
            mut_std = f.get("mutation_standard") or f.get("mutation") or ""
            mut_hgvs = f.get("mutation_hgvs") or ""
            
            # Case 1: Both forms provided by LLM
            if mut_std and mut_hgvs:
                # Ensure standard is in SHORT format (not HGVS)
                if mut_std.startswith("p."):
                    # Convert HGVS to standard
                    mut_std = _hgvs_to_standard(mut_std, base["protein"])
                
                base["mutation_standard"] = mut_std
                base["mutation_hgvs"] = mut_hgvs
                spawn_rows = [base]
            
            # Case 2: Only standard form provided
            elif mut_std and not mut_hgvs:
                # Check if it's actually HGVS format
                if mut_std.startswith("p."):
                    base["mutation_hgvs"] = mut_std
                    base["mutation_standard"] = _hgvs_to_standard(mut_std, base["protein"])
                else:
                    # Convert standard to HGVS
                    base["mutation_standard"] = mut_std
                    match = MUT_PREFIX_RE.search(mut_std) or MUT_SIMPLE_RE.search(mut_std)
                    if match:
                        old = match.group('old')
                        pos = match.group('pos')
                        new = match.group('new')
                        base["mutation_hgvs"] = _to_hgvs(old, pos, new)
                        if not base["protein"] and match.groupdict().get('prefix'):
                            base["protein"] = match.group('prefix')
                
                spawn_rows = [base]
            
            # Case 3: Only HGVS form provided
            elif mut_hgvs and not mut_std:
                base["mutation_hgvs"] = mut_hgvs
                base["mutation_standard"] = _hgvs_to_standard(mut_hgvs, base["protein"])
                spawn_rows = [base]
            
            # Case 4: No mutation provided, try extracting from quote
            else:
                spawn_rows = _spawn_from_quote(base, quote)
            
            # Add rows with confidence scores
            for r in spawn_rows:
                # Skip if both mutation fields are empty
                if not r.get("mutation_standard") and not r.get("mutation_hgvs"):
                    continue
                
                r["confidence"] = _confidence(f, quote)
                
                k = _dedup_key(r)
                if k in out:
                    # Merge quotes
                    out[k]["quote"] = _merge_quotes([out[k]["quote"], r["quote"]])
                    # Keep higher confidence entry
                    if r["confidence"] > out[k]["confidence"]:
                        out[k] = r
                else:
                    out[k] = r
    
    # Create DataFrame
    df = pd.DataFrame(list(out.values()))
    
    if not df.empty:
        # Ensure all required columns exist
        cols = [
            "pmid", "pmcid", "title", "virus", "protein",
            "mutation_standard", "mutation_hgvs",
            "position", "target_type", "region_range",
            "confidence", "quote"
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        
        df = df[cols]
        
        # Clean up empty mutation rows
        df = df[(df["mutation_standard"] != "") | (df["mutation_hgvs"] != "")]
    
    return df

__all__ = ["flatten_to_rows"]
