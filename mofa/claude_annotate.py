#!/usr/bin/env python3
"""
annotate_with_opus.py
=====================

Send a MOFA factor's top genes and top m/z values to Claude Opus 4.7
for putative metabolite identification, with the gene signature used as
a biological prior.

WHY CLAUDE
-------------------------
This task isn't one lookup; it's a chain of small reasoning steps:
  1. Recognize the gene signature -> infer brain region / cell type
  2. Detect adduct ladders in the m/z list (+21.98, +37.96, +1.003 etc.)
     and collapse adduct families to single parent neutral masses
  3. Match parent masses against HMDB/LipidMaps mentally
  4. Filter candidates by biological plausibility given the gene context
  5. Assign Schymanski confidence levels and flag what to validate

Each step is easy in isolation; the value is *integrating* them. That's
exactly the kind of multi-source pattern matching where Opus 4.7's
deeper reasoning pays off vs. Sonnet — and a single API call replaces
~1-2 hours of manually crossing HMDB results with the gene set.

Cost (claude-opus-4-7 at $5/M input, $25/M output, Nov 2026):
  Per factor: ~2-3K input + ~2-4K output tokens -> ~$0.06-0.10
  All 12 factors for one sample: ~$1
  All 12 factors x 3 samples: ~$3
  -> Trivial relative to the manual-annotation alternative.

WHAT THIS GIVES YOU
-------------------
Schymanski Level 3 annotations: putative candidates supported by
accurate mass + adduct chemistry + biological context. NOT Level 1
(would require MS/MS + reference standard). Fine for a class project;
for a paper, follow up with METASPACE on the raw imzML files.

USAGE
-----
    pip install anthropic pandas
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 annotate_with_opus.py \\
        --gene_csv gene_loadings_mofa_ALL_FACTORS_mouse_V11L12_038_D1.csv \\
        --msi_csv  msi_loadings_mofa_ALL_FACTORS_mouse_V11L12_038_D1.csv \\
        --factor 5

    # Or annotate every factor in one go:
    python3 annotate_with_opus.py --gene_csv ... --msi_csv ... --factor all

Outputs one markdown file per factor:
    metabolite_id_opus_<sample>_factor<k>.md
Each file contains the exact prompt sent + the full Opus response,
so the analysis is reproducible.
"""

import argparse
import os
import re
import sys
import pathlib
import pandas as pd

try:
    from anthropic import Anthropic
except ImportError:
    sys.exit("Missing dep: pip install anthropic")

MODEL = "claude-opus-4-7"

# System prompt
SYSTEM_PROMPT = """You are an expert in mouse brain neurochemistry and \
mass spectrometry imaging (MSI). You specialize in putative metabolite \
identification from MALDI-MSI accurate-mass data, integrating:

- Adduct chemistry (negative mode: [M-H]-, [M+Cl]-, [M-2H+Na]-, [M-3H+2Na]-, [M-H2O-H]-, etc.)
- Isotopologue patterns (+1.003 Da for 13C1, +2.006 Da for 13C2, +1.996 Da for 34S)
- Biological priors from co-localized gene expression (Allen Brain Atlas anatomy, cell-type markers)
- HMDB, METLIN, LipidMaps, and KEGG database contents from your training data

RULES:

1. Never invent HMDB IDs. If you are not highly confident of the exact ID, write "HMDB ID unknown" or give a candidate name without an ID.
2. Use Schymanski confidence levels:
   - Level 1: confirmed by MS/MS + reference standard. NEVER claim this -- you don't have MS/MS data.
   - Level 2: probable structure (library MS/MS match or strong literature precedent). Use sparingly.
   - Level 3: tentative candidate from accurate mass + biological context. THIS IS YOUR DEFAULT.
   - Level 4: unequivocal molecular formula only.
   - Level 5: just an interesting exact mass.
3. Default to Level 3. If a peak has no plausible candidate at <50 ppm, say "no good match" -- do not force-fit.
4. Always cross-check candidates against the gene-based biological context. A candidate that is biochemically plausible but localized to a different brain region than the gene signature suggests should be flagged.
5. Be honest about uncertainty. "Most likely" / "possible" / "unclear" are valid answers.
"""

# User prompt template
USER_PROMPT = """# Task: Putative metabolite identification for one MOFA factor

## Experimental setup
- Tissue: adult mouse brain, coronal section
- Modality: paired 10x Visium spatial transcriptomics + MALDI mass spectrometry imaging
- MSI polarity: NEGATIVE ion mode
- Mass accuracy: assume ~10-50 ppm (TOF-class unless otherwise stated)
- Sample ID: {sample}
- Analysis: MOFA+ multi-omics factor analysis identified {n_factors} factors. \
This request concerns FACTOR {factor}, which co-varies between RNA and MSI across spots.

## Top genes loading on this factor (sign indicates direction)
{gene_block}

## Top m/z values loading on this factor (sign indicates direction)
{mz_block}

## Adduct shifts to look for (negative mode)
{adduct_hints}

## Deliverables (use markdown headers)

### 1. Brain region / cell type inference
From the gene signature alone, what brain region and/or cell type does this factor capture? One paragraph. Be specific (e.g. "dorsal striatum, medium spiny neurons -- direct and indirect pathways co-expressed").

### 2. Adduct family grouping
Walk through the m/z list and group peaks that look like adducts/isotopologues of one parent. For each family give:
- representative [M-H]- m/z
- inferred parent neutral mass (Da, to 4 decimals)
- table of member peaks with their assigned adduct type and observed mass difference
Singleton peaks that don't fit any family should be listed separately.

### 3. Candidate metabolite assignments
For each adduct family / singleton, propose 1-3 candidate metabolites. For each:
- molecular formula
- theoretical neutral mass + ppm error vs observed parent
- candidate name (and HMDB ID **only if highly confident**, else "HMDB unknown")
- one-line biological justification, **explicitly using the gene context above**
- Schymanski level (default 3)
- "matches gene context": yes / partial / no -- does the candidate's typical brain localization match what the genes predict?

If a peak has no good <50 ppm match, say so. Do not force matches.

### 4. Top 3 priorities for validation
Rank the 3 candidates that are most worth following up (targeted MS/MS, isotope check, or purchasing a standard). Explain why each one matters for the biology.

### 5. Caveats
Mass accuracy assumptions, possible alternative adduct interpretations, ambiguity in the gene signature, anything else that affects confidence.
"""

ADDUCT_HINTS = """- +21.982 Da : [M-2H+Na]- (Na-for-H swap)
- +37.956 Da : [M-2H+K]- (K-for-H swap)
- +43.964 Da : [M-3H+2Na]- (double Na-for-H swap)
- +34.969 Da : [M+35Cl]- (chloride adduct)
- +36.966 Da : [M+37Cl]- (chloride adduct, 37Cl isotope)
- +1.003 / +2.007 Da : 13C1 / 13C2 isotopologues
- -18.011 Da : [M-H2O-H]- (water loss)
- +1.996 Da : 34S isotopologue (sulfur-containing compounds)"""


# ============================================================
# Helpers
# ============================================================
def top_per_factor(df, factor, key_col, n):
    """Return top-N rows for a given factor, sorted by |w| descending."""
    sub = df[df["component"] == factor].copy()
    if "w_abs" not in sub.columns:
        sub["w_abs"] = sub["w"].abs()
    return sub.sort_values("w_abs", ascending=False).head(n)


def format_block(rows, name_col):
    """Format a top-N table as a bulleted list with signed weights."""
    return "\n".join(
        f"- {getattr(r, name_col).replace('mz.', '') if name_col == 'msi' else getattr(r, name_col)}  "
        f"(w = {r.w:+.4f})"
        for r in rows.itertuples()
    )


def extract_base(csv_path):
    m = re.search(r"ALL_FACTORS_(.+?)\.csv", os.path.basename(csv_path))
    return m.group(1) if m else "unknown_sample"


def annotate_factor(client, gene_df, msi_df, factor, top_genes, top_mz,
                    sample, n_factors, out_dir, dry_run=False):
    genes = top_per_factor(gene_df, factor, "gene", top_genes)
    mzs   = top_per_factor(msi_df,  factor, "msi",  top_mz)
    if len(genes) == 0 or len(mzs) == 0:
        print(f"  factor {factor}: empty, skipping")
        return None

    user_msg = USER_PROMPT.format(
        sample=sample,
        n_factors=n_factors,
        factor=factor,
        gene_block=format_block(genes, "gene"),
        mz_block=format_block(mzs, "msi"),
        adduct_hints=ADDUCT_HINTS,
    )

    out_path = pathlib.Path(out_dir) / f"metabolite_id_opus_{sample}_factor{factor}.md"

    if dry_run:
        print(f"\n--- DRY RUN: prompt for factor {factor} ---")
        print(user_msg)
        return None

    print(f"\n=== Calling {MODEL} for factor {factor} "
          f"({len(genes)} genes, {len(mzs)} m/z) ===")
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "\n".join(b.text for b in resp.content if b.type == "text")

    with open(out_path, "w") as f:
        f.write(f"# Opus annotation -- {sample}, factor {factor}\n\n")
        f.write(f"- **Model**: `{MODEL}`\n")
        f.write(f"- **Input tokens**: {resp.usage.input_tokens}\n")
        f.write(f"- **Output tokens**: {resp.usage.output_tokens}\n")
        f.write(f"- **Approx cost**: "
                f"${resp.usage.input_tokens / 1e6 * 5 + resp.usage.output_tokens / 1e6 * 25:.4f}\n\n")
        f.write("---\n\n## System prompt\n\n```\n")
        f.write(SYSTEM_PROMPT)
        f.write("\n```\n\n## User prompt\n\n```\n")
        f.write(user_msg)
        f.write("\n```\n\n---\n\n## Opus response\n\n")
        f.write(text)

    print(f"    -> {out_path}  "
          f"({resp.usage.input_tokens} in, {resp.usage.output_tokens} out)")
    return out_path


# Main
def main():
    ap = argparse.ArgumentParser(__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene_csv", required=True,
                    help="Path to gene_loadings_mofa_ALL_FACTORS_<base>.csv")
    ap.add_argument("--msi_csv",  required=True,
                    help="Path to msi_loadings_mofa_ALL_FACTORS_<base>.csv")
    ap.add_argument("--factor",   required=True,
                    help="Factor index (0-based int) or 'all'")
    ap.add_argument("--top_genes", type=int, default=30,
                    help="Top N genes per factor in prompt (default 30)")
    ap.add_argument("--top_mz",    type=int, default=25,
                    help="Top N m/z per factor in prompt (default 25)")
    ap.add_argument("--out_dir",   default=".",
                    help="Output directory (default: cwd)")
    ap.add_argument("--dry_run",   action="store_true",
                    help="Print prompts only, don't call the API")
    args = ap.parse_args()

    if not args.dry_run and "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("ERROR: export ANTHROPIC_API_KEY=sk-ant-... before running")

    gene_df = pd.read_csv(args.gene_csv)
    msi_df  = pd.read_csv(args.msi_csv)
    sample  = extract_base(args.gene_csv)
    factors = sorted(gene_df["component"].unique().tolist())
    n_factors = len(factors)

    if args.factor.lower() == "all":
        targets = factors
    else:
        targets = [int(args.factor)]

    client = None if args.dry_run else Anthropic()

    print(f"Sample      : {sample}")
    print(f"Factors     : {n_factors} total ({factors[0]}..{factors[-1]})")
    print(f"Targets     : {targets}")
    print(f"top_genes={args.top_genes}, top_mz={args.top_mz}")

    for k in targets:
        annotate_factor(client, gene_df, msi_df, k,
                        args.top_genes, args.top_mz,
                        sample, n_factors, args.out_dir,
                        dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()