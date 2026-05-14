# Opus annotation -- mouse_V11L12_038_D1, factor 5

- **Model**: `claude-opus-4-7`
- **Input tokens**: 2706
- **Output tokens**: 4096
- **Approx cost**: $0.1159

---

## System prompt

```
You are an expert in mouse brain neurochemistry and mass spectrometry imaging (MSI). You specialize in putative metabolite identification from MALDI-MSI accurate-mass data, integrating:

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

```

## User prompt

```
# Task: Putative metabolite identification for one MOFA factor

## Experimental setup
- Tissue: adult mouse brain, coronal section
- Modality: paired 10x Visium spatial transcriptomics + MALDI mass spectrometry imaging
- MSI polarity: NEGATIVE ion mode
- Mass accuracy: assume ~10-50 ppm (TOF-class unless otherwise stated)
- Sample ID: mouse_V11L12_038_D1
- Analysis: MOFA+ multi-omics factor analysis identified 12 factors. This request concerns FACTOR 5, which co-varies between RNA and MSI across spots.

## Top genes loading on this factor (sign indicates direction)
- Penk  (w = +0.5566)
- Gpr88  (w = +0.5165)
- Pde10a  (w = +0.4979)
- Ppp1r1b  (w = +0.4916)
- Rgs9  (w = +0.4905)
- Pde1b  (w = +0.4733)
- Adcy5  (w = +0.4719)
- Gng7  (w = +0.4708)
- Adora2a  (w = +0.4655)
- Rasd2  (w = +0.4585)
- Drd2  (w = +0.4553)
- Olfm1  (w = -0.4523)
- Vsnl1  (w = -0.4474)
- Syndig1l  (w = +0.4466)
- Scn4b  (w = +0.4311)
- Stmn1  (w = -0.4264)
- Slc17a7  (w = -0.4261)
- Pcp4l1  (w = +0.4197)
- Nrn1  (w = -0.4153)
- Syt13  (w = -0.4139)
- Cck  (w = -0.4109)
- Lrrc10b  (w = +0.4091)
- Necab2  (w = +0.4077)
- Bcl11b  (w = +0.4077)
- Tac1  (w = +0.4065)
- Rxrg  (w = +0.4023)
- Stx1a  (w = -0.4019)
- Meis2  (w = +0.3990)
- Arpp21  (w = +0.3964)
- 1110008P14Rik  (w = -0.3890)

## Top m/z values loading on this factor (sign indicates direction)
- 275.03193999999996  (w = +0.3790)
- 253.049875  (w = +0.3730)
- 296.99915  (w = +0.3605)
- 275.01739  (w = +0.3391)
- 881.51323  (w = -0.3281)
- 909.5483999999999  (w = -0.3237)
- 264.05398  (w = -0.3054)
- 271.00396  (w = +0.3004)
- 312.973185  (w = +0.3003)
- 550.0064150000001  (w = +0.2987)
- 272.99984  (w = +0.2981)
- 271.00518  (w = +0.2947)
- 541.97996  (w = +0.2933)
- 278.03395  (w = +0.2927)
- 271.98805  (w = +0.2873)
- 132.030355  (w = -0.2795)
- 238.00025499999998  (w = +0.2716)
- 286.97941000000003  (w = +0.2679)
- 260.96626000000003  (w = +0.2671)
- 287.96182  (w = +0.2657)
- 286.97793  (w = +0.2632)
- 506.07591  (w = +0.2629)
- 475.97911  (w = +0.2615)
- 910.5514049999999  (w = -0.2591)
- 495.92363  (w = -0.2570)

## Adduct shifts to look for (negative mode)
- +21.982 Da : [M-2H+Na]- (Na-for-H swap)
- +37.956 Da : [M-2H+K]- (K-for-H swap)
- +43.964 Da : [M-3H+2Na]- (double Na-for-H swap)
- +34.969 Da : [M+35Cl]- (chloride adduct)
- +36.966 Da : [M+37Cl]- (chloride adduct, 37Cl isotope)
- +1.003 / +2.007 Da : 13C1 / 13C2 isotopologues
- -18.011 Da : [M-H2O-H]- (water loss)
- +1.996 Da : 34S isotopologue (sulfur-containing compounds)

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

```

---

## Opus response

# Factor 5 — Putative Metabolite ID

## 1. Brain region / cell type inference

This is a textbook **striatal medium spiny neuron (MSN)** signature, with both pathways represented. The pan-MSN/striatal markers are unmistakable: **Ppp1r1b (DARPP-32), Gpr88, Pde10a, Rgs9, Bcl11b (Ctip2), Rasd2 (Rhes), Arpp21, Pde1b, Adcy5, Foxp-axis partners (Meis2, Rxrg)**. The indirect (iMSN/D2) pathway is explicitly marked by **Adora2a, Drd2, Penk, Gpr6 (implied family)**, and the direct (dMSN/D1) pathway by **Tac1**. The strong negative loadings of **Slc17a7 (vGlut1), Nrn1, Olfm1, Vsnl1, Stmn1, Cck** indicate anti-correlation with cortical/hippocampal glutamatergic neurons — i.e. this factor separates striatum from cortex. Best interpretation: **dorsal (and likely ventral) striatum, MSNs of both D1 and D2 classes, contrasted against cortical excitatory neurons.**

---

## 2. Adduct family grouping

The positive-loading m/z list is dominated by low, **halogen/sulfur/phosphate-looking masses with characteristic Cl isotope pairs (Δ ≈ 1.997)** and Na/K swaps. The two large negative-loading peaks at 881.5 / 909.5 are clearly lipids (cortical PI/PS class). Let me group:

### Family A — parent ~254.05 Da (sulfate/phosphate ester?)
| Observed m/z | Assignment | Δ vs [M-H]- | Notes |
|---|---|---|---|
| 253.04988 | [M-H]- | 0 | parent |
| 275.03194 | [M-2H+Na]- | +21.982 | Na swap ✓ |
| 278.03395 | – | +25.0 | likely a different parent (see Family C 13C) |

Neutral mass ≈ **254.0571 Da**.

### Family B — parent ~272.01 Da, chlorinated / sulfated species
| Observed m/z | Assignment | Δ | Notes |
|---|---|---|---|
| 271.00396 | [M-H]- | 0 | parent |
| 271.00518 | [M-H]- (dup./drift) | +0.001 | same peak, binning artifact |
| 272.99984 | 13C1 or 34S? | +1.996 | **34S isotopologue** → S-containing |
| 296.99915 | [M-2H+Na]- | +25.995 | not clean Na-swap; could be [M+Na-2H]- on a +25 variant |

Reinterpretation: 271.004 and 272.000 differ by 0.996 (13C1) — actually **+1.996 = 34S**, supporting sulfur. 296.999 = 271.004 + 25.995, which does **not** match a standard adduct (+21.98 Na or +34.97 Cl). Possibly an unrelated peak.

Neutral mass ≈ **272.0112 Da**.

### Family C — parent ~275.02 Da (chloride/sulfate cluster — taurine-conjugate-like)
| Observed m/z | Assignment | Δ | Notes |
|---|---|---|---|
| 275.01739 | [M-H]- | 0 | parent |
| 286.97941 | ? | +11.962 | not standard |
| 286.97793 | ? (dup) | – | – |
| 287.96182 | 13C/34S of 286.979 | +0.982 | within-family of 286.98 |
| 312.97319 | [M+Cl]- on a 278-Da neutral | – | see Family D |

275.017 likely unrelated to 275.032 (Family A) — different by 0.015 = 55 ppm, clearly distinct.

### Family D — heavy halogenated / iodinated-looking cluster (~261, 287, 313, ...)
This cluster spacing **+25.99** repeats (260.966 → 286.979 → 312.973). +25.99 is not a clean common adduct; however **Δ = 51.99 between 260.97 and 312.97 ≈ 2× Cl (69.94) — not matching**. Better: differences between 271.988, 286.979, 312.973 suggest a halogenated/iodide-rich contaminant series, **possibly matrix cluster ions** (DHB, 9-AA, or CHCA clusters with Na/K/Cl).

| Observed m/z | Tentative |
|---|---|
| 238.00026 | matrix cluster? |
| 260.96626 | matrix/Cs/heavy adduct? |
| 271.98805 | – |
| 286.97941 / 286.97793 | – |
| 287.96182 | 34S of 286 series |
| 312.97319 | – |
| 475.97911 | dimer of ~238? (2×238.0 -H ≈ 475.0) ✓ within ~1 Da |
| 495.92363 | – |
| 506.07591 | – |
| 541.97996 | dimer of 272.0 minus 2H ≈ 541.99 ✓ |
| 550.00642 | dimer of 275.01 ≈ 550.03 (off) |

**These low-mass, oddly-spaced, halogen-rich peaks are most consistent with MALDI matrix/inorganic cluster ions (e.g., (NaCl)n adducts, DHB clusters, or trifluoroacetate clusters)** rather than endogenous metabolites. Flagging as suspect.

### Family E — phospholipids (negative-loading, cortical)
| Observed m/z | Assignment |
|---|---|
| 881.51323 | [M-H]- of **PI 38:4** (C47H83O13P) |
| 909.54840 | [M-H]- of **PI 40:4** or **PS 40:?** |
| 910.55140 | 13C1 of 909.548 ✓ (+1.003) |

### Singletons
- **132.03036** (negative loading): small acidic metabolite, possibly **aspartate [M-H]- (theoretical 132.0302)** — excellent match, <5 ppm
- **264.05398** (negative loading): possible **N-acetylaspartylglutamate fragment** or other — see below
- **506.07591**: unclear

---

## 3. Candidate metabolite assignments

### Family A — neutral 254.057 Da
**Candidate A1: Adenosine** (C10H13N5O4, monoisotopic 267.097) — **does NOT fit** (too heavy).
**Candidate A2: Guanine + ribose fragment?** — no good <50 ppm match in HMDB for 254.057 in striatal context.
**Candidate A3: dAMP fragment / cGMP-related?** cGMP = C10H12N5O7P = 345.047 — no.
**Candidate A3: Xanthosine?** C10H12N4O6 = 284.075 — no.

Honestly: **no confident <50 ppm endogenous match for 254.057** that fits striatal MSN biology. Level 5.

### Family B — neutral 272.011 Da, sulfur-containing (34S evidence)
- **Candidate B1: Adenosine 3′,5′-cyclic monophosphate (cAMP)?** C10H12N5O6P = 329.053 — no.
- **Candidate B2: 3′,5′-cyclic GMP related sulfate?** 
- **Candidate B3: Inosine monophosphate fragment?**

With the 34S signature and m/z 271.004, this looks like a **small sulfated/phosphorylated species**. Best biochemical candidate given striatal/cAMP-signaling gene context (Adcy5, Pde10a, Pde1b, Pde —> cyclic nucleotides):

- **Adenosine-2′,3′-cyclic phosphate / adenosine monophosphate fragment** — doesn't fit mass.
- **2′-deoxyadenosine-5′-monophosphate (dAMP)**: C10H14N5O6P, neutral 331.068 — no.

**No confident match.** The 34S signal is the strongest constraint and argues for a true sulfur-containing endogenous molecule. Level 4 at best (formula not even confidently assignable from one peak).

### Family C — neutral 276.025 Da
- **Candidate C1: Adenosine-5′-monophosphate (AMP)** C10H14N5O7P, neutral mass 347.0631 — no.
- **Candidate C2: Inosine** C10H12N4O5 = 268.0808 — no.
- **Candidate C3: Guanosine-related** — no clean match.

**No confident <50 ppm match.**

### Family D — matrix/inorganic clusters (suspected)
Treat as analytical artifacts unless validated. Level 5.

### Family E — phospholipids (negative loadings = cortical)
- **881.51323**: **PI 38:4 (1-stearoyl-2-arachidonoyl-sn-glycero-3-phosphoinositol)** [M-H]-, C47H83O13P, theoretical 881.5180 → **~5 ppm**. HMDB unknown (multiple isomers; PI 18:0/20:4 is HMDB0009889 family — not asserting). **Level 3**. Gene-context match: **partial** — PI 38:4 is abundant brain-wide; negative loading vs MSN signature is consistent with cortical/hippocampal enrichment, where Slc17a7+ neurons dominate. **Yes, matches "cortex side" of the contrast.**
- **909.54840**: **PI 40:4** (C49H87O13P, theoretical 909.5493) → <1 ppm. **Level 3**. Same cortex-leaning interpretation. **Matches: yes (cortical side).**
- **910.55140**: 13C1 of PI 40:4. Confirmatory.

### Singleton 132.03036
- **Candidate: L-Aspartate** [M-H]-, C4H7NO4, theoretical [M-H]- = 132.0302 → **~0.5 ppm**. **HMDB0000191** (L-aspartic acid) — I'm confident on this ID.
- Biological justification: aspartate is enriched in **glutamatergic cortical neurons** (it is co-released with glutamate, and is a precursor of NAA via Nat8l in neurons). Negative loading = enriched on the cortical/Slc17a7+ side, depleted in striatum. **Matches gene context: yes.**
- **Level 2** (very strong mass match + well-established biology + classic MALDI-negative-mode peak).

### Singleton 264.05398
- **Candidate: 3′,5′-cyclic AMP–related? cAMP [M-H]- = 328.045** — no.
- **Candidate: N-Acetylaspartate (NAA) Na adduct?** NAA = C6H9NO5, [M-H]- = 174.0408; +Na swap = 195.99 — no.
- **Candidate: Phosphoethanolamine dimer / GPC fragment?** — no clean match.
- **No confident <50 ppm endogenous match.** Could be a lipid headgroup fragment. Level 5.

---

## 4. Top 3 priorities for validation

1. **m/z 881.513 and 909.548 — PI 38:4 and PI 40:4.** These are the most confident IDs in the list, with sub-5-ppm mass match and a clear, interpretable biology: the factor anti-correlates these phosphoinositides with striatal MSN markers, implicating **cortical PI-pool enrichment / striatal PI depletion**. Worth confirming by MS/MS (diagnostic fatty-acyl losses: 283.26 for 18:0, 303.23 for 20:4) because PI signaling differences across striatum vs cortex are biologically meaningful (PI3K/PLC pathways).

2. **m/z 132.030 — L-A