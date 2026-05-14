# Opus annotation -- mouse_V11T16_085_C1, factor 0

- **Model**: `claude-opus-4-7`
- **Input tokens**: 2682
- **Output tokens**: 4096
- **Approx cost**: $0.1158

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
- Sample ID: mouse_V11T16_085_C1
- Analysis: MOFA+ multi-omics factor analysis identified 6 factors. This request concerns FACTOR 0, which co-varies between RNA and MSI across spots.

## Top genes loading on this factor (sign indicates direction)
- Slc17a7  (w = +0.2348)
- Pacs2  (w = -0.1985)
- Rpl4  (w = -0.1949)
- Nrgn  (w = +0.1825)
- Bc1  (w = +0.1821)
- Mobp  (w = -0.1763)
- Malat1  (w = +0.1709)
- Olfm1  (w = +0.1691)
- Endod1  (w = -0.1655)
- mt-Co2  (w = +0.1539)
- Nnat  (w = +0.1539)
- mt-Nd3  (w = +0.1457)
- Sparc  (w = -0.1398)
- Ptk2b  (w = +0.1374)
- mt-Atp6  (w = +0.1362)
- mt-Nd4  (w = +0.1348)
- Cpne7  (w = +0.1347)
- Slc6a11  (w = -0.1343)
- Slc32a1  (w = -0.1330)
- Pvalb  (w = -0.1310)
- Vamp1  (w = -0.1288)
- Slc1a2  (w = +0.1280)
- Poglut1  (w = -0.1280)
- Syt2  (w = -0.1265)
- Mag  (w = -0.1246)
- Plp1  (w = -0.1239)
- Stmn4  (w = -0.1228)
- Prkcg  (w = +0.1222)
- Ldhb  (w = -0.1200)
- Ptgds  (w = -0.1193)

## Top m/z values loading on this factor (sign indicates direction)
- 431.08294  (w = +0.5136)
- 432.08618  (w = +0.5105)
- 433.08094  (w = +0.4917)
- 431.08737  (w = +0.4907)
- 415.10892  (w = +0.4827)
- 469.03851  (w = +0.4784)
- 862.09285  (w = +0.4773)
- 469.04201  (w = +0.4690)
- 433.07846  (w = +0.4538)
- 416.11226  (w = +0.4504)
- 453.06455  (w = +0.4489)
- 482.10025  (w = +0.4481)
- 352.13361  (w = +0.4480)
- 351.14952  (w = +0.4447)
- 325.08221  (w = +0.4360)
- 324.07894  (w = +0.4321)
- 382.14402  (w = +0.4301)
- 624.27776  (w = +0.4298)
- 433.08939  (w = +0.4294)
- 470.04199  (w = +0.4285)
- 622.29085  (w = +0.4271)
- 431.07851  (w = +0.4257)
- 862.08498  (w = +0.4254)
- 310.12302  (w = +0.4248)
- 554.0458  (w = +0.4228)

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

# MOFA Factor 0 — Putative Metabolite ID

## 1. Brain region / cell type inference

The positive side of this factor is dominated by **excitatory cortical / hippocampal pyramidal neuron markers**: *Slc17a7* (VGLUT1), *Nrgn* (neurogranin), *Bc1* (dendritic ncRNA enriched in forebrain neurons), *Olfm1*, *Cpne7*, *Prkcg*, *Ptk2b*, plus the dendritic lncRNA *Malat1* and high mitochondrial gene load (*mt-Co2/Nd3/Atp6/Nd4*) consistent with metabolically active neuropil. *Slc1a2* (astrocytic glutamate transporter EAAT2) and *Nnat* also load positive, fitting glutamatergic gray-matter neuropil with surrounding astrocytic support. The negative side carries **myelin/oligodendrocyte markers** (*Mobp*, *Mag*, *Plp1*, *Ptgds*) plus **GABAergic / parvalbumin interneuron** markers (*Slc32a1* = VGAT, *Pvalb*, *Syt2*, *Vamp1*, *Slc6a11* = GAT-3) and a stromal/ECM marker (*Sparc*). So Factor 0 is essentially a **glutamatergic gray matter (cortex + hippocampus CA fields) axis vs. white-matter / GABAergic-interneuron axis**. The positive-loading metabolites should mark neuropil-rich glutamatergic cortex/hippocampus.

## 2. Adduct family grouping

Working through the list, several peaks cluster tightly. All masses below are observed.

### Family A — parent ~432.09 Da (likely a hexose-phosphate / nucleotide-like compound, see §3)
| m/z | Δ vs 431.0829 | Interpretation |
|---|---|---|
| 431.08294 | 0 | [M-H]⁻ (reference) |
| 432.08618 | +1.0032 | ¹³C₁ isotopologue |
| 433.08094 | +1.9980 | ³⁴S or ¹³C₂ isotopologue |
| 431.08737 | +0.0044 | same peak, mass-recalibration variant (duplicate centroid) |
| 433.08939 | +2.0064 | ¹³C₂ isotopologue (alt centroid) |
| 431.07851 | −0.0044 | same peak, alt centroid |
| 433.07846 | +1.9955 | ³⁴S isotopologue (alt centroid) |

The presence of a clean +2.00 satellite slightly favors a **sulfur-containing** parent, but +2.006 ¹³C₂ cannot be excluded at TOF resolution.

### Family B — parent ~416.12 Da
| m/z | Δ | Interpretation |
|---|---|---|
| 415.10892 | 0 | [M-H]⁻ |
| 416.11226 | +1.0033 | ¹³C₁ |
| **Possible link to Family A:** 431.083 − 415.109 = 15.974 ≈ +O (oxidation) — could be oxidized/reduced pair |

### Family C — parent ~470.04 Da (sulfur-rich)
| m/z | Δ | Interpretation |
|---|---|---|
| 469.03851 | 0 | [M-H]⁻ |
| 470.04199 | +1.0035 | ¹³C₁ |
| 469.04201 | +0.0035 | duplicate centroid |
| 453.06455 | −15.974 | possible −O analog (or unrelated) |

### Family D — parent ~352.13 Da
| m/z | Δ | Interpretation |
|---|---|---|
| 351.14952 | 0 | [M-H]⁻ |
| 352.13361 | (+0.984, NOT 1.003) — actually this is **not** the ¹³C of 351.149; mass differs by 0.984. Treat 352.134 as a **separate** species; its true [M-H]⁻ partner is likely an unobserved 351.13. |
| 310.12302 | −41.027 | could be neutral loss of C₂H₃NO (acetamide) or unrelated |

So Family D splits: **D1 = 351.1495**, **D2 = 352.1336** (independent).

### Family E — parent ~325.08 Da
| m/z | Δ | Interpretation |
|---|---|---|
| 324.07894 | 0 | [M-H]⁻ |
| 325.08221 | +1.0033 | ¹³C₁ |

### Family F — parent ~862.09 Da
| m/z | Δ | Interpretation |
|---|---|---|
| 862.09285 | 0 | [M-H]⁻ |
| 862.08498 | −0.0079 | likely duplicate centroid of same ion (peak-picking artifact) |

Note: 862.09 ≈ 2 × 431.08 + ~0 → could be a **dimer/cluster [2M-H]⁻ of Family A**. That is strong evidence Family A and Family F are the same molecule.

### Singletons / unassigned
- 482.10025
- 624.27776
- 622.29085 (Δ = −1.687 from 624.278 → not a clean isotope; the pair 624.278 / 622.291 differ by ~1.987, possibly a redox pair or unrelated lipids)
- 382.14402
- 554.0458

---

## 3. Candidate metabolite assignments

### Family A (parent neutral mass = 432.0902 Da; dimer at 862.09 supports)

| Candidate | Formula | Theor. neutral | ppm | HMDB | Justification (gene context) | Level | Matches context |
|---|---|---|---|---|---|---|---|
| **UDP** (uridine diphosphate) | C₉H₁₄N₂O₁₂P₂ | 404.0022 | far off | — | reject | — | — |
| **AMP** (adenosine-5′-monophosphate) | C₁₀H₁₄N₅O₇P | 347.0631 | far off | — | reject | — | — |
| **Glutathione disulfide (GSSG)** | C₂₀H₃₂N₆O₁₂S₂ | 612.152 | far off | — | reject | — | — |
| **dGMP (2′-deoxyguanosine-5′-monophosphate)** | C₁₀H₁₄N₅O₇P | 347.063 | reject | | | | |
| **Glucose-1,6-bisphosphate** | C₆H₁₄O₁₂P₂ | 340.000 | reject | | | | |
| **Pantothenoyl-related / CoA fragments** | — | — | — | — | — | — | — |
| **Sedoheptulose 1,7-bisphosphate** | C₇H₁₆O₁₃P₂ | 370.006 | reject | | | | |
| **N-acetylneuraminic acid (Neu5Ac) dimer-like** | — | — | — | — | — | — | — |
| **Cytidine-5′-diphosphate (CDP)** | C₉H₁₅N₃O₁₁P₂ | 403.018 | reject | | | | |
| **No confident <50 ppm canonical metabolite at 432.090** | | | | | | | |

Honestly, **432.090 Da does not cleanly match a common HMDB neutral**. If the +2 satellite is really ³⁴S, plausible S-containing candidates near 432.09 include nothing canonical I can confidently name. **I'd flag this family as unidentified pending exact-mass refinement and MS/MS.** The dimer at 862.09 implies high local concentration → likely an abundant small molecule, possibly a **matrix cluster** (DHB, 9-AA, CHCA derivatives can produce clusters in this range — strongly consider MALDI matrix adduct/cluster). Level 4–5.

### Family B (parent = 416.1162 Da)

| Candidate | Formula | Theor. | ppm | Notes | Level | Context |
|---|---|---|---|---|---|---|
| **AMP + something** | — | — | — | doesn't match | — | — |
| **5′-Methylthioadenosine-related** | — | — | — | — | — | — |
| **Glutathione (GSH) adduct?** GSH = 307.0838; 416 − 307 = 109 (not a clean adduct) | — | — | — | — | — | — |
| **No confident match** | | | | If Family B is Family A − O, both are likely the same chemical class (unidentified). | 4 | — |

### Family C (parent = 470.0458 Da; sulfur suggested by m/z accuracy + bromine/chlorine ruled out by isotope spacing)

| Candidate | Formula | Theor. | ppm | HMDB | Justification | Level | Context |
|---|---|---|---|---|---|---|---|
| **3′-Phosphoadenosine-5′-phosphosulfate (PAPS)** [M-H]⁻ would be 506.0… | reject | | | | | | |
| **GDP** (guanosine-5′-diphosphate) | C₁₀H₁₅N₅O₁₁P₂ | 443.0247 | reject | | | | |
| **ADP** (adenosine-5′-diphosphate) | C₁₀H₁₅N₅O₁₀P₂ | 427.0294 | reject | | | | |
| **Cyclic ADP-ribose fragment / NAD-related** | — | — | — | — | — | — | — |
| **No confident <50 ppm canonical match** | | | | possibly an S-containing nucleotide derivative; flag for follow-up | 4 | — |

### D1 — 351.1495 ([M-H]⁻), parent = 352.157 Da

| Candidate | Formula | Theor. | ppm | HMDB | Justification | Level | Context |
|---|---|---|---|---|---|---|---|
| **N-acetylaspartylglutamate (NAAG)** | C₁₁H₁₆N₂O₈ | 304.0907 | reject | | NAAG would give [M-H]⁻ = 303.08 | | |
| **Glutathione conjugate small** | — | — | — | — | — | — | — |
| **No strong canonical match at neutral 352.157**; possible **dipeptide / acyl-carnitine** | — | — | — | — | — | — | — |

### D2 — 352.1336 ([M-H]⁻), parent = 353.141 Da

| Candidate | Formula | Theor. | ppm | HMDB | Justification | Level | Context |
|---|---|---|---|---|---|---|---|
| **5′-Methylthioadenosine (MTA)** | C₁₁H₁₅N₅O₃S | 297.090 | reject | | | | |
| **Adenosine-related** | — | — | — | — | — | — | — |
| **No confident match** | | | | | | 4 | — |

### Family E — 324.0789 ([M-H]⁻), parent = 325.086 Da

| Candidate | Formula | Theor. | ppm | HMDB | Justification | Level | Context |
|---|---|---|---|---|---|---|---|
| **AMP** (adenosine-5′-monophosphate) | C₁₀H₁₄N₅O₇P | 347.0631 | reject | | | | |
| **GMP fragment / IMP**? IMP = C₁₀H₁₃N₄O₈P = 348.047 | reject | | | | | | |
| **Cytidine-5′-monophosphate (CMP)** | C₉H₁₄N₃O₈P | 323.0519 | 102 ppm off → reject | | | | |
| **Glutathione (γ-Glu-Cys-Gly, GSH)** | C₁₀H₁₇N₃O₆S | 307.0838 | reject | | | | |
| **Uridine-5′-monophosphate (UMP)** | C₉H₁₃N₂O₉P |