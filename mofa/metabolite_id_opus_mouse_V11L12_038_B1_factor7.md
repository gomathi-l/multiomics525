# Opus annotation -- mouse_V11L12_038_B1, factor 7

- **Model**: `claude-opus-4-7`
- **Input tokens**: 2696
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
- Sample ID: mouse_V11L12_038_B1
- Analysis: MOFA+ multi-omics factor analysis identified 13 factors. This request concerns FACTOR 7, which co-varies between RNA and MSI across spots.

## Top genes loading on this factor (sign indicates direction)
- Gpr88  (w = +0.5341)
- Penk  (w = +0.5332)
- Pde10a  (w = +0.5002)
- Ppp1r1b  (w = +0.4976)
- Pde1b  (w = +0.4812)
- Rasd2  (w = +0.4783)
- Rgs9  (w = +0.4772)
- Arpp21  (w = +0.4137)
- Scn4b  (w = +0.4132)
- Adora2a  (w = +0.4082)
- Pcp4l1  (w = +0.4046)
- Drd2  (w = +0.3924)
- Adcy5  (w = +0.3887)
- Gng7  (w = +0.3876)
- Syndig1l  (w = +0.3680)
- Olfm1  (w = -0.3651)
- Hpca  (w = +0.3549)
- Foxp1  (w = +0.3534)
- Pcp4  (w = +0.3491)
- Dgkb  (w = +0.3412)
- Bcl11b  (w = +0.3361)
- Slc17a7  (w = -0.3310)
- Ppp3ca  (w = +0.3271)
- Drd1  (w = +0.3215)
- Inf2  (w = +0.3208)
- Rxrg  (w = +0.3175)
- Lrrc10b  (w = +0.3146)
- Cck  (w = -0.3110)
- Tac1  (w = +0.3001)
- Sncb  (w = -0.2957)

## Top m/z values loading on this factor (sign indicates direction)
- 364.90382  (w = +0.5919)
- 364.902325  (w = +0.5807)
- 364.90545999999995  (w = +0.5699)
- 348.92984  (w = +0.5657)
- 348.92861500000004  (w = +0.5551)
- 348.93134  (w = +0.5521)
- 366.90242  (w = +0.5334)
- 287.173085  (w = +0.5333)
- 384.99937  (w = +0.5018)
- 540.910025  (w = +0.4766)
- 369.025065  (w = +0.4675)
- 217.9387  (w = +0.4614)
- 837.66747  (w = +0.4610)
- 240.122615  (w = +0.4602)
- 240.123325  (w = +0.4574)
- 863.68026  (w = +0.4495)
- 380.907875  (w = +0.4432)
- 366.89971  (w = +0.4374)
- 240.12396999999999  (w = +0.4197)
- 350.927835  (w = +0.4113)
- 524.9369200000001  (w = +0.4082)
- 203.932345  (w = +0.4036)
- 203.93183499999998  (w = +0.4036)
- 241.12645  (w = +0.4033)
- 203.930065  (w = +0.4012)

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

# Factor 7 Analysis — mouse_V11L12_038_B1

## 1. Brain region / cell type inference

This is a textbook **striatal medium spiny neuron (MSN) signature**, capturing both direct (dMSN: *Drd1, Tac1*) and indirect (iMSN: *Drd2, Adora2a, Penk*) pathway MSNs simultaneously. The pan-MSN markers *Ppp1r1b* (DARPP-32), *Gpr88, Pde10a, Pde1b, Rgs9, Rasd2* (Rhes), *Arpp21, Bcl11b* (Ctip2), *Foxp1*, and *Adcy5* are essentially the canonical striatal cassette. The negative loadings on *Slc17a7* (VGLUT1, cortical glutamatergic) and *Cck* support this — the factor explicitly contrasts striatum vs cortex. Most likely this captures **dorsal + ventral striatum (caudoputamen ± nucleus accumbens)**, with both MSN classes mixed because Visium spots are not single-cell. *Rxrg* and *Pcp4l1* further support striatal identity.

## 2. Adduct family grouping

The most striking feature is dense clustering around 203, 217, 348, 364, 366, 380, 384, 524, 540 Da with spacings characteristic of **Cl, Na-for-H, and isotopologue chemistry** — this is the spectral fingerprint of **inorganic / halide-rich species**, not lipids or neurotransmitters.

### Family A — parent ≈ ~330 Da region (chloride/sodium cluster)
Differences within the m/z list:
- 348.9298 − 364.9038 = −15.99 (O loss? unlikely) 
- 364.9038 − 380.9079 = −16.00 (O / +O pair)
- 348.9298 − 366.9024 = −17.97 → consistent with **Cl(35) vs (Cl37 + something)** but more cleanly:
- 366.9024 − 364.9038 = +1.999 → **37Cl isotopologue** of 364.90
- 350.9278 − 348.9298 = +1.998 → **37Cl isotopologue** of 348.93
- 364.9038 − 348.9298 = +15.974 → **+O** (oxidation) OR swap of one Cl/OH; could also be **NaCl substitution pattern**
- 380.9079 − 364.9038 = +16.004 → another +O
- 384.9994 − 366.9024 = +18.097 (not clean)

**Representative parent ion: 348.9298 [M-H]⁻**, neutral mass **349.9371 Da**
**Representative parent ion: 364.9038 [M-H]⁻**, neutral mass **365.9111 Da** (likely +O of family A or different parent)

| Observed m/z | Δ vs 348.9298 | Assignment |
|---|---|---|
| 348.9286 / 348.9298 / 348.9313 | 0 | [M-H]⁻ (replicates / mass jitter) |
| 350.9278 | +1.998 | 37Cl isotopologue |
| 364.9023 / 364.9038 / 364.9054 | +15.974 | +O oxidation product or co-eluting species |
| 366.8997 / 366.9024 | +1.998 vs 364.90 | 37Cl isotopologue of the +O species |
| 380.9079 | +15.978 vs 364.90 | further +O |
| 384.9994 | +36.07 | possibly +Cl+2H (uncertain) |

### Family B — heavy chlorinated cluster ~524–540 Da
- 540.9100 − 524.9369 = +15.973 → **+O**
- These differ from 348.93 by +175.98 and +191.98 — not a clean adduct relationship, but could be **dimeric / metal cluster** species.

### Family C — low-mass halide cluster ~203–217 Da
- 203.9301 / 203.9318 / 203.9323 (jitter)
- 217.9387 − 203.9323 = +14.006 → **+CH₂ or +N** (ambiguous)
- 203 region is consistent with small halogenated/metal species.

### Family D — clean organic, ~287 Da
- **287.1731 [M-H]⁻**, neutral 288.1804 Da. Singleton, no adduct partners visible.

### Family E — organic ~240 Da
- 240.1226 / 240.1233 / 240.1240 (replicates)
- 241.1265 − 240.1240 = +1.003 → **13C₁ isotopologue** ✓
- Neutral mass **241.1306 Da** (from [M-H]⁻ = 240.1233)

### Family F — lipid range ~837 / 863
- 863.6803 − 837.6675 = +26.013 → **+C₂H₂** (chain elongation, typical PI/PS series)
- Both look like glycerophospholipids [M-H]⁻.

### Singletons / unclear
- 369.0251 — no clean partner
- 384.9994 — possibly Family A satellite

---

## 3. Candidate metabolite assignments

### Family A (348.93, 364.90, 366.90, 380.91, 350.93)

The mass defect is **very low** (348.93 → only 0.93 over the integer mass), implying **high halogen / phosphorus / metal content, low H content**. This is NOT a typical organic metabolite. Combined with the clear ³⁷Cl isotope pattern (+1.998 Da partners), these are most plausibly **inorganic/MALDI matrix-related cluster ions** (e.g., chloride-rich clusters, Au/metal-Cl, or salt clusters from the tissue/matrix), or possibly **iron-sulfur / phosphate-chloride clusters**.

- **No confident biological metabolite match at <50 ppm.** I would not force-fit. Schymanski **Level 5**. **Matches gene context: no** — striatal MSNs don't predict an unusual halogenated metabolite.

### Family B (524.94 / 540.91)
Same low mass defect issue. Likely **matrix/salt cluster** or contaminant. **No good match. Level 5.**

### Family C (203.93, 217.94)
Mass defect again ~−0.07 — inorganic-like. Possible candidates: small Cl/Br/I-containing fragments, or NaCl·H₂O clusters. **No confident metabolite assignment. Level 5.**

### Family D — 287.1731 [M-H]⁻, neutral 288.1804

| Candidate | Formula | Theo. neutral | ppm | Notes |
|---|---|---|---|---|
| **N-arachidonoyl glycine fragment / fatty amide** | C₁₈H₂₄O₃ | 288.1725 | +27 ppm | borderline |
| **Sebacoyl-related dicarboxylate ester** | C₁₇H₂₄N₂O | 288.1838 | −12 ppm | unlikely brain |
| **Decanoyl-glycerol / MAG 10:0** | C₁₃H₂₆O₅ | 262.18 | no | reject |

Honestly, **no high-confidence striatal match.** A C₁₈H₂₄O₃ would be a hydroxy/oxo-fatty acid; not a known striatal marker. **Level 4 (formula only, tentatively C₁₈H₂₄O₃ or C₁₇H₂₄N₂O).** Matches gene context: **no / unclear**.

### Family E — 240.1233 [M-H]⁻, neutral 241.1306 (13C confirmed)

| Candidate | Formula | Theo. neutral | ppm | Justification | Level | Gene match |
|---|---|---|---|---|---|---|
| **Acetyl-aspartyl-glutamate (NAAG) fragment? No — NAAG is 304.** | | | | | | |
| **Deoxyguanosine** (dG) | C₁₀H₁₃N₅O₄ | 267.0967 | no | reject | | |
| **N-acetylaspartylglutamate-related?** | | | | | | |
| **Phosphocholine / GPC fragment** | | | | | | |
| **Dopamine-sulfate / dopamine-glucuronide?** Dopamine-3-O-sulfate C₈H₁₁NO₅S = 233.04, no. | | | | | |
| **Adenosine** C₁₀H₁₃N₅O₄ = 267.10, no. | | | | | |
| **N-acetyl-L-aspartyl-methyl ester / Trp metabolites:** L-kynurenine C₁₀H₁₂N₂O₃ = 208.08, no. | | | | | |
| **Riboflavin-related, FAD fragments:** no fit. | | | | | |

Closest plausible: **C₁₅H₁₇N₃ (241.144)** off by ~50 ppm — reject. **C₁₂H₁₉NO₄ (241.131)** = ~0 ppm fit, e.g., a small acyl-carnitine-like or amino acid conjugate. **Acetylcarnitine** is C₉H₁₇NO₄ (203.116) — no. **Propionylcarnitine** C₁₀H₁₉NO₄ (217.131) — no. **Butyrylcarnitine** C₁₁H₂₁NO₄ (231.147) — no.

**Best tentative: C₁₂H₁₉NO₄, neutral 241.1314, −3 ppm**. Could be a small acylcarnitine or dipeptide. **No high-confidence ID. Level 4.** Gene match: unclear.

### Family F — 837.6675 and 863.6803 [M-H]⁻

These masses are characteristic of **phosphatidylinositols (PI)** or **phosphatidylserines (PS)**:

| Observed [M-H]⁻ | Candidate | Formula (neutral) | Theo. [M-H]⁻ | ppm | Notes |
|---|---|---|---|---|---|
| 837.6675 | **PI 36:4** (e.g., 18:0/18:2 or 16:0/20:4) | C₄₅H₈₃O₁₃P | 837.5499 | far off | reject |
| 837.6675 | **PS 40:6** | C₄₆H₇₈NO₁₀P | 836.5443 | reject | |
| 837.6675 | **PE-Cer / PE 42:1**? | | | | |
| 837.6675 | **PG 42:7** C₄₈H₈₃O₁₀P | 850.57 | reject | |

The mass defect ~0.67 is high — consistent with very saturated long-chain lipid. **PC 40:6** C₄₈H₈₄NO₈P neutral = 833.59 → no. **SM d18:1/24:1** C₄₇H₉₃N₂O₆P = 836.68 → [M-H]⁻ = 835.67, no. 

**Best fit: a sulfatide / ST or HexCer species.** Sulfatide **ST(d18:1/24:1) (sulfo-galactosylceramide C24:1)** = C₄₈H₉₁NO₁₁S, neutral 889.63 — no. **HexCer(d18:1/24:1)** C₄₈H₉₁NO₈ = 809.68 — no.

Try **GalCer / GlcCer + something**: HexCer(d18:1/26:1) C₅₀H₉₅NO₈ neutral 837.71, [M-H]⁻ 836.70 — too low.

**ST(d18:1/18:0)** C₄₂H₈₁NO₁₁S, neutral 807.56 — no.

Honestly the 837.67 / 863.68 pair with Δ = 26.01 (= +C₂H₂) is **classic sulfatide or PI homologous series**. Without tighter mass accuracy I can only say: **Level 4 — likely a sphingolipid or phosphoinositide homologous pair**. Gene