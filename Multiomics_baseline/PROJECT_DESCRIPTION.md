# Option 9: Spatial Multi-Omics Integration in Disease Pathology

---

## 1. Background

Recent advances in spatial omics technologies allow researchers to measure thousands of molecular features while preserving their spatial location within tissue. This project leverages two complementary platforms:

- **10x Genomics Visium** (spatial transcriptomics): captures genome-wide gene expression (~18,000-32,000 genes) at ~55 um spot resolution with full tissue histology.
- **Mass Spectrometry Imaging (MSI)**: maps the spatial distribution of biomolecules (metabolites, neurotransmitters, lipids) at comparable or finer resolution. Two sub-types appear in this project: **MALDI-MSI** (matrix-assisted laser desorption/ionization) and **DESI-MSI** (desorption electrospray ionization).

Visium reveals the active transcriptomic landscape -- *what genes are being expressed*. MSI reveals the metabolic phenotype -- *what small molecules and lipids are present*. Computationally integrating these modalities is a significant challenge because the measurements often come from different spatial grids, resolutions, and coordinate systems (and may even be performed on consecutive rather than identical tissue sections).

Students will tackle the challenge of integrating these modalities to understand molecular mechanisms in complex diseases. The two core goals are: (1) **cross-modal molecular correspondence** -- discovering which genes and metabolites are functionally related by analyzing their spatial co-localization patterns, and (2) **disease biomarker discovery** -- identifying genes and/or metabolites whose spatial patterns are specifically altered in disease.

---

## 2. Proposed Solutions

### Direction A: Single-Modality Biomarker Discovery (Recommended starting point)

Apply deep learning to analyze spatial patterns in either Visium or MSI data to identify structural domains or biomolecules whose spatial patterns distinguish diseased tissue from controls.

**Approach:**
- **Graph Neural Networks (GNNs)** on Visium data: Spots form a natural spatial graph. Use graph autoencoders (GCN/GAT/GraphSAGE encoders + reconstruction decoder) to learn spot embeddings, then cluster to identify spatial domains. Compare clusters between diseased and control tissue to find disease-associated regions. Compare against methods like SpaGCN and STAGATE.
- **CNNs on spatial images**: Convert spatial data to 2D image grids and train CNNs for unsupervised feature learning or classification. For Visium, grid construction is non-trivial (spots are on a hexagonal lattice) — students must choose a strategy (binning, interpolation, or Gaussian smoothing). MSI data is already on a Cartesian grid and is more directly amenable to CNNs.
- **Transformers**: Recent spatial transcriptomics work has applied transformer architectures (e.g., spatial attention) to learn long-range dependencies in the tissue.
- **Biomarker extraction**: Use attention weights, gradient-based saliency (Grad-CAM), or feature importance (e.g., SHAP) to identify which genes/metabolites and which spatial regions drive the model's predictions.

### Direction B: Cross-Modal Correspondence & Coregistration (Exploratory)

**(B1) Unsupervised landmark detection for coregistration.** When Visium and MSI are measured on consecutive (not identical) sections, their coordinate systems must be aligned. Traditional coregistration requires manual landmark selection. Train a CNN or other computer-vision model to detect corresponding tissue landmarks automatically across the two modalities. The MAGPIE pipeline (see Baselines) currently relies on manual landmarks -- students could build a DL replacement.

**(B2) Gene-metabolite correspondence learning.** Given coregistered Visium and MSI data, learn which genes and metabolites are functionally related. Methods to explore:
- *Cross-modal prediction*: MLP or GNN that predicts metabolite profiles from gene expression (or vice versa). Feature importance reveals the strongest gene-metabolite correspondences.
- *Multi-modal autoencoder*: Separate encoders for each modality mapping to a shared latent space. Features that map to similar latent representations are likely functionally coupled.
- *Contrastive learning*: Treat spatially co-located gene/metabolite profiles as positive pairs, non-colocated as negatives (InfoNCE loss).
- *Deep Canonical Correlation Analysis (Deep CCA)*: Learn maximally correlated nonlinear projections between gene expression and metabolite features.

**Data-limitation caveat:** Each dataset has a small number of samples (3-8). Deep learning gains over simple linear baselines may be modest. Interpret results cautiously.

### Direction C: Integrated Multi-Omics (Most Ambitious)

Combine Directions A and B: build joint multi-modal embeddings of both Visium and MSI data, then analyze which dimensions separate disease microenvironments from normal tissue. Alternatively, find specific gene-metabolite pairs whose spatial relationships change under disease/injury conditions (e.g., a gene-metabolite pair spatially co-localized in healthy tissue but decoupled in disease).

---

## 3. Baseline

**Baseline-to-direction mapping:**

| Baseline | Direction A | Direction B | Direction C |
|----------|:-----------:|:-----------:|:-----------:|
| 1. Scanpy/Squidpy pipeline | ✓ | ✓ | ✓ |
| 2. Graph Autoencoder | ✓ | | ✓ |
| 3. Sample-level Pearson correlation | | ✓ | ✓ |
| 4. MAGPIE coregistration (external) | | ✓ (B1/B2) | ✓ |

### Baseline 1: Scanpy/Squidpy Spatial Analysis Pipeline (non-DL)

Standard preprocessing, QC filtering, normalization, Leiden clustering, UMAP visualization, and Moran's I spatial autocorrelation. This is both a data-loading reference and a non-DL baseline to compare against. Runnable on all three datasets. See `notebooks/01_data_exploration.ipynb`.

### Baseline 2: Graph Autoencoder for Spatial Embedding

A GNN baseline that learns spatial embeddings from a single Visium sample. Uses a GCN encoder + linear reconstruction decoder, then KMeans on the embeddings to identify spatial domains. See `notebooks/02_gnn_spatial_clustering.ipynb`. 

### Baseline 3: Sample-Level Pearson Correlation (non-DL, pseudocode)

For Direction B: for each sample, compute the mean expression of each gene and the mean intensity of each metabolite; correlate across all samples. Provided as pseudocode (not a runnable notebook). Simple, linear, and serves as the lower bound DL methods should beat.

```python
import anndata as ad
import numpy as np
from scipy.stats import pearsonr

# Example: compute sample-level mean features, then correlate
# visium_means[s] = mean expression per gene for sample s (shape: n_genes)
# msi_means[s]    = mean intensity per m/z feature for sample s (shape: n_mz)
#
# For the SMA brain dataset (6 samples), n_samples = 6.
# For the bleomycin lung dataset (8 samples), n_samples = 8.
# For the breast cancer dataset (3 samples), n_samples = 3.

n_samples = len(visium_means)
correlations = np.zeros((n_genes, n_mz))
for i in range(n_genes):
    for j in range(n_mz):
        gene_vals  = [visium_means[s][i] for s in range(n_samples)]
        lipid_vals = [msi_means[s][j]    for s in range(n_samples)]
        r, p = pearsonr(gene_vals, lipid_vals)
        correlations[i, j] = r

# Find top correlated gene-metabolite pairs
top_pairs = np.dstack(np.unravel_index(
    np.argsort(-np.abs(correlations).ravel()), correlations.shape))[0][:20]
```

This baseline is intentionally simple. Deep learning may be explored to incorporate nonlinear structure or richer spatial summaries, but with only 3-8 paired samples per dataset, improvements over linear baselines may be modest and should be interpreted cautiously.

### Baseline 4: MAGPIE Coregistration (External Tool)

For Direction B1 specifically, the published **MAGPIE** pipeline ([Core-Bioinformatics/magpie](https://github.com/Core-Bioinformatics/magpie)) is the reference coregistration tool. It is Python+Snakemake-based, fully open-source, and produces SpaceRanger-compatible outputs that can be loaded directly with `scanpy.read_visium()`. The same publication generated all three datasets in this project (see Dataset section). Students pursuing B1 can either run MAGPIE to get aligned data for downstream modeling, or attempt to replace MAGPIE's manual landmark-selection step with a deep learning approach.

Reference publication: Williams et al., "Spatially resolved integrative analysis of transcriptomic and metabolomic changes in tissue injury studies," *Nature Communications* 17, 205 (2026).

---

## 4. Dataset

All three datasets are distributed as a single pre-processed package by the MAGPIE authors, making setup straightforward.

### Download (1.34 GB)

The datasets are available from Zenodo as `magpie_inputs.zip`:

```bash
# Download (1.34 GB)
curl -L -o magpie_inputs.zip \
    "https://zenodo.org/records/17789448/files/magpie_inputs.zip?download=1"

# Extract
unzip magpie_inputs.zip
```

Or visit https://zenodo.org/records/17789448 and download `magpie_inputs.zip` (license: CC BY 4.0).

### Dataset 1: Parkinson's Disease (6-OHDA mouse brain + human brain)

- **Location in archive:** `magpie_inputs/sma_vicari_brain/`
- **Samples:** 6 total (3 mouse, 3 human)
  - Mouse (6-OHDA unilateral lesion): `mouse_V11L12-038_B1`, `mouse_V11L12-038_D1`, `mouse_V11T16-085_C1`
  - Human (postmortem PD patient): `human_V11T17-102_A1`, `human_V11T17-102_B1`, `human_V11T17-102_D1`
- **Modalities:** Visium + MALDI-MSI, **same tissue section** (SMA protocol: Vicari et al. 2024, *Nat Biotechnol*)
- **Per sample (Visium):** ~32,000 genes x 2,900-8,400 spots
- **Per sample (MSI):** 1,538-3,658 m/z features x 4,000-8,400 pixels
- **Biological task:** Compare lesioned vs. intact hemispheres; map dopamine and related neurotransmitters/metabolites alongside gene expression.

### Dataset 2: Pulmonary Fibrosis (bleomycin mouse lung)

- **Location in archive:** `magpie_inputs/bleo_mouse_lung/`
- **Samples:** 8 total (6 bleomycin d21, 2 control d21)
- **Modalities:** Visium + DESI-MSI, **consecutive sections** (generated by MAGPIE authors)
- **Per sample (Visium):** ~32,000 genes x ~4,000-4,500 spots
- **Per sample (MSI):** 2,634 m/z features x 9,600-18,700 pixels
- **Biological task:** Map extracellular matrix genes, immune markers, and metabolites (e.g., lactate) in fibrotic scars; compare diseased vs. control tissue.

### Dataset 3: Human Breast Cancer

- **Location in archive:** `magpie_inputs/desium_godfrey_breast/`
- **Samples:** 3 total (`BC_1_515`, `BC_1_525`, `BC_2_515`)
- **Modalities:** Visium (CytAssist) + DESI-MSI, **same tissue section** (Godfrey et al. 2025)
- **Per sample (Visium):** ~18,000 genes x 2,700-4,000 spots
- **Per sample (MSI):** 589 m/z features x 2,700-4,000 pixels
- **Biological task:** Characterize spatial relationships between cancer epithelial cells, the tumor core (containing necrotic ceramide species), and surrounding stroma.

### Directory structure (all three datasets follow this layout)

```
magpie_inputs/<dataset>/<sample>/
├── visium/
│   ├── filtered_feature_bc_matrix.h5    # Load with sc.read_visium()
│   └── spatial/
│       ├── tissue_positions_list.csv     # or tissue_positions.csv for CytAssist
│       ├── tissue_hires_image.png
│       ├── tissue_lowres_image.png
│       ├── scalefactors_json.json
│       ├── aligned_fiducials.jpg
│       └── detected_tissue_image.jpg
└── msi/
    ├── MSI_intensities.csv   # rows=pixels, cols=m/z features (first col = spot_id)
    ├── MSI_metadata.csv      # spot_id, x, y per pixel
    └── MSI_HE.jpg            # optional, only in bleo_mouse_lung
```

See `notebooks/01_data_exploration.ipynb` for loading code.

### Practical notes

- Total disk space needed: ~3 GB (1.34 GB zip + ~2.7 GB extracted).
- After loading Visium data, always call `adata.var_names_make_unique()`.
- A GPU with 8+ GB VRAM is recommended for GNN/CNN training. Google Colab Pro or university clusters work well.
- For the SMA brain dataset, Visium and MSI are measured on the **same section** so their coordinate systems must still be aligned (the raw coords differ by several orders of magnitude), but coregistration is tractable. For the bleomycin lung dataset, sections are **consecutive**, making coregistration more challenging.

---

## 5. Using MAGPIE for Coregistration (Optional)

If your project requires spatially aligned multi-modal data (Direction B2 or C), the easiest path is to run MAGPIE on the data you downloaded.

### Installation

```bash
git clone https://github.com/Core-Bioinformatics/magpie.git
cd magpie/snakemake
conda env create -f magpie_environment.yml
conda activate magpie
```

### Running the pipeline

MAGPIE expects input in the exact same directory structure that `magpie_inputs.zip` provides. Example for one SMA mouse brain sample:

```bash
# 1. Copy or symlink the sample into magpie/snakemake/input/
mkdir -p input
ln -s /path/to/magpie_inputs/sma_vicari_brain/mouse_V11L12-038_D1 \
      input/mouse_V11L12-038_D1

# 2. Select landmarks interactively (required)
shiny run magpie_shiny_app.py
# -> opens browser; click ~10 corresponding points on MSI and H&E images;
#    click "Download landmarks" to save input/mouse_V11L12-038_D1/landmarks_noHE.csv

# 3. Run Snakemake pipeline (~4 minutes per sample)
snakemake --cores 1
```

### Output

MAGPIE produces `output/<sample>/spaceranger_aggregated/` in standard SpaceRanger format:

```python
import scanpy as sc
msi_aligned = sc.read_visium("output/mouse_V11L12-038_D1/spaceranger_aggregated/")
# MSI pixels are now aligned to Visium spots; each spot has an aggregated m/z profile.
```

This allows seamless downstream analysis with the same scanpy/squidpy tools used for Visium.

### Notes for students

- The Shiny landmark step is manual. If you want a headless pipeline (e.g., for Direction B1), you must supply `landmarks_noHE.csv` yourself (columns: `X_left, Y_left, X_right, Y_right`, ~10 rows).
- MAGPIE's tutorial at https://core-bioinformatics.github.io/magpie/tutorial/SMA_tutorial.html walks through the SMA mouse sample `V11L12-038_D1` specifically.
- For Direction B1 (unsupervised landmark detection), your deep learning model would *replace* the Shiny step.

---

## 6. References

[1] Vicari, M., Mirzazadeh, R., Nilsson, A. et al. "Spatial multimodal analysis of transcriptomes and metabolomes in tissues." *Nature Biotechnology* 42, 1046-1050 (2024). -- Source of the PD brain dataset (SMA protocol).

[2] Williams, E.C., Franzén, L., Olsson Lindvall, M. et al. "Spatially resolved integrative analysis of transcriptomic and metabolomic changes in tissue injury studies." *Nature Communications* 17, 205 (2026). -- Source of the bleomycin lung dataset and the MAGPIE coregistration pipeline.

[3] Zuo, C. et al. "SpatialGlue: Integration of spatial multi-omics data with deep learning." *Nature Methods* (2024). -- Deep learning method for spatial multi-omics integration.

### Supporting references (for methods)

[4] Hu, J. et al. "SpaGCN: Integrating gene expression, spatial location and histology to identify spatial domains and spatially variable genes by graph convolutional network." *Nature Methods* 18 (2021): 1342-1351.

[5] Dong, K. & Zhang, S. "Deciphering spatial domains from spatially resolved transcriptomics with an adaptive graph attention auto-encoder." *Nature Communications* 13 (2022): 1739. *(STAGATE)*

[6] Kipf, T.N. & Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR* (2017).

[7] Velickovic, P. et al. "Graph Attention Networks." *ICLR* (2018).

[8] Kingma, D.P. & Welling, M. "Auto-Encoding Variational Bayes." *ICLR* (2014).

### Software references

[9] Wolf, F.A. et al. "SCANPY: large-scale single-cell gene expression data analysis." *Genome Biology* 19 (2018): 15. -- https://scanpy.readthedocs.io/

[10] Palla, G. et al. "Squidpy: a scalable framework for spatial omics analysis." *Nature Methods* 19 (2022): 171-178. -- https://squidpy.readthedocs.io/

[11] PyTorch Geometric -- https://pytorch-geometric.readthedocs.io/

[12] AnnData -- https://anndata.readthedocs.io/

[13] MAGPIE source code -- https://github.com/Core-Bioinformatics/magpie

