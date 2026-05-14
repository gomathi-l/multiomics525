# %%
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import anndata as ad
import h5py

import mofax as mfx

# %%
# Config
data_dir = "datapreprocessed_output"
#base     = "mouse_V11L12_038_D1"            
base = sys.argv[1] if len(sys.argv) > 1 else "mouse_V11L12_038_D1"
mofa_hdf5 = f"mofa_{base}.hdf5"

out_dir   = "."
out_genes = os.path.join(out_dir, f"gene_loadings_mofa_ALL_FACTORS_{base}.csv")
out_msi   = os.path.join(out_dir, f"msi_loadings_mofa_ALL_FACTORS_{base}.csv")
out_pairs = os.path.join(out_dir, f"gene_metabolite_pairs_mofa_ALL_FACTORS_{base}.csv")

top_n_genes = 200    # top genes per factor (by |W|)
top_n_msi   = 50     # top m/z per factor (by |W|)

# %%
# Data loader
X = np.load(os.path.join(data_dir, f"{base}_X_rna.npy"))
Y = np.load(os.path.join(data_dir, f"{base}_X_msi.npy"))

scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X)
Y = scaler_y.fit_transform(Y)

print("X (RNA):", X.shape, "  Y (MSI):", Y.shape)

# Strip log1p base if present (anndata can choke on it)
with h5py.File(os.path.join(data_dir, f"{base}_preprocessed.h5ad"), "r+") as f:
    if "base" in f["uns/log1p"]:
        del f["uns/log1p/base"]

adata      = ad.read_h5ad(os.path.join(data_dir, f"{base}_preprocessed.h5ad"))
gene_names = adata.var.index.tolist()
msi_names  = adata.uns["msi_features"].tolist()
coords     = np.asarray(adata.obsm["spatial"], dtype=float)

# Drop zero-variance features so the column order matches what MOFA was trained on
gene_mask = X.std(axis=0) > 0
msi_mask  = Y.std(axis=0) > 0

X_fit = X[:, gene_mask]
Y_fit = Y[:, msi_mask]

gene_names_fit = [g for g, k in zip(gene_names, gene_mask) if k]
msi_names_fit  = [m for m, k in zip(msi_names,  msi_mask)  if k]

print("X_fit:", X_fit.shape, "  Y_fit:", Y_fit.shape)
print(f"  kept {len(gene_names_fit)}/{len(gene_names)} genes,",
      f"{len(msi_names_fit)}/{len(msi_names)} m/z")

# %%
# Load trained MOFA model

m = mfx.mofa_model(mofa_hdf5)
shape = m.get_shape()
print(f"  shape (N, D_total): {shape}   n_factors: {m.nfactors}")

Z     = m.get_factors()                # (N, K)
W_rna = m.get_weights(views="rna")     # (D_rna, K)
W_msi = m.get_weights(views="msi")     # (D_msi, K)
K = Z.shape[1]
print(f"  Z: {Z.shape}   W_rna: {W_rna.shape}   W_msi: {W_msi.shape}")

# Sanity checks (loaders must match what MOFA was trained on)
assert W_rna.shape[0] == len(gene_names_fit), \
    f"RNA weights have {W_rna.shape[0]} features, gene_names_fit has {len(gene_names_fit)} -- " \
    "data loader doesn't match the trained model"
assert W_msi.shape[0] == len(msi_names_fit), \
    f"MSI weights have {W_msi.shape[0]} features, msi_names_fit has {len(msi_names_fit)} -- " \
    "data loader doesn't match the trained model"
assert Z.shape[0] == X_fit.shape[0] == Y_fit.shape[0]

# %%
# R^2 table (which factors explain variance in each view)

r2 = m.get_r2()
r2_pivot = (
    r2.pivot_table(index="Factor", columns="View", values="R2")
      .fillna(0)
)
r2_pivot["joint"] = r2_pivot.min(axis=1)
r2_pivot = r2_pivot.sort_values("joint", ascending=False)
print("\nR^2 per factor per view (sorted by joint R^2):")
print(r2_pivot)

# %%
# Per-factor gene and m/z loading tables

gene_rows = []
msi_rows  = []
for k in range(K):
    gene_rows.append(
        pd.DataFrame({"gene": gene_names_fit, "w": W_rna[:, k], "component": k})
          .assign(w_abs=lambda d: d["w"].abs())
          .sort_values("w_abs", ascending=False)
    )
    msi_rows.append(
        pd.DataFrame({"msi": msi_names_fit, "w": W_msi[:, k], "component": k})
          .assign(w_abs=lambda d: d["w"].abs())
          .sort_values("w_abs", ascending=False)
    )

gene_loadings_all = pd.concat(gene_rows, ignore_index=True)
msi_loadings_all  = pd.concat(msi_rows,  ignore_index=True)

os.makedirs(out_dir, exist_ok=True)
gene_loadings_all.to_csv(out_genes, index=False)
msi_loadings_all.to_csv(out_msi,   index=False)
print(f"\nWrote gene loadings ({len(gene_loadings_all):,} rows) -> {out_genes}")
print(f"Wrote msi  loadings ({len(msi_loadings_all):,} rows) -> {out_msi}")

# %%
# Per-factor gene x m/z pair correlations 

all_pairs = []
n_spots = X_fit.shape[0]
print(f"\nComputing pairs for all {K} factors "
      f"(top {top_n_genes} genes x top {top_n_msi} m/z each)")
for k in range(K):
    g_idx = np.argsort(-np.abs(W_rna[:, k]))[:top_n_genes]
    m_idx = np.argsort(-np.abs(W_msi[:, k]))[:top_n_msi]
    # Skip features with exactly zero weight (sparse spike-and-slab)
    g_idx = g_idx[np.abs(W_rna[g_idx, k]) > 0]
    m_idx = m_idx[np.abs(W_msi[m_idx, k]) > 0]
    if len(g_idx) == 0 or len(m_idx) == 0:
        print(f"  factor {k:2d}: no nonzero loadings, skipped")
        continue

    A = X_fit[:, g_idx]                  # (n_spots, ng)
    B = Y_fit[:, m_idx]                  # (n_spots, nm)
    R = (A.T @ B) / n_spots              # (ng, nm) -- Pearson r

    # Flatten to long form
    ng, nm = R.shape
    gi_grid = np.repeat(g_idx, nm)
    mj_grid = np.tile(m_idx, ng)
    r_flat  = R.reshape(-1)
    block = pd.DataFrame({
        "gene":      [gene_names_fit[i] for i in gi_grid],
        "msi":       [msi_names_fit[j]  for j in mj_grid],
        "component": k,
        "r":         r_flat,
        "r_abs":     np.abs(r_flat),
    })
    all_pairs.append(block)

    top = block.nlargest(1, "r_abs").iloc[0]
    print(f"  factor {k:2d}: {ng}x{nm} pairs, "
          f"max |r| = {np.abs(R).max():.3f}  ({top['gene']} x {top['msi']})")

pairs_df = (
    pd.concat(all_pairs, ignore_index=True)
      .sort_values(["component", "r_abs"], ascending=[True, False])
)
pairs_df.to_csv(out_pairs, index=False)
print(f"\nWrote {len(pairs_df):,} pair rows -> {out_pairs}")

# %%
# All
print("\n Per-factor top pair ")
top_per = (
    pairs_df.groupby("component", as_index=False).head(1)
            [["component", "gene", "msi", "r"]]
)
print(top_per.to_string(index=False))

print("\n Top 5 genes per factor (by |W|) ")
top_genes_per = (
    gene_loadings_all.groupby("component", as_index=False)
                     .head(5)[["component", "gene", "w"]]
)
print(top_genes_per.to_string(index=False))

print("\n Top 5 m/z per factor (by |W|) ")
top_msi_per = (
    msi_loadings_all.groupby("component", as_index=False)
                    .head(5)[["component", "msi", "w"]]
)
print(top_msi_per.to_string(index=False))