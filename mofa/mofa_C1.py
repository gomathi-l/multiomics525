# %%
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import anndata as ad
import h5py

from mofapy2.run.entry_point import entry_point
import mofax as mfx

# %%
data_dir = "datapreprocessed_output"
base = "mouse_V11T16_085_C1"

X = np.load(os.path.join(data_dir, f"{base}_X_rna.npy"))
Y = np.load(os.path.join(data_dir, f"{base}_X_msi.npy"))

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X = scaler_x.fit_transform(X)
Y = scaler_y.fit_transform(Y)

print(X.shape, Y.shape)

# %%
with h5py.File(os.path.join(data_dir, f"{base}_preprocessed.h5ad"), "r+") as f:
    if "base" in f["uns/log1p"]:
        del f["uns/log1p/base"]

adata = ad.read_h5ad(os.path.join(data_dir, f"{base}_preprocessed.h5ad"))
gene_names = adata.var.index.tolist()
msi_names  = adata.uns["msi_features"].tolist()
coords     = np.asarray(adata.obsm["spatial"], dtype=float)

# Drop zero-variance features (MOFA+ can't fit a Gaussian likelihood to them)
gene_mask = X.std(axis=0) > 0
msi_mask  = Y.std(axis=0) > 0

X_fit = X[:, gene_mask]
Y_fit = Y[:, msi_mask]

gene_names_fit = [g for g, k in zip(gene_names, gene_mask) if k]
msi_names_fit  = [m for m, k in zip(msi_names,  msi_mask)  if k]

print(X_fit.shape, Y_fit.shape)

# %%
n_factors = 15
outfile = f"mofa_{base}.hdf5"

ent = entry_point()

ent.set_data_options(
    scale_views=False,        # already standardized above
    scale_groups=False,
    use_float32=True,
)

ent.set_data_matrix(
    data=[[X_fit], [Y_fit]],  # outer: views, inner: groups -> (samples, features)
    likelihoods=["gaussian", "gaussian"],
    views_names=["rna", "msi"],
    groups_names=["group0"],
    features_names=[gene_names_fit, msi_names_fit],
)

ent.set_model_options(
    factors=n_factors,
    spikeslab_weights=True,   # sparse loadings -> no thresholding hack downstream
    ard_weights=True,         # per-view, per-factor relevance -> prunes dead factors
)

ent.set_train_options(
    iter=1000,
    convergence_mode="medium",
    dropR2=0.001,             # drop factors explaining <0.1% variance
    gpu_mode=False,
    seed=42,
    verbose=True,             # watch ELBO converge
)

ent.build()
ent.run()
ent.save(outfile)

# %%
m = mfx.mofa_model(outfile)
shape = m.get_shape()
print("shape (N, D_total):", shape, "  n_factors:", m.nfactors)

Z      = m.get_factors()                  # (N, K)   shared factor scores
W_rna  = m.get_weights(views="rna")       # (D_rna, K)  gene loadings
W_msi  = m.get_weights(views="msi")       # (D_msi, K)  msi loadings

K = Z.shape[1]

# %%
r2 = m.get_r2()
r2_pivot = (
    r2.pivot_table(index="Factor", columns="View", values="R2")
      .fillna(0)
)
r2_pivot["joint"] = r2_pivot.min(axis=1)
r2_pivot = r2_pivot.sort_values("joint", ascending=False)
print(r2_pivot)

# %%
df = pd.DataFrame({
    "x": coords[:, 0],
    "y": coords[:, 1],
})
for i in range(K):
    df[f"factor_{i}"] = Z[:, i]

n_cols = 4
n_rows = (K + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
for i in range(K):
    ax  = axes.flat[i]
    col = df[f"factor_{i}"]
    sc  = ax.scatter(df["x"], df["y"], c=col, cmap="RdBu_r",
                     vmin=np.percentile(col, 5), vmax=np.percentile(col, 95), s=8)
    plt.colorbar(sc, ax=ax)
    ax.set_title(f"Factor {i}")
    ax.set_aspect("equal")
    ax.axis("off")
for j in range(K, n_rows * n_cols):
    axes.flat[j].axis("off")

plt.tight_layout()
plt.savefig(f"mofa_factors_{base}.png", dpi=150)
plt.show()

# %%
gene_corrs_all = {}
msi_corrs_all  = {}

for comp in range(K):
    gene_corrs_all[comp] = (
        pd.DataFrame({"gene": gene_names_fit, "w": W_rna[:, comp], "component": comp})
          .assign(w_abs=lambda d: d["w"].abs())
          .sort_values("w_abs", ascending=False)
    )
    msi_corrs_all[comp] = (
        pd.DataFrame({"msi": msi_names_fit, "w": W_msi[:, comp], "component": comp})
          .assign(w_abs=lambda d: d["w"].abs())
          .sort_values("w_abs", ascending=False)
    )

# %%
# Pick the factor that explains the most joint variance across both views
best_comp = int(r2_pivot.index[0].replace("Factor", "")) - 1  # MOFA uses 1-indexed names

print("best_comp:", best_comp)

# Spike-and-slab makes loadings sparse; take the top-N by |w| rather than a fixed threshold
top_n_genes = 200
top_n_msi   = 50

sig_genes = gene_corrs_all[best_comp].head(top_n_genes)
sig_genes = sig_genes[sig_genes["w_abs"] > 0]["gene"].tolist()

sig_msi   = msi_corrs_all[best_comp].head(top_n_msi)
sig_msi   = sig_msi[sig_msi["w_abs"] > 0]["msi"].tolist()

print("Sig genes:", len(sig_genes))
print("Sig msi:",   len(sig_msi))

pairs = pd.DataFrame([
    {"gene": g, "msi": m, "component": best_comp,
     "r": pearsonr(X[:, gene_names.index(g)], Y[:, msi_names.index(m)])[0]}
    for g in sig_genes for m in sig_msi
]).assign(r_abs=lambda d: d["r"].abs()).sort_values("r_abs", ascending=False)

print(pairs.head(20)[["gene", "msi", "r"]].to_string(index=False))
pairs.to_csv(f"gene_metabolite_pairs_mofa_{base}.csv", index=False)