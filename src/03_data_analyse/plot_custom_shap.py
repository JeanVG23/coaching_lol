#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import riotlib as rl

DATASET = rl.DATA / "04_dataset" / "adc_dataset.parquet"
MODEL_DIR = rl.DATA / "05_model"
OUT = rl.DATA / "06_shap"

def main():
    FEATURES = json.loads((MODEL_DIR / "features.json").read_text())
    
    # Load dataset
    df = pd.read_parquet(DATASET)
    ref_mask = df["source"] == "referentiel"
    spad_mask = df["source"].str.startswith("personal:spadzze", na=False)
    
    # Load ensemble SHAP values (computed in shap_analysis.py)
    sv_ref_vals = np.load(OUT / "sv_ensemble.npy")
    sv_spad_vals = np.load(OUT / "spadzze_sv_ensemble.npy")
    
    # Top 10 features based on referentiel absolute mean
    mean_abs = np.abs(sv_ref_vals).mean(axis=0)
    top10_idx = np.argsort(mean_abs)[::-1][:10]
    top10_features = [FEATURES[i] for i in top10_idx]
    
    plot_data = []
    
    for feature_idx in top10_idx:
        feat_name = FEATURES[feature_idx]
        
        # Referentiel points
        # For background density, we use Grandmaster and Challenger
        ranks_to_plot = ["grandmaster", "challenger"]
        for rank in ranks_to_plot:
            rank_mask_in_ref = df[ref_mask]["rank"] == rank
            for val in sv_ref_vals[rank_mask_in_ref, feature_idx]:
                plot_data.append({
                    "Feature": feat_name,
                    "SHAP Value": val,
                    "Rank": rank.capitalize(),
                    "Type": "Ref"
                })
        
        # Spadzze points
        for val in sv_spad_vals[:, feature_idx]:
            plot_data.append({
                "Feature": feat_name,
                "SHAP Value": val,
                "Rank": "Spadzze",
                "Type": "Spadzze"
            })
            
    plot_df = pd.DataFrame(plot_data)
    
    # Create 2 subplots with different heights and separate x-axes
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 9]}, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.1)  # Reduce space between plots
    
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8f9fa"})
    
    top_feature = top10_features[0]
    other_features = top10_features[1:]
    
    ref_df = plot_df[plot_df["Type"] == "Ref"]
    spad_df = plot_df[plot_df["Type"] == "Spadzze"]
    spad_mean = spad_df.groupby("Feature")["SHAP Value"].mean().reset_index()

    # --- TOP AXIS (kda_2v2) ---
    sns.violinplot(
        data=ref_df[ref_df["Feature"] == top_feature], x="SHAP Value", y="Feature", 
        color="#a1c9f4", ax=ax_top, inner=None, linewidth=0, alpha=0.6
    )
    sns.stripplot(
        data=spad_df[spad_df["Feature"] == top_feature], x="SHAP Value", y="Feature", color="#333333",
        jitter=0.1, size=4, marker="D", label="Spadzze (Games)", alpha=0.6, dodge=False, ax=ax_top
    )
    sns.scatterplot(
        data=spad_mean[spad_mean["Feature"] == top_feature], x="SHAP Value", y="Feature", color="#00ffcc",
        marker="*", s=400, label="Spadzze (Moyenne)", edgecolor="black", zorder=100, linewidth=1.5, ax=ax_top
    )
    ax_top.axvline(0, color="#333333", linestyle="--", linewidth=1.5, zorder=0)
    ax_top.set_xlabel("")
    ax_top.set_ylabel("")
    ax_top.set_title("Top 10 Features (Ensemble SHAP) : Objectif GM/Chall vs Spadzze", fontsize=18, fontweight="bold", pad=20)
    
    # --- BOTTOM AXIS (Other 9 features) ---
    sns.violinplot(
        data=ref_df[ref_df["Feature"] != top_feature], x="SHAP Value", y="Feature", 
        color="#a1c9f4", order=other_features, ax=ax_bottom, inner=None, linewidth=0, alpha=0.6
    )
    sns.stripplot(
        data=spad_df[spad_df["Feature"] != top_feature], x="SHAP Value", y="Feature", color="#333333",
        order=other_features, jitter=0.1, size=4, marker="D", alpha=0.6, dodge=False, ax=ax_bottom
    )
    sns.scatterplot(
        data=spad_mean[spad_mean["Feature"] != top_feature], x="SHAP Value", y="Feature", color="#00ffcc",
        marker="*", s=400, edgecolor="black", zorder=100, linewidth=1.5, ax=ax_bottom
    )
    ax_bottom.axvline(0, color="#333333", linestyle="--", linewidth=1.5, zorder=0)
    ax_bottom.set_xlabel("Valeur SHAP (Impact : < 0 pousse vers Low Elo, > 0 pousse vers High Elo)", fontsize=14, labelpad=10)
    ax_bottom.set_ylabel("Features", fontsize=14)
    
    # Legend fixing (take handles from top axis)
    handles, labels = ax_top.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_top.legend(by_label.values(), by_label.keys(), title="Légende", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, title_fontsize=13)
    
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    img_path = OUT / "custom_shap_spadzze.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Graphique généré dans {img_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
