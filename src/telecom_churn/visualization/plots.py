from __future__ import annotations

import logging
import os
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from telecom_churn.config import IMAGES_DIR, ensure_dirs
from telecom_churn.etl.extract import extract
from telecom_churn.etl.transform import transform
from telecom_churn.logging_config import setup_logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

COLORS = ["#4C72B0", "#DD8452"]


def _save(fig_name: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / fig_name, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", fig_name)


def generate_plots() -> None:
    setup_logging()
    ensure_dirs()
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["font.size"] = 11
    sns.set_theme(style="whitegrid", palette="muted")

    raw_df = extract()
    raw_df["TotalCharges"] = pd.to_numeric(raw_df["TotalCharges"], errors="coerce")
    raw_df.dropna(subset=["TotalCharges"], inplace=True)
    processed_df = transform(raw_df.copy())

    counts = raw_df["Churn"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(counts.index, counts.values, color=COLORS, edgecolor="white", width=0.5)
    axes[0].set_title("Churn Count", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Count")
    for i, value in enumerate(counts.values):
        axes[0].text(i, value + 30, str(value), ha="center", fontweight="bold")
    axes[1].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=COLORS,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Churn Rate", fontsize=14, fontweight="bold")
    _save("churn_distribution.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"], strict=True):
        for label, color in zip(["No", "Yes"], COLORS, strict=True):
            ax.hist(
                raw_df[raw_df["Churn"] == label][col],
                bins=30,
                alpha=0.6,
                color=color,
                label=f"Churn={label}",
                edgecolor="none",
            )
        ax.set_title(col, fontsize=13, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend()
    plt.suptitle("Numeric Features by Churn", fontsize=15, fontweight="bold", y=1.02)
    _save("numeric_features_by_churn.png")

    churn_corr = (
        processed_df.corr()["Churn"].drop("Churn").sort_values(key=abs, ascending=False)
    )
    bar_colors = ["#DD8452" if v > 0 else "#4C72B0" for v in churn_corr.values]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(churn_corr.index, churn_corr.values, color=bar_colors, edgecolor="white")
    ax.set_xlabel("Pearson Correlation with Churn")
    ax.set_title("Feature Correlation with Churn", fontsize=14, fontweight="bold")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(
        handles=[
            mpatches.Patch(color="#DD8452", label="Positive"),
            mpatches.Patch(color="#4C72B0", label="Negative"),
        ]
    )
    _save("feature_correlation_with_churn.png")

    corr = processed_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Matrix", fontsize=15, fontweight="bold")
    _save("correlation_matrix.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    sns.boxplot(
        data=raw_df,
        x="Contract",
        y="MonthlyCharges",
        hue="Churn",
        palette=COLORS,
        ax=axes[0],
    )
    axes[0].set_title("Monthly Charges by Contract & Churn", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=10)
    sns.boxplot(
        data=raw_df,
        x="Contract",
        y="tenure",
        hue="Churn",
        palette=COLORS,
        ax=axes[1],
    )
    axes[1].set_title("Tenure by Contract & Churn", fontsize=12, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=10)
    _save("boxplots_contract.png")

    logger.info("All images saved to %s", IMAGES_DIR.resolve())


def main() -> None:
    generate_plots()


if __name__ == "__main__":
    main()
