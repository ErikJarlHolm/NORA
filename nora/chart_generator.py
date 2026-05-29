"""
Chart Generator – lag grafer, kakediagram og andre visualiseringer som PNG.

Tilgjengelig som verktøy for NORA-agenten slik at den kan visualisere
tallmateriale og lagre resultatet som PNG-bilder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PNG rendering
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

# Standard output folder
DEFAULT_OUTPUT_FOLDER = Path(
    r"C:\Users\erikholm\OneDrive - Atea\Documents\Kunder\Atea AI Norge\Agent tallknusing"
)


def _ensure_output_folder(folder: Path) -> Path:
    """Opprett mappen hvis den ikke finnes."""
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _save_figure(fig: plt.Figure, filename: str, output_folder: Optional[str] = None) -> str:
    """Lagre figuren som PNG og returner full filsti."""
    folder = Path(output_folder) if output_folder else DEFAULT_OUTPUT_FOLDER
    _ensure_output_folder(folder)

    if not filename.endswith(".png"):
        filename += ".png"

    filepath = folder / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Graf lagret: %s", filepath)
    return str(filepath)


def _labels_have_year_prefix(labels: list[str]) -> bool:
    """Sjekk om labels ser ut som datoer med årstalprefix (f.eks. '2019-01')."""
    if not labels:
        return False
    try:
        years = set(lbl[:4] for lbl in labels)
        return len(years) > 1 and all(y.isdigit() for y in years)
    except (IndexError, TypeError):
        return False


def create_bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "Stolpediagram",
    xlabel: str = "",
    ylabel: str = "",
    filename: str = "bar_chart.png",
    output_folder: Optional[str] = None,
    color: Optional[str] = None,
    color_by_group: bool = False,
) -> str:
    """Lag et stolpediagram og lagre som PNG.

    Hvis color_by_group=True eller labels inneholder årstall-prefiks,
    fargelegges stolpene automatisk per gruppe/år.
    """
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.2), 6))

    if color_by_group or (color is None and _labels_have_year_prefix(labels)):
        groups = [lbl[:4] for lbl in labels]
        unique_groups = sorted(set(groups), key=groups.index)
        palette = plt.cm.tab10(np.linspace(0, 1, min(len(unique_groups), 10)))
        group_colors = {g: palette[i % 10] for i, g in enumerate(unique_groups)}
        bar_colors = [group_colors[g] for g in groups]
        ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.5)

        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=group_colors[g], label=g) for g in unique_groups]
        ax.legend(handles=patches, title="År", loc="upper right", framealpha=0.9)
    else:
        ax.bar(labels, values, color=color or "#2196F3", edgecolor="white", linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=90 if len(labels) > 12 else (45 if len(labels) > 6 else 0))
    fig.tight_layout()

    return _save_figure(fig, filename, output_folder)


def create_line_chart(
    x: list[float | str],
    y: Optional[list[float]] = None,
    title: str = "Linjediagram",
    xlabel: str = "",
    ylabel: str = "",
    filename: str = "line_chart.png",
    output_folder: Optional[str] = None,
    series_labels: Optional[list[str]] = None,
    y_series: Optional[list[list[float]]] = None,
) -> str:
    """
    Lag et linjediagram og lagre som PNG.

    For flere linjer, bruk y_series (liste av y-verdier per serie)
    og series_labels for å navngi dem. Enkeltlinje bruker y direkte.
    None-verdier i y_series hoppes over (brudd i linjen).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if y_series and series_labels:
        colors = plt.cm.tab10(np.linspace(0, 1, len(y_series)))
        for i, (ys, label) in enumerate(zip(y_series, series_labels)):
            # Filter out None values while keeping x alignment
            x_vals = []
            y_vals = []
            for j, val in enumerate(ys):
                if val is not None and j < len(x):
                    x_vals.append(x[j])
                    y_vals.append(val)
            if y_vals:
                ax.plot(x_vals, y_vals, marker="o", markersize=4, label=label, color=colors[i])
        ax.legend(framealpha=0.9)
    elif y:
        ax.plot(x, y, marker="o", markersize=4, color="#1976D2", linewidth=2)
    else:
        raise ValueError("Enten 'y' eller 'y_series' + 'series_labels' må oppgis.")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return _save_figure(fig, filename, output_folder)


def create_pie_chart(
    labels: list[str],
    values: list[float],
    title: str = "Kakediagram",
    filename: str = "pie_chart.png",
    output_folder: Optional[str] = None,
    explode_largest: bool = False,
) -> str:
    """Lag et kakediagram og lagre som PNG. Returnerer filstien."""
    fig, ax = plt.subplots(figsize=(8, 8))

    explode = None
    if explode_largest:
        max_idx = values.index(max(values))
        explode = [0.05 if i == max_idx else 0 for i in range(len(values))]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        explode=explode,
        colors=plt.cm.Set3(np.linspace(0, 1, len(values))),
        textprops={"fontsize": 10},
    )

    for autotext in autotexts:
        autotext.set_fontweight("bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()

    return _save_figure(fig, filename, output_folder)


def create_scatter_plot(
    x: list[float],
    y: list[float],
    title: str = "Spredningsdiagram",
    xlabel: str = "",
    ylabel: str = "",
    filename: str = "scatter_plot.png",
    output_folder: Optional[str] = None,
    add_trendline: bool = False,
) -> str:
    """Lag et spredningsdiagram og lagre som PNG. Returnerer filstien."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color="#FF5722", alpha=0.7, edgecolors="white", linewidth=0.5, s=60)

    if add_trendline and len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p(x_line), "--", color="#333", linewidth=1.5, alpha=0.7, label="Trendlinje")
        ax.legend()

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return _save_figure(fig, filename, output_folder)


def create_horizontal_bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "Horisontalt stolpediagram",
    xlabel: str = "",
    ylabel: str = "",
    filename: str = "hbar_chart.png",
    output_folder: Optional[str] = None,
    color: Optional[str] = None,
    color_by_group: bool = False,
) -> str:
    """Lag et horisontalt stolpediagram og lagre som PNG.

    Hvis color_by_group=True, fargelegges stolpene automatisk basert på
    gruppeprefikset i labels (f.eks. årstall fra '2019-01' → '2019').
    """
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4)))

    if color_by_group or (color is None and _labels_have_year_prefix(labels)):
        # Auto-detect groups from label prefix (year)
        groups = [lbl[:4] for lbl in labels]
        unique_groups = sorted(set(groups), key=groups.index)
        palette = plt.cm.tab10(np.linspace(0, 1, min(len(unique_groups), 10)))
        group_colors = {g: palette[i % 10] for i, g in enumerate(unique_groups)}
        bar_colors = [group_colors[g] for g in groups]
        ax.barh(labels, values, color=bar_colors, edgecolor="white", linewidth=0.5)

        # Legend
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=group_colors[g], label=g) for g in unique_groups]
        ax.legend(handles=patches, title="År", loc="lower right", framealpha=0.9)
    else:
        ax.barh(labels, values, color=color or "#4CAF50", edgecolor="white", linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    return _save_figure(fig, filename, output_folder)


def create_stacked_bar_chart(
    labels: list[str],
    series_data: list[list[float]],
    series_labels: list[str],
    title: str = "Stablet stolpediagram",
    xlabel: str = "",
    ylabel: str = "",
    filename: str = "stacked_bar_chart.png",
    output_folder: Optional[str] = None,
) -> str:
    """Lag et stablet stolpediagram og lagre som PNG."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(series_data)))

    bottoms = np.zeros(len(labels))
    for i, (data, label) in enumerate(zip(series_data, series_labels)):
        ax.bar(labels, data, bottom=bottoms, label=label, color=colors[i], edgecolor="white", linewidth=0.5)
        bottoms += np.array(data)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=45 if len(labels) > 6 else 0)
    fig.tight_layout()

    return _save_figure(fig, filename, output_folder)


def create_area_chart(
    x: list[float | str],
    y: list[float],
    title: str = "Arealdiagram",
    xlabel: str = "",
    ylabel: str = "",
    filename: str = "area_chart.png",
    output_folder: Optional[str] = None,
) -> str:
    """Lag et arealdiagram og lagre som PNG."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(range(len(y)), y, alpha=0.4, color="#2196F3")
    ax.plot(range(len(y)), y, color="#1565C0", linewidth=2)

    if all(isinstance(v, str) for v in x):
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=45 if len(x) > 6 else 0)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return _save_figure(fig, filename, output_folder)
