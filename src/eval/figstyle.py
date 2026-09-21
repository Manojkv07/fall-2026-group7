"""Figure style and vector export.

Every results figure in this project is written as **vector** — SVG plus PDF —
because a raster figure cannot be enlarged, re-typeset, or read cleanly in
print, and ``dpi=300`` on a PNG is still a fixed grid of pixels.

Font handling differs by format on purpose:

* PDF uses ``fonttype 42`` (TrueType), which embeds the font and keeps the text
  as selectable, searchable text.
* SVG uses ``fonttype 'none'``, which leaves text as real ``<text>`` elements so
  the file stays editable in draw.io or Illustrator.

Colors come from the project's validated categorical/diverging palette rather
than matplotlib's defaults. Identity is never carried by color alone: every
point is positioned against a zero line and carries a direct label.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: the instance has no display
import matplotlib.pyplot as plt  # noqa: E402

# --- palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

POSITIVE = "#2a78d6"   # diverging pole: the condition helped
NEGATIVE = "#e34948"   # diverging pole: the condition hurt
INCONCLUSIVE = "#8a8984"  # CI spans zero — no direction claimed


def apply_style() -> None:
    """Recessive axes, thin marks, vector-safe fonts."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,

        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_MUTED,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.color": INK_MUTED,
        "ytick.color": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "grid.color": GRID,
        "grid.linewidth": 0.6,

        "font.size": 10,
        "font.family": "sans-serif",

        # Vector export. Without these, text can be rasterised or lose its
        # embedded font, which is the usual reason a "vector" figure is rejected.
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_vector(fig, stem: str | Path, formats=("svg", "pdf")) -> list[Path]:
    """Write one figure to every vector format and report the paths.

    Deliberately offers no PNG path. A results figure that only exists as a
    raster has to be regenerated to be usable, and regeneration is exactly
    what gets skipped under deadline.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, format=ext, bbox_inches="tight")
        written.append(path)
    return written
