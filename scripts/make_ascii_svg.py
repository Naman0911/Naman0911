"""
make_ascii_svg.py
=================
Orchestrator for the ASCII portrait pipeline.

This script wires together all pipeline stages in sequence:

    Stage 1: prep_photo.py    -- preprocess the source image
    Stage 2: edge_detector.py -- compute edge map + gradient direction
    Stage 3: ascii_engine.py  -- convert to ASCII character grid
    Stage 4: svg_renderer.py  -- render animated SVG

Run this script to regenerate output/avi-ascii.svg from scratch.

Usage:
    python scripts/make_ascii_svg.py
    python scripts/make_ascii_svg.py --width 100
    python scripts/make_ascii_svg.py --color "#00bcd4"
    python scripts/make_ascii_svg.py --no-animation
    python scripts/make_ascii_svg.py --debug
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add scripts/ to path so sibling modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

from prep_photo   import run as prep_run
from edge_detector import run as edge_run
from ascii_engine  import AsciiEngine
from svg_renderer  import SvgRenderer, ROW_DELAY_MS


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
SOURCE_PHOTO  = Path("assets/source-photo.jpg")
PREPPED_IMAGE = Path("assets/source-prepped.png")
EDGE_MAP_PNG  = Path("assets/edge-map.png")
EDGE_DIR_NPY  = Path("assets/edge-direction.npy")
OUTPUT_SVG    = Path("output/avi-ascii.svg")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source:        Path  = SOURCE_PHOTO,
    output:        Path  = OUTPUT_SVG,
    width:         int   = 120,
    color:         str   = "#39d353",
    row_delay_ms:  int   = ROW_DELAY_MS,
    edge_threshold: float = 0.28,
    debug:         bool  = False,
    skip_prep:     bool  = False,
) -> Path:
    """
    Run the full ASCII portrait pipeline end-to-end.

    Parameters
    ----------
    source        : path to the raw source photo
    output        : path to write the final SVG
    width         : number of ASCII columns
    color         : primary text colour (hex)
    row_delay_ms  : ms between each row appearing in the animation
    edge_threshold: edge injection threshold [0-1]
    debug         : save intermediate images to assets/debug/
    skip_prep     : skip Stage 1+2 (reuse existing prepped image + edge map)

    Returns
    -------
    Path — path to the written SVG file
    """
    t0 = time.time()
    print("\n========================================")
    print("  ASCII Portrait Pipeline")
    print("========================================")

    # ── Stage 1: Preprocessing ────────────────────────────────────────────
    if not skip_prep:
        prep_run(
            source=source,
            output=PREPPED_IMAGE,
            cols=width,
            debug=debug,
        )
    else:
        print("\n[Stage 1] Skipped (--skip-prep)")

    # ── Stage 2: Edge Detection ───────────────────────────────────────────
    if not skip_prep:
        edge_map, direction = edge_run(
            source=PREPPED_IMAGE,
            out_map=EDGE_MAP_PNG,
            out_dir=EDGE_DIR_NPY,
            debug=debug,
        )
    else:
        print("[Stage 2] Loading existing edge map...")
        edge_uint8 = cv2.imread(str(EDGE_MAP_PNG), cv2.IMREAD_GRAYSCALE)
        edge_map   = edge_uint8.astype(np.float32) / 255.0 if edge_uint8 is not None else None
        direction  = np.load(str(EDGE_DIR_NPY)) if EDGE_DIR_NPY.exists() else None

    # ── Stage 3: ASCII Engine ─────────────────────────────────────────────
    print("\n-- Stage 3: ASCII Engine ----------------------------------------")
    gray = cv2.imread(str(PREPPED_IMAGE), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Preprocessed image not found: {PREPPED_IMAGE}")

    engine = AsciiEngine(edge_threshold=edge_threshold)
    ascii_rows = engine.convert(gray, edge_map, direction)
    print(f"  Grid: {len(ascii_rows[0])} cols x {len(ascii_rows)} rows")
    print(f"  Total characters: {sum(len(r) for r in ascii_rows):,}")

    # ── Stage 4: SVG Renderer ─────────────────────────────────────────────
    print("\n-- Stage 4: SVG Renderer ----------------------------------------")
    renderer = SvgRenderer(
        row_delay_ms=row_delay_ms,
    )
    svg = renderer.render(ascii_rows, title="Naman Upadhyay | ASCII Portrait")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")

    elapsed = time.time() - t0
    size_kb  = len(svg.encode("utf-8")) / 1024

    print(f"\n========================================")
    print(f"  Done!  {elapsed:.1f}s")
    print(f"  Output: {output}")
    print(f"  Size:   {size_kb:.1f} KB")
    print(f"========================================\n")

    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an animated ASCII portrait SVG from a source photo."
    )
    parser.add_argument(
        "--source", type=Path, default=SOURCE_PHOTO,
        help="Path to source photo (default: assets/source-photo.jpg)",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_SVG,
        help="Path for output SVG (default: output/avi-ascii.svg)",
    )
    parser.add_argument(
        "--width", type=int, default=120,
        help="Number of ASCII columns (default: 120)",
    )
    parser.add_argument(
        "--color", type=str, default="#39d353",
        help="Primary text colour hex (default: #39d353 GitHub green)",
    )
    parser.add_argument(
        "--row-delay", type=int, default=ROW_DELAY_MS,
        help="Milliseconds between row reveals (default: 35)",
    )
    parser.add_argument(
        "--edge-threshold", type=float, default=0.28,
        help="Edge injection threshold 0-1 (default: 0.28)",
    )
    parser.add_argument(
        "--no-animation", action="store_true",
        help="Output a static SVG (no animation delays)",
    )
    parser.add_argument(
        "--skip-prep", action="store_true",
        help="Skip preprocessing + edge detection (reuse cached files)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate images to assets/debug/",
    )
    args = parser.parse_args()

    run_pipeline(
        source         = args.source,
        output         = args.output,
        width          = args.width,
        color          = args.color,
        row_delay_ms   = 0 if args.no_animation else args.row_delay,
        edge_threshold = args.edge_threshold,
        debug          = args.debug,
        skip_prep      = args.skip_prep,
    )