"""
make_ascii_svg.py
=================
Mode-aware orchestrator for the ASCII pipeline.

Wires together all four pipeline stages:
    Stage 1: prep_photo.py    -- preprocess (portrait or landscape path)
    Stage 2: edge_detector.py -- compute edge map + gradient direction
    Stage 3: ascii_engine.py  -- convert to ASCII character grid
    Stage 4: svg_renderer.py  -- render animated SVG

Modes
-----
portrait  (default)
    Source:  assets/source-photo.jpg
    Prepped: assets/source-prepped.png
    Grid:    ~120 x auto (tall, face-shaped)
    Output:  output/avi-ascii.svg
    Ramp:    Standard density ramp

landscape
    Source:  assets/landscape-photo.jpg
    Prepped: assets/landscape-prepped.png
    Grid:    140 x 45 (wide, horizon-preserving)
    Output:  output/landscape-ascii.svg
    Ramp:    Extended light end to handle wide uniform skies

Usage:
    python scripts/make_ascii_svg.py
    python scripts/make_ascii_svg.py --mode portrait
    python scripts/make_ascii_svg.py --mode landscape --source assets/mountains.jpg
    python scripts/make_ascii_svg.py --mode landscape --cols 160 --rows 50
    python scripts/make_ascii_svg.py --mode portrait --skip-prep
    python scripts/make_ascii_svg.py --mode landscape --no-animation --debug
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add scripts/ to path so sibling modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

from prep_photo    import run as prep_run
from prep_photo    import PORTRAIT_COLS, LANDSCAPE_COLS, LANDSCAPE_ROWS
from edge_detector import run as edge_run
from ascii_engine  import AsciiEngine
from svg_renderer  import SvgRenderer, ROW_DELAY_MS


# ---------------------------------------------------------------------------
# Mode configuration table
# ---------------------------------------------------------------------------

MODE_DEFAULTS = {
    "portrait": {
        "source":        Path("assets/source-photo.jpg"),
        "prepped":       Path("assets/source-prepped.png"),
        "edge_map":      Path("assets/edge-map.png"),
        "edge_dir":      Path("assets/edge-direction.npy"),
        "output":        Path("output/avi-ascii.svg"),
        "cols":          PORTRAIT_COLS,         # 120
        "rows":          None,                  # auto from aspect ratio
        "edge_threshold": 0.28,
        "row_delay_ms":  35,
        "svg_title":     "Naman Upadhyay | ASCII Portrait",
    },
    "landscape": {
        "source":        Path("assets/landscape-photo.jpg"),
        "prepped":       Path("assets/landscape-prepped.png"),
        "edge_map":      Path("assets/landscape-edge-map.png"),
        "edge_dir":      Path("assets/landscape-edge-direction.npy"),
        "output":        Path("output/landscape-ascii.svg"),
        "cols":          LANDSCAPE_COLS,        # 140
        "rows":          LANDSCAPE_ROWS,        # 45
        "edge_threshold": 0.22,                 # lower — landscape edges are softer
        "row_delay_ms":  25,                    # faster per-row, wider grid feels right at ~2.5s total
        "svg_title":     "Naman Upadhyay | ASCII Landscape",
    },
}

# Landscape ASCII ramp — extended light tail for sky regions.
# Standard ramp: "@%#WMB&8*Xxo+=-:,. "
# Landscape ramp adds extra sparse chars so wide uniform skies
# get a visible gradient rather than collapsing to pure blank space.
LANDSCAPE_RAMP = "@%#WMB&8*Xxo+=-~:,.. "


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    mode:           str   = "portrait",
    source:         Path  = None,
    output:         Path  = None,
    cols:           int   = None,
    rows:           int   = None,
    row_delay_ms:   int   = None,
    edge_threshold: float = None,
    debug:          bool  = False,
    skip_prep:      bool  = False,
) -> Path:
    """
    Run the full ASCII pipeline end-to-end for the given mode.

    Parameters
    ----------
    mode            : 'portrait' or 'landscape'
    source          : override source photo path
    output          : override output SVG path
    cols            : override ASCII column count
    rows            : override ASCII row count (landscape only)
    row_delay_ms    : ms between row reveals in animation
    edge_threshold  : edge injection threshold [0-1]
    debug           : save intermediate debug images
    skip_prep       : skip Stage 1+2 (reuse cached prepped image + edge map)
    """
    if mode not in MODE_DEFAULTS:
        raise ValueError(f"Unknown mode {mode!r}. Use 'portrait' or 'landscape'.")

    cfg = MODE_DEFAULTS[mode]

    # Resolve final parameters (CLI overrides > mode defaults)
    source_path   = source         or cfg["source"]
    output_path   = output         or cfg["output"]
    final_cols    = cols           or cfg["cols"]
    final_rows    = rows           or cfg["rows"]
    final_delay   = row_delay_ms   if row_delay_ms is not None else cfg["row_delay_ms"]
    final_thresh  = edge_threshold if edge_threshold is not None else cfg["edge_threshold"]

    prepped_path = cfg["prepped"]
    edge_map_path = cfg["edge_map"]
    edge_dir_path = cfg["edge_dir"]

    t0 = time.time()
    print("\n========================================")
    print(f"  ASCII Pipeline  [{mode.upper()}]")
    print("========================================")

    # ── Stage 1: Preprocessing ────────────────────────────────────────────
    if not skip_prep:
        prep_run(
            source=source_path,
            output=prepped_path,
            mode=mode,
            cols=final_cols,
            rows=final_rows,
            debug=debug,
        )
    else:
        print(f"\n[Stage 1] Skipped (--skip-prep). Using: {prepped_path}")

    # ── Stage 2: Edge Detection ───────────────────────────────────────────
    if not skip_prep:
        edge_map, direction = edge_run(
            source=prepped_path,
            out_map=edge_map_path,
            out_dir=edge_dir_path,
            debug=debug,
        )
    else:
        print(f"[Stage 2] Loading cached edge map: {edge_map_path}")
        edge_uint8 = cv2.imread(str(edge_map_path), cv2.IMREAD_GRAYSCALE)
        edge_map   = (edge_uint8.astype(np.float32) / 255.0
                      if edge_uint8 is not None else None)
        direction  = (np.load(str(edge_dir_path))
                      if edge_dir_path.exists() else None)

    # ── Stage 3: ASCII Engine ─────────────────────────────────────────────
    print("\n-- Stage 3: ASCII Engine ----------------------------------------")
    gray = cv2.imread(str(prepped_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Preprocessed image not found: {prepped_path}")

    # Landscape mode: pass the extended ramp to the engine
    engine_kwargs: dict = {"edge_threshold": final_thresh}
    if mode == "landscape":
        engine_kwargs["ramp"] = LANDSCAPE_RAMP

    engine = AsciiEngine(**engine_kwargs)
    ascii_rows = engine.convert(gray, edge_map, direction)
    actual_cols = len(ascii_rows[0]) if ascii_rows else 0
    actual_rows = len(ascii_rows)
    print(f"  Grid: {actual_cols} cols x {actual_rows} rows")
    print(f"  Total characters: {sum(len(r) for r in ascii_rows):,}")

    # ── Stage 4: SVG Renderer ─────────────────────────────────────────────
    print("\n-- Stage 4: SVG Renderer ----------------------------------------")
    renderer = SvgRenderer(row_delay_ms=final_delay)
    svg = renderer.render(ascii_rows, title=cfg["svg_title"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")

    elapsed = time.time() - t0
    size_kb  = len(svg.encode("utf-8")) / 1024

    print(f"\n========================================")
    print(f"  Done!  {elapsed:.1f}s")
    print(f"  Mode:   {mode}")
    print(f"  Output: {output_path}")
    print(f"  Size:   {size_kb:.1f} KB")
    print(f"========================================\n")

    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an animated ASCII SVG from a source photo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Portrait (default)
  python scripts/make_ascii_svg.py

  # Landscape
  python scripts/make_ascii_svg.py --mode landscape --source assets/mountains.jpg

  # Portrait, skip preprocessing (reuse cached)
  python scripts/make_ascii_svg.py --mode portrait --skip-prep

  # Landscape, custom grid size, named output
  python scripts/make_ascii_svg.py --mode landscape --cols 160 --rows 50 --output output/scene.svg

  # Debug — save all intermediate images
  python scripts/make_ascii_svg.py --mode portrait --debug
        """,
    )
    parser.add_argument(
        "--mode", choices=["portrait", "landscape"], default="portrait",
        help="Pipeline mode. Default: portrait",
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Override source photo path",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Override output SVG path",
    )
    parser.add_argument(
        "--cols", type=int, default=None,
        help="ASCII columns (default: 120 portrait / 140 landscape)",
    )
    parser.add_argument(
        "--rows", type=int, default=None,
        help="ASCII rows, landscape only (default: 45)",
    )
    parser.add_argument(
        "--row-delay", type=int, default=None,
        help="Milliseconds between row reveals (default: 35 portrait / 25 landscape)",
    )
    parser.add_argument(
        "--edge-threshold", type=float, default=None,
        help="Edge injection threshold 0-1 (default: 0.28 portrait / 0.22 landscape)",
    )
    parser.add_argument(
        "--no-animation", action="store_true",
        help="Output a static SVG (row_delay=0)",
    )
    parser.add_argument(
        "--skip-prep", action="store_true",
        help="Skip Stage 1+2 preprocessing — reuse cached prepped image and edge map",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate images to assets/debug/",
    )
    args = parser.parse_args()

    run_pipeline(
        mode            = args.mode,
        source          = args.source,
        output          = args.output,
        cols            = args.cols,
        rows            = args.rows,
        row_delay_ms    = 0 if args.no_animation else args.row_delay,
        edge_threshold  = args.edge_threshold,
        debug           = args.debug,
        skip_prep       = args.skip_prep,
    )