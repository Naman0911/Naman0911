"""
svg_renderer.py
===============
Stage 4 of the ASCII portrait pipeline.

Responsibility:
    Take a 2D list of ASCII character rows and render them into a
    fully animated SVG file compatible with GitHub README display.

Animation strategy (GitHub-compatible — zero JavaScript):
    - Each text row gets a CSS animation with a unique staggered delay
    - Rows reveal top-to-bottom like a terminal being typed
    - A blinking cursor line sweeps downward as rows appear
    - A one-shot scanline effect sweeps over the whole image on load
    - SVG <style> block drives all animation (no <script> tags)

GitHub SVG constraints:
    - No external resources (fonts, images, stylesheets)
    - No JavaScript
    - CSS animations are allowed
    - Inline styles are allowed
    - @keyframes inside <style> blocks work

Usage:
    from svg_renderer import SvgRenderer
    renderer = SvgRenderer()
    svg_text = renderer.render(ascii_rows)
    Path("output/avi-ascii.svg").write_text(svg_text, encoding="utf-8")

    Or standalone:
        python scripts/svg_renderer.py
"""

from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Monospace font stack — GitHub renders SVG with system fonts
FONT_FAMILY = "Consolas, 'Courier New', 'Lucida Console', monospace"

# Font metrics (pixels in the SVG coordinate system)
FONT_SIZE   = 8          # px — minimum readable at GitHub README widths
CHAR_W      = 4.8        # px — width of one monospace character at this size
LINE_H      = 10         # px — line height (slightly more than font size)

# Padding around the character grid
PAD_X = 10
PAD_Y = 12

# Colour scheme — GitHub terminal green on dark background
BG_COLOR    = "#0d1117"   # GitHub dark background
TEXT_COLOR  = "#39d353"   # GitHub contribution green (primary text)
DIM_COLOR   = "#1a4d2e"   # Dimmer green for background characters
CURSOR_COLOR = "#58a6ff"  # Blue cursor — pops against green text

# Animation timing
ROW_DELAY_MS   = 35       # ms between each row appearing
TOTAL_ANIM_S   = None     # Auto-calculated from row count

# Edge characters that indicate structure — rendered slightly brighter
EDGE_CHARS = set("|/-\\+")


# ---------------------------------------------------------------------------
# Colour grading
# ---------------------------------------------------------------------------

def char_to_color(char: str) -> str:
    """
    Return the hex colour for a character.

    Strategy:
        - Edge/structural characters (#, @, |, -, /) → bright green
        - Dense characters (M, B, W, 8) → medium green
        - Light characters (., ',  ) → dim green
        - Spaces → transparent (background)

    This gives a subtle luminance grading that makes the portrait
    look more three-dimensional.
    """
    if char == " ":
        return "none"          # transparent — don't render spaces

    dense = "@%#WMB&8"
    mid   = "*Xxo+="
    light = "-~:,'. "

    if char in EDGE_CHARS:
        return "#6ee7a0"       # bright edge highlight
    elif char in dense:
        return "#39d353"       # main green
    elif char in mid:
        return "#2ea043"       # mid green
    else:
        return "#1a7f37"       # dim green for near-white areas


# ---------------------------------------------------------------------------
# SVG building blocks
# ---------------------------------------------------------------------------

def _build_style(num_rows: int, row_delay_ms: int) -> str:
    """
    Generate the <style> block containing all CSS @keyframes and
    per-row animation classes.

    We use a single @keyframes 'rowReveal' and assign each row a
    different animation-delay via inline style rather than per-class
    rules — this keeps the SVG size down.
    """
    total_s = (num_rows * row_delay_ms) / 1000.0 + 1.5   # +1.5s settle

    return f"""<style>
  /* Row reveal animation */
  @keyframes rowReveal {{
    0%   {{ opacity: 0; transform: translateY(3px); }}
    100% {{ opacity: 1; transform: translateY(0);   }}
  }}

  /* Cursor blink */
  @keyframes cursorBlink {{
    0%, 49%  {{ opacity: 1; }}
    50%, 100% {{ opacity: 0; }}
  }}

  /* One-shot scanline sweep */
  @keyframes scanSweep {{
    0%   {{ transform: translateY(-100%); opacity: 0.12; }}
    80%  {{ transform: translateY(100%); opacity: 0.06; }}
    100% {{ transform: translateY(100%); opacity: 0;    }}
  }}

  .ascii-row {{
    opacity: 0;
    animation: rowReveal 0.25s ease forwards;
  }}

  .cursor {{
    animation: cursorBlink 0.9s step-start infinite;
    fill: {CURSOR_COLOR};
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE}px;
  }}

  .scanline {{
    animation: scanSweep {total_s:.1f}s ease-out forwards;
  }}
</style>"""


def _build_defs(svg_w: float, svg_h: float) -> str:
    """Build <defs> — clipPath to keep content within bounds."""
    return f"""<defs>
  <clipPath id="frame-clip">
    <rect x="0" y="0" width="{svg_w}" height="{svg_h}"/>
  </clipPath>
</defs>"""


def _row_to_svg_spans(row: str, y: float) -> str:
    """
    Convert one ASCII row into SVG <tspan> elements grouped by colour.

    Runs of the same colour are merged into a single tspan to keep
    file size reasonable.  Each tspan has an explicit dx positioning.

    Returns an inner SVG string (no wrapping <text> tag).
    """
    if not row.strip():
        # Blank row — emit a zero-width space to preserve line height
        return "&#x200B;"

    # Build list of (char, color) pairs, skip spaces entirely
    segments: list[tuple[str, str]] = []
    for ch in row:
        color = char_to_color(ch)
        segments.append((ch, color))

    # Group consecutive same-colour chars into runs
    spans: list[tuple[str, str]] = []   # [(text_run, color)]
    run_chars: list[str] = []
    run_color = segments[0][1] if segments else "none"

    for ch, color in segments:
        if color == run_color:
            run_chars.append(ch)
        else:
            spans.append(("".join(run_chars), run_color))
            run_chars = [ch]
            run_color = color
    if run_chars:
        spans.append(("".join(run_chars), run_color))

    # Build tspan elements
    # We use xml:space="preserve" on the parent text element and
    # rely on character count for positioning — simpler than dx lists
    parts: list[str] = []
    x_pos = 0.0
    for text_run, color in spans:
        safe = escape(text_run)
        if color == "none":
            # Still need to advance x position for spaces
            x_pos += len(text_run) * CHAR_W
            parts.append(
                f'<tspan x="{x_pos:.1f}" fill="none">{safe}</tspan>'
            )
        else:
            parts.append(
                f'<tspan x="{x_pos:.1f}" fill="{color}">{safe}</tspan>'
            )
            x_pos += len(text_run) * CHAR_W

    return "\n    ".join(parts)


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------

class SvgRenderer:
    """
    Renders ASCII rows into an animated GitHub-compatible SVG.

    Parameters
    ----------
    font_size     : px size of monospace font
    char_w        : px width of one character
    line_h        : px height of one line
    pad_x, pad_y  : padding around the grid
    bg_color      : background rectangle fill
    row_delay_ms  : milliseconds between each row reveal
    """

    def __init__(
        self,
        font_size: int    = FONT_SIZE,
        char_w: float     = CHAR_W,
        line_h: int       = LINE_H,
        pad_x: int        = PAD_X,
        pad_y: int        = PAD_Y,
        bg_color: str     = BG_COLOR,
        row_delay_ms: int = ROW_DELAY_MS,
    ):
        self.font_size    = font_size
        self.char_w       = char_w
        self.line_h       = line_h
        self.pad_x        = pad_x
        self.pad_y        = pad_y
        self.bg_color     = bg_color
        self.row_delay_ms = row_delay_ms

    def render(
        self,
        ascii_rows: list[str],
        title: str = "ASCII Portrait",
    ) -> str:
        """
        Render ASCII rows to an SVG string.

        Parameters
        ----------
        ascii_rows : list of strings, one per row
        title      : SVG <title> element text

        Returns
        -------
        str — complete SVG document
        """
        num_rows = len(ascii_rows)
        num_cols = max(len(r) for r in ascii_rows) if ascii_rows else 0

        svg_w = num_cols * self.char_w + 2 * self.pad_x
        svg_h = num_rows * self.line_h + 2 * self.pad_y

        # ── Header ────────────────────────────────────────────────────────
        lines: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg"',
            f'     width="{svg_w:.0f}" height="{svg_h:.0f}"',
            f'     viewBox="0 0 {svg_w:.0f} {svg_h:.0f}"',
            f'     role="img" aria-label="{escape(title)}">',
            f'  <title>{escape(title)}</title>',
            "",
            _build_defs(svg_w, svg_h),
            "",
            _build_style(num_rows, self.row_delay_ms),
            "",
            # Background
            f'  <rect width="{svg_w:.0f}" height="{svg_h:.0f}" fill="{self.bg_color}"/>',
            "",
            '  <!-- ASCII character rows -->',
            '  <g clip-path="url(#frame-clip)"',
            f'     font-family="{FONT_FAMILY}"',
            f'     font-size="{self.font_size}"',
            f'     xml:space="preserve">',
        ]

        # ── Rows ──────────────────────────────────────────────────────────
        for i, row in enumerate(ascii_rows):
            y = self.pad_y + (i + 1) * self.line_h
            delay_s = (i * self.row_delay_ms) / 1000.0

            # Cursor appears on the current "typing" row
            cursor_char = "&#9611;" if i < num_rows - 1 else ""  # ▋

            row_content = _row_to_svg_spans(row, y)

            lines.append(
                f'    <text class="ascii-row"'
                f' x="{self.pad_x}" y="{y}"'
                f' style="animation-delay:{delay_s:.3f}s">'
            )
            lines.append(f"      {row_content}")
            lines.append(f"    </text>")

        # ── Cursor (sweeps down as rows reveal) ───────────────────────────
        # One cursor element per row, each blinks only during its reveal window
        # then stops — achieved by a short animation duration + delay
        cursor_total_s = (num_rows * self.row_delay_ms) / 1000.0
        lines.append("")
        lines.append("    <!-- Sweeping cursor -->")
        for i in range(num_rows):
            y = self.pad_y + (i + 1) * self.line_h
            start_s = (i * self.row_delay_ms) / 1000.0
            end_s   = start_s + (self.row_delay_ms / 1000.0) * 3
            # Cursor visible from start_s to end_s then hidden
            # Using a single opacity animation
            lines.append(
                f'    <text class="cursor"'
                f' x="{self.pad_x}" y="{y}"'
                f' style="animation-delay:{start_s:.3f}s;'
                f'animation-duration:0.7s;'
                f'animation-iteration-count:3;">'
                f'&#9611;</text>'
            )

        lines.append("")
        lines.append("  </g>")

        # ── Scanline overlay ──────────────────────────────────────────────
        lines.append("")
        lines.append("  <!-- Scanline sweep effect -->")
        lines.append(
            f'  <rect class="scanline"'
            f' x="0" y="0"'
            f' width="{svg_w:.0f}" height="{self.line_h * 4}"'
            f' fill="rgba(57,211,83,0.08)"'
            f' style="animation-duration:{cursor_total_s + 0.5:.1f}s"/>'
        )

        lines.append("")
        lines.append("</svg>")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from ascii_engine import AsciiEngine

    parser = argparse.ArgumentParser(
        description="Render ASCII rows as an animated SVG (test mode)."
    )
    parser.add_argument(
        "--gray", type=Path, default=Path("assets/source-prepped.png"),
    )
    parser.add_argument(
        "--edge-map", type=Path, default=Path("assets/edge-map.png"),
    )
    parser.add_argument(
        "--edge-dir", type=Path, default=Path("assets/edge-direction.npy"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/avi-ascii.svg"),
    )
    parser.add_argument(
        "--no-animation", action="store_true",
        help="Render static SVG (no animation, for quick inspection)",
    )
    args = parser.parse_args()

    print("\n-- Step 4: SVG Renderer ----------------------------------------")

    # Load inputs
    gray = cv2.imread(str(args.gray), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot open: {args.gray}")

    edge_map = direction = None
    edge_uint8 = cv2.imread(str(args.edge_map), cv2.IMREAD_GRAYSCALE)
    if edge_uint8 is not None:
        edge_map = edge_uint8.astype(np.float32) / 255.0
    if args.edge_dir.exists():
        direction = np.load(str(args.edge_dir))

    # Generate ASCII
    engine = AsciiEngine(edge_threshold=0.28)
    ascii_rows = engine.convert(gray, edge_map, direction)
    print(f"  ASCII grid: {len(ascii_rows[0])} x {len(ascii_rows)}")

    # Render SVG
    renderer = SvgRenderer(
        row_delay_ms=0 if args.no_animation else ROW_DELAY_MS
    )
    svg = renderer.render(ascii_rows, title="Naman Upadhyay — ASCII Portrait")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"\nOK  SVG saved to: {args.output}  ({len(svg):,} bytes)")
