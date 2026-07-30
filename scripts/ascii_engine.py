"""
ascii_engine.py
===============
Stage 3 of the ASCII portrait pipeline.

Responsibility:
    Convert the preprocessed grayscale image + edge map into a 2D
    list of ASCII characters.

Algorithm:
    For every character cell (col, row):
      1. Sample the mean brightness of the corresponding pixel block.
      2. Map brightness to a character using a curated ramp.
      3. If the edge map strength exceeds a threshold, override the
         brightness-mapped character with a directional edge character
         (| - / \\) based on the dominant gradient angle.

Why block sampling?
    A single pixel maps poorly to a character.  A 4×7 block average
    produces smoother tonal transitions and more accurate density.

Why edge injection?
    Hard boundaries (glasses frame, hair outline) are best represented
    by structural characters, not density-mapped ones.  This gives the
    portrait a hand-crafted look.

Output:
    List[str]  — each string is one row of ASCII characters

Usage:
    Can be imported as a module:
        from ascii_engine import AsciiEngine
        rows = AsciiEngine().convert(gray, edge_map, direction)

    Or run standalone for a quick preview:
        python scripts/ascii_engine.py
"""

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Character ramp
# ---------------------------------------------------------------------------

# 16-level ramp from darkest (most filled) to lightest (emptiest).
# Characters are ordered by visual density (approximate ink coverage):
#
#   '@' = densest (use for near-black pixels)
#   ' ' = emptiest (use for near-white pixels / background)
#
# We deliberately avoid characters like '0', 'O', 'Q' that look too
# similar to circles and break the terminal feel.
#
RAMP_DARK_TO_LIGHT = "@%#W8BM&*Xx+=-:,. "

# Extended ramp for highlights — 24 levels for more gradual transitions.
# Uncomment to use instead:
# RAMP_DARK_TO_LIGHT = "@%#WMB&8*Xxo+=-~:,'. "

RAMP_LENGTH = len(RAMP_DARK_TO_LIGHT)      # e.g. 18


# ---------------------------------------------------------------------------
# Directional characters for edge injection
# ---------------------------------------------------------------------------

# We select based on gradient angle (in radians).
# The angle from arctan2 is in range [-pi, pi]:
#
#   ~0 or ~±pi  → horizontal edge  → use '-'
#   ~±pi/2      → vertical edge    → use '|'
#   ~pi/4       → diagonal         → use '/'
#   ~-pi/4      → diagonal         → use '\\'
#
EDGE_CHARS = {
    "horizontal":  "-",
    "vertical":    "|",
    "diag_up":     "/",
    "diag_down":   "\\",
}


def angle_to_edge_char(angle_rad: float) -> str:
    """
    Map a gradient angle (radians) to the most appropriate edge character.

    We normalise the angle to [0, pi) (ignoring sign, since we only care
    about orientation not direction) then assign quadrants.
    """
    # Normalise to [0, pi)
    angle = angle_rad % math.pi

    if angle < math.pi / 8 or angle >= 7 * math.pi / 8:
        return EDGE_CHARS["horizontal"]   # near 0 or 180 deg
    elif angle < 3 * math.pi / 8:
        return EDGE_CHARS["diag_up"]      # near 45 deg
    elif angle < 5 * math.pi / 8:
        return EDGE_CHARS["vertical"]     # near 90 deg
    else:
        return EDGE_CHARS["diag_down"]    # near 135 deg


# ---------------------------------------------------------------------------
# Brightness-to-character mapping
# ---------------------------------------------------------------------------

def brightness_to_char(brightness: float) -> str:
    """
    Map a normalised brightness value [0, 1] to a character.

    We apply square-root gamma compression so that the midtones
    (skin, hair) get more character variety instead of mapping
    all facial tones to the same dense characters.

    brightness = 0.0 → darkest char (pixel is black)
    brightness = 1.0 → lightest char (pixel is white / background)
    """
    # Square-root compression expands the dark range
    compressed = math.sqrt(brightness)

    # Map to ramp index
    index = int(compressed * (RAMP_LENGTH - 1))
    index = max(0, min(index, RAMP_LENGTH - 1))

    return RAMP_DARK_TO_LIGHT[index]


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class AsciiEngine:
    """
    Converts a grayscale image into ASCII art using block sampling
    and optional edge injection.

    Parameters
    ----------
    edge_threshold : float
        Edge map strength above which we inject a structural character.
        Range [0, 1].  0.28 is portrait default (captures clear edges,
        ignores subtle gradients that would look noisy as characters).
        0.22 is landscape default (softer edges in terrain/sky).
    block_w, block_h : int
        Pixel dimensions of each character cell.
    ramp : str | None
        Custom character ramp (dark-to-light). If None, uses RAMP_DARK_TO_LIGHT.
        Landscape mode passes an extended ramp with extra light chars for skies.
    """

    def __init__(self,
                 edge_threshold: float = 0.28,
                 block_w: int = 4,
                 block_h: int = 7,
                 ramp: str | None = None):
        self.edge_threshold = edge_threshold
        self.block_w = block_w
        self.block_h = block_h
        self._ramp = ramp if ramp is not None else RAMP_DARK_TO_LIGHT
        self._ramp_length = len(self._ramp)

    def convert(
        self,
        gray: np.ndarray,
        edge_map: Optional[np.ndarray] = None,
        direction: Optional[np.ndarray] = None,
    ) -> list[str]:
        """
        Convert a grayscale image to ASCII rows.

        Parameters
        ----------
        gray       : uint8 grayscale image, shape (H, W)
        edge_map   : float32 [0,1] edge strength map, same shape as gray
                     If None, edge injection is disabled.
        direction  : float32 gradient direction in radians, same shape as gray
                     If None, all edge chars default to '|'.

        Returns
        -------
        List[str] — one string per row of ASCII output
        """
        img_h, img_w = gray.shape
        cols = img_w      # already resized to final character count
        rows = img_h

        ascii_rows: list[str] = []

        for row_idx in range(rows):
            row_chars: list[str] = []

            for col_idx in range(cols):
                # ── Block sampling ──────────────────────────────────────
                # Compute the pixel region this character cell covers.
                # Since the image is already pre-resized to the exact
                # character grid, each "cell" is just 1 pixel wide.
                # We sample the pixel directly and treat block_w/block_h
                # as advisory (for future higher-res input support).
                pixel_val = int(gray[row_idx, col_idx])

                # Normalise to [0, 1] where 0=black, 1=white
                brightness = pixel_val / 255.0

                # ── Default: brightness-mapped character ─────────────────
                char = brightness_to_char(brightness)

                # ── Edge injection ───────────────────────────────────────
                if edge_map is not None:
                    edge_strength = float(edge_map[row_idx, col_idx])

                    if edge_strength >= self.edge_threshold:
                        if direction is not None:
                            angle = float(direction[row_idx, col_idx])
                            char = angle_to_edge_char(angle)
                        else:
                            char = "|"

                row_chars.append(char)

            ascii_rows.append("".join(row_chars))

        return ascii_rows


# ---------------------------------------------------------------------------
# Standalone preview (for testing)
# ---------------------------------------------------------------------------

def preview_in_terminal(rows: list[str], max_rows: int = 40) -> None:
    """Print a portion of the ASCII art to the terminal for a quick sanity check."""
    print("\n-- ASCII Preview (first {} rows) --------------------------------".format(max_rows))
    for row in rows[:max_rows]:
        print(row)
    print("..." if len(rows) > max_rows else "")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert preprocessed portrait to ASCII rows (preview mode)."
    )
    parser.add_argument(
        "--gray", type=Path, default=Path("assets/source-prepped.png"),
        help="Preprocessed grayscale PNG",
    )
    parser.add_argument(
        "--edge-map", type=Path, default=Path("assets/edge-map.png"),
        help="Blended edge map PNG",
    )
    parser.add_argument(
        "--edge-dir", type=Path, default=Path("assets/edge-direction.npy"),
        help="Gradient direction .npy file",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.28,
        help="Edge injection threshold [0-1] (default: 0.28)",
    )
    parser.add_argument(
        "--no-edges", action="store_true",
        help="Disable edge injection (pure brightness mapping)",
    )
    args = parser.parse_args()

    print("\n-- Step 3: ASCII Engine ----------------------------------------")

    # Load inputs
    gray = cv2.imread(str(args.gray), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot open: {args.gray}")

    edge_map = direction = None
    if not args.no_edges:
        edge_uint8 = cv2.imread(str(args.edge_map), cv2.IMREAD_GRAYSCALE)
        if edge_uint8 is not None:
            edge_map = edge_uint8.astype(np.float32) / 255.0
            print(f"  Edge map loaded: {args.edge_map}")
        if args.edge_dir.exists():
            direction = np.load(str(args.edge_dir))
            print(f"  Gradient dir loaded: {args.edge_dir}")

    # Run engine
    engine = AsciiEngine(edge_threshold=args.threshold)
    rows = engine.convert(gray, edge_map, direction)

    print(f"\n  Grid: {len(rows[0])} cols x {len(rows)} rows")
    preview_in_terminal(rows, max_rows=30)
    print(f"\nOK  ASCII engine ran successfully.")
