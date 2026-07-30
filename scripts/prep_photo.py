"""
prep_photo.py
=============
Stage 1 of the ASCII pipeline — mode-aware image preprocessing.

Supports two modes:
  portrait   -- face/subject photo. Bilateral filter + sharp unsharp mask +
                gamma lift to pull out facial detail against dark background.
  landscape  -- scenic/wide photo. Full frame kept (no subject isolation),
                gentler CLAHE to avoid blowing out skies, pre-blur to suppress
                fine texture aliasing, histogram stretch instead of gamma lift.

Usage:
    python scripts/prep_photo.py --mode portrait
    python scripts/prep_photo.py --mode landscape --input assets/scene.jpg
    python scripts/prep_photo.py --mode portrait --debug
    python scripts/prep_photo.py --mode landscape --cols 140 --rows 45 --debug

Pipeline (portrait):
    Load -> Bilateral filter -> LAB-CLAHE (clip=2.5) -> Grayscale ->
    Unsharp mask -> Gamma correction (0.75) -> Resize -> Save

Pipeline (landscape):
    Load -> Gaussian pre-blur -> LAB-CLAHE (clip=1.2, gentler) -> Grayscale ->
    Mild unsharp mask -> Histogram stretch -> Letterbox crop -> Resize -> Save
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PORTRAIT  = Path("assets/source-photo.jpg")
OUTPUT_PORTRAIT = Path("assets/source-prepped.png")

INPUT_LANDSCAPE  = Path("assets/landscape-photo.jpg")
OUTPUT_LANDSCAPE = Path("assets/landscape-prepped.png")

DEBUG_DIR = Path("assets/debug")

# ---------------------------------------------------------------------------
# Grid defaults per mode
# ---------------------------------------------------------------------------
# Characters are taller than wide — this ratio corrects for that
CHAR_ASPECT = 0.55   # (char_width / char_height) for monospace fonts

PORTRAIT_COLS  = 120   # ~taller output matching portrait aspect
LANDSCAPE_COLS = 140   # wider grid — landscapes need horizontal breathing room
LANDSCAPE_ROWS = 45    # explicit row cap keeps sky from eating all real estate


# ---------------------------------------------------------------------------
# Shared processing functions
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load image as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    print(f"  Loaded: {path}  ({img.shape[1]}x{img.shape[0]} px)")
    return img


def bilateral_smooth(bgr: np.ndarray,
                     d: int = 9,
                     sigma_color: float = 75,
                     sigma_space: float = 75) -> np.ndarray:
    """
    Bilateral filter: blurs uniform regions while preserving hard edges.
    Used in portrait mode to smooth skin noise without blurring glasses/hair.
    """
    return cv2.bilateralFilter(bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def gaussian_preblur(bgr: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Mild Gaussian blur pre-pass for landscape mode.

    Purpose: suppress fine texture (leaves, water ripples, foliage) that
    would alias into visual noise when downsampled to the character grid's
    low resolution. We use a small kernel (3x3) — just enough to prevent
    Moire patterns without destroying horizon/terrain structure.
    """
    return cv2.GaussianBlur(bgr, (ksize, ksize), 0)


def lab_clahe(bgr: np.ndarray, clip_limit: float = 2.5,
              tile_size: int = 8) -> np.ndarray:
    """
    LAB-space CLAHE: enhances local contrast without colour shift.

    clip_limit controls aggressiveness:
      2.5 — portrait default (punchy, pulls out facial shadow detail)
      1.2 — landscape default (gentle; skies already have wide tonal range
             and over-boosting creates noisy gradients in uniform sky regions)

    Applied only to the L (lightness) channel in LAB space.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def unsharp_mask(gray: np.ndarray,
                 strength: float = 1.2,
                 blur_radius: int = 2) -> np.ndarray:
    """
    Unsharp masking to sharpen fine detail.

    Portrait: strength=1.2 (aggressive — glasses, hair strands matter)
    Landscape: strength=0.5 (mild — we want edge clarity but not harsh halos
               on horizon lines or tree canopy edges)
    """
    blurred = cv2.GaussianBlur(
        gray, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0
    )
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def gamma_correct(gray: np.ndarray, gamma: float = 0.75) -> np.ndarray:
    """
    Power-law gamma correction (portrait mode).
    gamma < 1.0 brightens dark areas (lifts shadow detail in faces).
    Not used in landscape mode — histogram_stretch handles that instead.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(gray, table)


def histogram_stretch(gray: np.ndarray,
                      low_pct: float = 1.0,
                      high_pct: float = 99.0) -> np.ndarray:
    """
    Percentile-based histogram stretch (landscape mode).

    Gamma correction is tuned for faces (lift shadows).
    Landscapes need a different mapping: the brightest region (sky) should
    map toward the ramp's sparse end (' ' space chars) and the darkest
    region (terrain silhouettes) should map to dense glyphs ('@' '#').

    We clip at the 1st and 99th percentiles to handle extreme highlights
    (sun disc, specular water) and deep shadows without crushing everything.
    """
    p_low  = float(np.percentile(gray, low_pct))
    p_high = float(np.percentile(gray, high_pct))

    if p_high <= p_low:
        return gray  # degenerate image, skip

    # Linear stretch: map [p_low, p_high] -> [0, 255]
    stretched = (gray.astype(np.float32) - p_low) / (p_high - p_low)
    stretched = np.clip(stretched * 255.0, 0, 255).astype(np.uint8)
    return stretched


def letterbox_crop(bgr: np.ndarray,
                   target_cols: int,
                   target_rows: int) -> np.ndarray:
    """
    Landscape-mode crop: extract the most information-dense region using
    a 16:9-ish target aspect, then resize to the character grid.

    The crop is centre-biased — landscapes usually have their horizon
    near the vertical centre, so we don't blindly take the top or bottom.

    Aspect ratio of the character grid at target_cols x target_rows:
        pixel_ratio = (target_cols * CHAR_W) / (target_rows * CHAR_H)
        With CHAR_W=4.8px, CHAR_H=10px -> each cell is 0.48:1
        target_cols=140, target_rows=45 -> grid is 140*4.8 / 45*10 = ~1.49:1

    We crop to that aspect ratio first, then resize to (cols, rows).
    """
    h, w = bgr.shape[:2]

    # Target pixel aspect ratio of the final ASCII character grid
    char_pixel_w = 4.8   # px per char column in SVG
    char_pixel_h = 10.0  # px per char row in SVG (line height)
    target_aspect = (target_cols * char_pixel_w) / (target_rows * char_pixel_h)

    current_aspect = w / h

    if current_aspect > target_aspect:
        # Image is wider than target — crop left/right
        new_w = int(h * target_aspect)
        x_start = (w - new_w) // 2
        cropped = bgr[:, x_start : x_start + new_w]
    else:
        # Image is taller than target — crop top/bottom (keep centre)
        new_h = int(w / target_aspect)
        y_start = (h - new_h) // 2
        cropped = bgr[y_start : y_start + new_h, :]

    return cv2.resize(
        cropped, (target_cols, target_rows), interpolation=cv2.INTER_AREA
    )


def smart_resize_portrait(gray: np.ndarray, cols: int) -> np.ndarray:
    """Resize portrait image preserving aspect ratio via char_aspect factor."""
    h, w = gray.shape
    rows = int(cols * (h / w) * CHAR_ASPECT)
    resized = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    print(f"  Resized to: {cols}x{rows} (cols x rows) character grid")
    return resized


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def save_debug(name: str, img: np.ndarray, debug: bool,
               prefix: str = "") -> None:
    if not debug:
        return
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_DIR / f"{prefix}{name}.png"
    cv2.imwrite(str(path), img)
    print(f"  [debug] Saved: {path}")


# ---------------------------------------------------------------------------
# Portrait pipeline
# ---------------------------------------------------------------------------

def run_portrait(
    source: Path,
    output: Path,
    cols: int = PORTRAIT_COLS,
    debug: bool = False,
) -> Path:
    """
    Portrait preprocessing pipeline.
    Optimised for face photos: bilateral smoothing + aggressive CLAHE +
    unsharp mask + gamma lift for shadow detail.
    """
    print("\n-- [portrait] Preprocessing ------------------------------------")

    print("[1/6] Loading image...")
    bgr = load_image(source)
    save_debug("01_original", bgr, debug, "p_")

    print("[2/6] Applying bilateral filter...")
    bgr = bilateral_smooth(bgr, d=9, sigma_color=75, sigma_space=75)
    save_debug("02_bilateral", bgr, debug, "p_")

    print("[3/6] Applying LAB-space CLAHE (clip=2.5)...")
    bgr = lab_clahe(bgr, clip_limit=2.5, tile_size=8)
    save_debug("03_clahe", bgr, debug, "p_")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    print("[4/6] Applying unsharp mask (strength=1.2)...")
    gray = unsharp_mask(gray, strength=1.2, blur_radius=2)
    save_debug("04_unsharp", gray, debug, "p_")

    print("[5/6] Applying gamma correction (gamma=0.75)...")
    gray = gamma_correct(gray, gamma=0.75)
    save_debug("05_gamma", gray, debug, "p_")

    print("[6/6] Resizing to character grid...")
    gray = smart_resize_portrait(gray, cols=cols)
    save_debug("06_resized", gray, debug, "p_")

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), gray)
    print(f"\nOK  Portrait prepped image saved: {output}")
    return output


# ---------------------------------------------------------------------------
# Landscape pipeline
# ---------------------------------------------------------------------------

def run_landscape(
    source: Path,
    output: Path,
    cols: int = LANDSCAPE_COLS,
    rows: int = LANDSCAPE_ROWS,
    debug: bool = False,
) -> Path:
    """
    Landscape preprocessing pipeline.
    Keeps the full frame (no background removal), applies gentler CLAHE
    to preserve sky gradients, and uses histogram stretch instead of gamma
    so skies map to sparse chars and terrain maps to dense glyphs.
    """
    print("\n-- [landscape] Preprocessing -----------------------------------")

    print("[1/6] Loading image...")
    bgr = load_image(source)
    save_debug("01_original", bgr, debug, "l_")

    # Pre-blur: suppress fine texture before downsampling
    print("[2/6] Applying Gaussian pre-blur (ksize=3)...")
    bgr = gaussian_preblur(bgr, ksize=3)
    save_debug("02_preblur", bgr, debug, "l_")

    # Gentle CLAHE — clip=1.2 avoids blowing out sky gradients
    print("[3/6] Applying LAB-space CLAHE (clip=1.2, gentle)...")
    bgr = lab_clahe(bgr, clip_limit=1.2, tile_size=12)
    save_debug("03_clahe", bgr, debug, "l_")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Mild unsharp mask — sharpen horizon/silhouette edges only
    print("[4/6] Applying mild unsharp mask (strength=0.5)...")
    gray = unsharp_mask(gray, strength=0.5, blur_radius=2)
    save_debug("04_unsharp", gray, debug, "l_")

    # Histogram stretch: maps actual tonal range to full 0-255
    # Result: sky -> near 255 (sparse chars), terrain -> near 0 (dense chars)
    print("[5/6] Applying histogram stretch (p1-p99)...")
    gray = histogram_stretch(gray, low_pct=1.0, high_pct=99.0)
    save_debug("05_stretch", gray, debug, "l_")

    # Letterbox crop + resize to landscape character grid
    print(f"[6/6] Letterbox crop + resize to {cols}x{rows} grid...")
    gray = letterbox_crop(gray, target_cols=cols, target_rows=rows)
    save_debug("06_resized", gray, debug, "l_")
    print(f"  Final grid: {gray.shape[1]}x{gray.shape[0]} (cols x rows)")

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), gray)
    print(f"\nOK  Landscape prepped image saved: {output}")
    return output


# ---------------------------------------------------------------------------
# Unified entry point (for import by orchestrator)
# ---------------------------------------------------------------------------

def run(
    source: Path,
    output: Path,
    mode: str = "portrait",
    cols: int | None = None,
    rows: int | None = None,
    debug: bool = False,
) -> Path:
    """
    Run the appropriate preprocessing pipeline based on mode.

    Parameters
    ----------
    source  : path to raw source photo
    output  : path to write the prepped grayscale PNG
    mode    : 'portrait' or 'landscape'
    cols    : override number of ASCII columns (None = use mode default)
    rows    : override number of ASCII rows — landscape only (None = mode default)
    debug   : save intermediate debug images
    """
    if mode == "portrait":
        return run_portrait(
            source=source,
            output=output,
            cols=cols or PORTRAIT_COLS,
            debug=debug,
        )
    elif mode == "landscape":
        return run_landscape(
            source=source,
            output=output,
            cols=cols or LANDSCAPE_COLS,
            rows=rows or LANDSCAPE_ROWS,
            debug=debug,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'portrait' or 'landscape'.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a photo for ASCII art conversion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prep_photo.py --mode portrait
  python scripts/prep_photo.py --mode portrait --input assets/my-face.jpg --debug
  python scripts/prep_photo.py --mode landscape --input assets/mountains.jpg
  python scripts/prep_photo.py --mode landscape --cols 140 --rows 45 --debug
        """,
    )
    parser.add_argument(
        "--mode", choices=["portrait", "landscape"], default="portrait",
        help="Processing mode: portrait (face) or landscape (scenic). Default: portrait",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to source photo. Default: assets/source-photo.jpg (portrait) "
             "or assets/landscape-photo.jpg (landscape)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path for output prepped PNG. Default: assets/source-prepped.png (portrait) "
             "or assets/landscape-prepped.png (landscape)",
    )
    parser.add_argument(
        "--cols", type=int, default=None,
        help=f"ASCII columns. Default: {PORTRAIT_COLS} (portrait) / {LANDSCAPE_COLS} (landscape)",
    )
    parser.add_argument(
        "--rows", type=int, default=None,
        help=f"ASCII rows (landscape only). Default: {LANDSCAPE_ROWS}",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate images to assets/debug/",
    )
    args = parser.parse_args()

    # Resolve defaults based on mode
    input_path  = args.input  or (INPUT_PORTRAIT  if args.mode == "portrait" else INPUT_LANDSCAPE)
    output_path = args.output or (OUTPUT_PORTRAIT  if args.mode == "portrait" else OUTPUT_LANDSCAPE)

    run(
        source=input_path,
        output=output_path,
        mode=args.mode,
        cols=args.cols,
        rows=args.rows,
        debug=args.debug,
    )