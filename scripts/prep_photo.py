"""
prep_photo.py
=============
Stage 1 of the ASCII portrait pipeline.

Responsibility:
    Take a raw source photo and produce a clean, high-contrast
    grayscale PNG that is optimised for ASCII-art conversion.

Pipeline:
    1. Load image
    2. Bilateral filter  — smooth noise while preserving edges
    3. LAB-space CLAHE   — local contrast enhancement (on L channel only)
    4. Unsharp mask      — sharpen fine detail (glasses, hair)
    5. Gamma correction  — lift shadow detail
    6. Resize            — to target character grid dimensions
    7. Save output + optional debug images

Usage:
    python scripts/prep_photo.py
    python scripts/prep_photo.py --debug   # saves intermediate images
    python scripts/prep_photo.py --width 120
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT = Path("assets/source-photo.jpg")
OUTPUT = Path("assets/source-prepped.png")
DEBUG_DIR = Path("assets/debug")

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
TARGET_COLS = 120          # number of ASCII columns we will generate
# Characters are taller than wide — this ratio corrects for that
CHAR_ASPECT  = 0.55        # (char_width / char_height) for monospace fonts


# ---------------------------------------------------------------------------
# Core preprocessing functions
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load image as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    print(f"  Loaded: {path}  ({img.shape[1]}×{img.shape[0]} px)")
    return img


def bilateral_smooth(bgr: np.ndarray) -> np.ndarray:
    """
    Bilateral filter: blurs uniform regions (skin) while preserving
    sharp edges (glasses frame, hair boundary, shirt collar).

    d=9         — neighbourhood diameter
    sigmaColor  — colour similarity threshold (higher = more smoothing)
    sigmaSpace  — spatial weight (higher = farther pixels influence)
    """
    return cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)


def lab_clahe(bgr: np.ndarray) -> np.ndarray:
    """
    Convert to LAB colour space, apply CLAHE only on the L (lightness)
    channel, then convert back.  This enhances local contrast without
    introducing colour shifts or over-saturating the image.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def unsharp_mask(gray: np.ndarray,
                 strength: float = 1.5,
                 blur_radius: int = 3) -> np.ndarray:
    """
    Unsharp masking: sharpen fine details by adding back the difference
    between the original and a blurred version.

    Formula:  sharpened = original + strength * (original - blurred)
    """
    blurred = cv2.GaussianBlur(gray, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def gamma_correct(gray: np.ndarray, gamma: float = 0.75) -> np.ndarray:
    """
    Power-law gamma correction.
    gamma < 1.0 → brightens dark areas (lifts shadow detail).
    gamma > 1.0 → darkens (increases contrast in highlights).

    We use 0.75 to pull detail out of the shadowed lower face.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(gray, table)


def smart_resize(gray: np.ndarray, cols: int, char_aspect: float) -> np.ndarray:
    """
    Resize to the target column count while preserving the visual aspect
    ratio when rendered as ASCII characters.

    ASCII characters are taller than wide, so we need fewer rows than
    the raw pixel aspect ratio would suggest.  char_aspect corrects for this.
    """
    h, w = gray.shape
    pixel_aspect = h / w
    rows = int(cols * pixel_aspect * char_aspect)
    resized = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    print(f"  Resized to: {cols}×{rows} (cols×rows) character grid")
    return resized


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def save_debug(name: str, img: np.ndarray, debug: bool) -> None:
    if not debug:
        return
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_DIR / f"{name}.png"
    cv2.imwrite(str(path), img)
    print(f"  [debug] Saved: {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(source: Path = INPUT,
        output: Path = OUTPUT,
        cols: int = TARGET_COLS,
        debug: bool = False) -> Path:
    """
    Run the full preprocessing pipeline.
    Returns the path of the saved output file.
    """
    print("\n-- Step 1: Preprocessing ---------------------------------------")

    # 1. Load
    print("[1/6] Loading image...")
    bgr = load_image(source)
    save_debug("01_original", bgr, debug)

    # 2. Bilateral smooth (noise → edges preserved)
    print("[2/6] Applying bilateral filter...")
    bgr = bilateral_smooth(bgr)
    save_debug("02_bilateral", bgr, debug)

    # 3. LAB-CLAHE (local contrast)
    print("[3/6] Applying LAB-space CLAHE...")
    bgr = lab_clahe(bgr)
    save_debug("03_clahe", bgr, debug)

    # 4. Convert to grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 5. Unsharp mask (sharpen fine detail)
    print("[4/6] Applying unsharp mask...")
    gray = unsharp_mask(gray, strength=1.2, blur_radius=2)
    save_debug("04_unsharp", gray, debug)

    # 6. Gamma correction (lift shadows)
    print("[5/6] Applying gamma correction...")
    gray = gamma_correct(gray, gamma=0.75)
    save_debug("05_gamma", gray, debug)

    # 7. Resize to target grid
    print("[6/6] Resizing to character grid dimensions...")
    gray = smart_resize(gray, cols=cols, char_aspect=CHAR_ASPECT)
    save_debug("06_resized", gray, debug)

    # 8. Save
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), gray)
    print(f"\nOK  Preprocessed image saved to: {output}")
    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a portrait photo for ASCII art conversion."
    )
    parser.add_argument(
        "--input", type=Path, default=INPUT,
        help="Path to source photo (default: assets/source-photo.jpg)",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT,
        help="Path to save preprocessed grayscale PNG (default: assets/source-prepped.png)",
    )
    parser.add_argument(
        "--width", type=int, default=TARGET_COLS,
        help="Target number of ASCII columns (default: 120)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate images to assets/debug/ for inspection",
    )
    args = parser.parse_args()

    run(
        source=args.input,
        output=args.output,
        cols=args.width,
        debug=args.debug,
    )