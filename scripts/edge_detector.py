"""
edge_detector.py
================
Stage 2 of the ASCII portrait pipeline.

Responsibility:
    Produce a clean edge map from the preprocessed grayscale image.
    The edge map is used by ascii_engine.py to inject structural
    characters (| - / \\) at detected boundaries instead of mapping
    by brightness alone.

Pipeline:
    1. Load the preprocessed grayscale image
    2. Run Canny edge detection  -> binary edge mask
    3. Run Sobel gradient        -> magnitude + direction maps
    4. Blend: 0.6 * Sobel_norm + 0.4 * Canny_binary
    5. Save edge-map.png (and optional debug images)

Output:
    assets/edge-map.png        -- blended float32 edge map, saved as uint8
    assets/edge-direction.npy  -- gradient direction array (for ascii_engine)

Usage:
    python scripts/edge_detector.py
    python scripts/edge_detector.py --debug
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT     = Path("assets/source-prepped.png")
OUT_MAP   = Path("assets/edge-map.png")
OUT_DIR   = Path("assets/edge-direction.npy")
DEBUG_DIR = Path("assets/debug")


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def canny_edges(gray: np.ndarray,
                low: int = 30,
                high: int = 100) -> np.ndarray:
    """
    Canny edge detector.

    Returns a binary mask (0 or 255) at detected edges.
    We use a slight Gaussian blur first to suppress noise
    that would create too many spurious edges.

    low/high: hysteresis thresholds — pixels above high are strong edges,
    pixels between low and high are weak edges connected to strong ones.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=low, threshold2=high)
    return edges


def sobel_gradient(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sobel gradient: computes partial derivatives in X and Y directions.

    Returns:
        magnitude  -- float32 gradient strength, range [0, 1]
        direction  -- float32 gradient angle in radians, range [-pi, pi]
                      used by ascii_engine to pick directional characters
    """
    # Use 64-bit floats during computation to avoid overflow
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.hypot(sobel_x, sobel_y)          # Euclidean distance
    direction = np.arctan2(sobel_y, sobel_x)         # angle in radians

    # Normalise magnitude to [0, 1]
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max

    return magnitude.astype(np.float32), direction.astype(np.float32)


def blend_maps(canny: np.ndarray, sobel_mag: np.ndarray,
               canny_weight: float = 0.4,
               sobel_weight: float = 0.6) -> np.ndarray:
    """
    Combine Canny (crisp boundaries) with Sobel magnitude (soft gradients).

    Canny gives us clean, single-pixel edges at hard boundaries.
    Sobel gives us gradient strength which captures softer transitions.
    Blending both gives the ASCII engine richer structural signal.

    Returns a float32 map in range [0, 1].
    """
    # Normalise Canny to [0, 1]
    canny_norm = canny.astype(np.float32) / 255.0

    blended = (sobel_weight * sobel_mag) + (canny_weight * canny_norm)

    # Clip to [0, 1]
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def save_debug(name: str, img: np.ndarray, debug: bool) -> None:
    if not debug:
        return
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    # Convert float images to uint8 for saving
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    path = DEBUG_DIR / f"{name}.png"
    cv2.imwrite(str(path), img)
    print(f"  [debug] Saved: {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(source: Path = INPUT,
        out_map: Path = OUT_MAP,
        out_dir: Path = OUT_DIR,
        debug: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the edge detection pipeline.

    Returns:
        edge_map   -- float32 blended edge strength, shape (H, W), range [0,1]
        direction  -- float32 gradient direction in radians, shape (H, W)
    """
    print("\n-- Step 2: Edge Detection --------------------------------------")

    # 1. Load preprocessed grayscale image
    print("[1/4] Loading preprocessed image...")
    gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot open: {source}  (run prep_photo.py first)")
    print(f"  Shape: {gray.shape[1]}x{gray.shape[0]} px")

    # 2. Canny edge detection
    print("[2/4] Running Canny edge detection...")
    canny = canny_edges(gray, low=25, high=90)
    save_debug("07_canny", canny, debug)

    # 3. Sobel gradient
    print("[3/4] Running Sobel gradient...")
    sobel_mag, direction = sobel_gradient(gray)
    save_debug("08_sobel_mag", sobel_mag, debug)

    # 4. Blend maps
    print("[4/4] Blending edge maps...")
    edge_map = blend_maps(canny, sobel_mag, canny_weight=0.4, sobel_weight=0.6)
    save_debug("09_edge_blend", edge_map, debug)

    # 5. Save outputs
    out_map.parent.mkdir(parents=True, exist_ok=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    edge_uint8 = (edge_map * 255).astype(np.uint8)
    cv2.imwrite(str(out_map), edge_uint8)
    np.save(str(out_dir), direction)

    print(f"\nOK  Edge map saved to:       {out_map}")
    print(f"OK  Gradient dir saved to:   {out_dir}")

    return edge_map, direction


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an edge map from the preprocessed portrait."
    )
    parser.add_argument(
        "--input", type=Path, default=INPUT,
        help="Preprocessed grayscale PNG (default: assets/source-prepped.png)",
    )
    parser.add_argument(
        "--out-map", type=Path, default=OUT_MAP,
        help="Output edge map PNG (default: assets/edge-map.png)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help="Output gradient direction .npy (default: assets/edge-direction.npy)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate images to assets/debug/",
    )
    args = parser.parse_args()

    run(
        source=args.input,
        out_map=args.out_map,
        out_dir=args.out_dir,
        debug=args.debug,
    )
