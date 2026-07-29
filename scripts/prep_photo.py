from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove


INPUT = Path("assets/source-photo.jpg")
OUTPUT = Path("assets/source-prepped.png")

print("Loading image...")

image = Image.open(INPUT).convert("RGBA")

print("Removing background...")

foreground = remove(image)


background = Image.new(
    "RGBA",
    foreground.size,
    (255, 255, 255, 255),
)

background.paste(
    foreground,
    mask=foreground.getchannel("A"),
)

rgb = background.convert("RGB")

opencv = cv2.cvtColor(
    np.array(rgb),
    cv2.COLOR_RGB2BGR,
)

gray = cv2.cvtColor(
    opencv,
    cv2.COLOR_BGR2GRAY,
)

print("Applying CLAHE...")

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)

gray = clahe.apply(gray)

# -----------------------
# Resize
# -----------------------

TARGET_WIDTH = 100

h, w = gray.shape

aspect = h / w

TARGET_HEIGHT = int(TARGET_WIDTH * aspect)

gray = cv2.resize(
    gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA,
)

cv2.imwrite(str(OUTPUT), gray)

print("Done!")
print("Saved:", OUTPUT)