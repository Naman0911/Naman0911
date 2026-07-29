from pathlib import Path
from PIL import Image, ImageOps

# -----------------------------
# Configuration
# -----------------------------
INPUT = Path("assets/source-prepped.png")
OUTPUT = Path("output/avi-ascii.svg")

# Light -> Dark
RAMP = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

NEW_WIDTH = 120
FONT_SIZE = 7
CHAR_WIDTH = 5.2
LINE_HEIGHT = 8

# -----------------------------
# Load Image
# -----------------------------
img = Image.open(INPUT).convert("L")

# Improve contrast automatically
img = ImageOps.autocontrast(img)

# -----------------------------
# Resize
# -----------------------------
width, height = img.size

aspect_ratio = height / width

# Characters are taller than they are wide
new_height = int(aspect_ratio * NEW_WIDTH * 0.55)

img = img.resize((NEW_WIDTH, new_height))

pixels = img.load()

ascii_rows = []

for y in range(new_height):
    row = ""

    for x in range(NEW_WIDTH):

        pixel = pixels[x, y]

        # Invert brightness
        index = int((255 - pixel) / 255 * (len(RAMP) - 1))

        row += RAMP[index]

    ascii_rows.append(row)

# -----------------------------
# SVG Dimensions
# -----------------------------
svg_width = int(NEW_WIDTH * CHAR_WIDTH) + 20
svg_height = new_height * LINE_HEIGHT + 20

svg = f"""<svg xmlns="http://www.w3.org/2000/svg"
width="{svg_width}"
height="{svg_height}"
viewBox="0 0 {svg_width} {svg_height}">

<rect width="100%" height="100%" fill="#0d1117"/>

<g
font-family="Consolas, 'Courier New', monospace"
font-size="{FONT_SIZE}"
fill="#39d353">
"""

# -----------------------------
# Draw Text
# -----------------------------
for i, row in enumerate(ascii_rows):

    y = (i + 1) * LINE_HEIGHT

    svg += f"""
<text
x="8"
y="{y}"
opacity="0">{row}
<animate
attributeName="opacity"
from="0"
to="1"
begin="{i*0.03}s"
dur="0.05s"
fill="freeze"/>
</text>
"""

svg += """
</g>
</svg>
"""

OUTPUT.write_text(svg, encoding="utf-8")

print("ASCII SVG generated successfully!")
print("Saved to:", OUTPUT)