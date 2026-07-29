from pathlib import Path

from PIL import Image
from rembg import remove

INPUT = Path(r"D:\Github Profile\assets\source-photo.jpg")

PERSON = Path("assets/person.png")
LANDSCAPE = Path("assets/landscape.png")

print("Loading image...")

img = Image.open(INPUT).convert("RGBA")

print("Removing background...")

person = remove(img)

person.save(PERSON)

print("Person saved.")

# White background
white = Image.new("RGBA", img.size, (255, 255, 255, 255))
white.paste(person, mask=person.getchannel("A"))

white.save(LANDSCAPE)

print("Landscape saved.")