import os
import sys
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import math

def process_photo(photo_path, dark_mode=True):
    print("Processing photo...")
    try:
        # Load and remove background if dark mode
        from rembg import remove
        img = Image.open(photo_path).convert("RGBA")
        if dark_mode:
            img = remove(img)
            
        # Composite on appropriate background
        bg_color = (0, 0, 0, 255) if dark_mode else (255, 255, 255, 255)
        bg = Image.new("RGBA", img.size, bg_color)
        bg.paste(img, mask=img.split()[3] if dark_mode else None)
        img = bg.convert("L")
        
        # Crop to head & shoulders roughly (assuming central)
        width, height = img.size
        min_dim = min(width, height)
        img = img.crop((width//2 - min_dim//2, 0, width//2 + min_dim//2, min_dim))
        
        # Resize to grid
        img = img.resize((100, 113), Image.Resampling.LANCZOS)
        
        # Contrast & Unsharp
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
        
        return img
    except Exception as e:
        print(f"Error processing photo: {e}")
        return Image.new("L", (300, 340), 255)

def floyd_steinberg_dither(img):
    print("Dithering...")
    arr = np.array(img, dtype=float)
    h, w = arr.shape
    out = np.zeros_like(arr, dtype=bool)
    
    for y in range(h):
        # Serpentine
        x_range = range(w) if y % 2 == 0 else range(w-1, -1, -1)
        for x in x_range:
            old_val = arr[y, x]
            new_val = 255 if old_val > 128 else 0
            out[y, x] = (new_val == 0) # True means dot (dark part of image)
            
            err = old_val - new_val
            
            # Simple error diffusion for serpentine
            dir_x = 1 if y % 2 == 0 else -1
            if 0 <= x + dir_x < w:
                arr[y, x + dir_x] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - dir_x < w:
                    arr[y + 1, x - dir_x] += err * 3 / 16
                arr[y + 1, x] += err * 5 / 16
                if 0 <= x + dir_x < w:
                    arr[y + 1, x + dir_x] += err * 1 / 16
    return out

def get_logo_points(logo_path, num_points=900, scale=0.8, offset=(100, 50)):
    print(f"Processing logo: {logo_path}")
    try:
        img = Image.open(logo_path).convert("L")
        # Resize to fit within right panel
        img = img.resize((150, 150), Image.Resampling.LANCZOS)
        img = ImageOps.invert(img)
        arr = np.array(img)
        # Threshold
        y_idx, x_idx = np.where(arr > 128)
        
        if len(y_idx) == 0:
            raise ValueError("No points found")
            
        # Sample points
        indices = np.random.choice(len(y_idx), min(num_points, len(y_idx)), replace=False)
        pts = np.column_stack((x_idx[indices], y_idx[indices]))
        
        # Scale and offset
        pts = pts * scale + np.array(offset)
        
        # Pad if needed
        if len(pts) < num_points:
            padding = np.zeros((num_points - len(pts), 2))
            pts = np.vstack([pts, padding])
            
        return pts
    except Exception as e:
        print(f"Error with logo {logo_path}: {e}")
        # Return random circle
        angles = np.random.uniform(0, 2*np.pi, num_points)
        r = np.random.uniform(0, 50, num_points)
        return np.column_stack((r*np.cos(angles), r*np.sin(angles))) + np.array(offset)

def match_points(pts1, pts2):
    print("Matching points (Optimal Transport)...")
    dists = cdist(pts1, pts2)
    row_ind, col_ind = linear_sum_assignment(dists)
    return pts2[col_ind]

def generate_svg(dark_mode, dots_mask, logo_points_list, user_details):
    print(f"Generating {'dark' if dark_mode else 'light'} SVG...")
    
    bg_color = "#0A101F" if dark_mode else "#FFFFFF"
    portrait_color = "#A78BFA" if dark_mode else "#7C3AED"
    chrome_color = "#22D3EE" if dark_mode else "#0891B2"
    text_color = "#FFFFFF" if dark_mode else "#000000"
    
    width, height = 1180, 610
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    svg.append(f'<rect width="{width}" height="{height}" fill="{bg_color}" />')
    
    # SYSTEM.INFO Chrome
    svg.append(f'<text x="500" y="50" font-family="monospace" font-size="14" fill="{chrome_color}">SYSTEM.INFO</text>')
    
    y_offset = 100
    for key, val in user_details.items():
        svg.append(f'<text x="500" y="{y_offset}" font-family="monospace" font-size="14" fill="{text_color}">{key.upper()}</text>')
        # Dotted leader
        leader = "." * max(1, 50 - len(key) - len(str(val)))
        svg.append(f'<text x="650" y="{y_offset}" font-family="monospace" font-size="14" fill="{chrome_color}">{leader}</text>')
        svg.append(f'<text x="800" y="{y_offset}" font-family="monospace" font-size="14" fill="{text_color}">{val}</text>')
        y_offset += 30
        
    # Portrait (Optimized: use 3x3 rects for larger, fewer elements)
    scale_factor = 3  # Each dither pixel becomes 3x3 SVG pixels
    svg.append(f'<path fill="{portrait_color}" shape-rendering="crispEdges" d="')
    y_idx, x_idx = np.where(dots_mask)
    
    path_data = []
    for x, y in zip(x_idx, y_idx):
        sx = x * scale_factor + 50
        sy = y * scale_factor + 50
        path_data.append(f"M{sx} {sy}h{scale_factor}v{scale_factor}h-{scale_factor}z")
    
    svg.append("".join(path_data))
    svg.append('" />')
    
    # Logo Travellers (simplified animation using CSS for demo)
    if logo_points_list:
        svg.append(f'<g fill="{chrome_color}" shape-rendering="crispEdges">')
        pts1, pts2, pts3 = logo_points_list
        for i in range(len(pts1)):
            cx1, cy1 = round(pts1[i,0]+800,1), round(pts1[i,1]+400,1)
            cx2, cy2 = round(pts2[i,0]+800,1), round(pts2[i,1]+400,1)
            cx3, cy3 = round(pts3[i,0]+800,1), round(pts3[i,1]+400,1)
            svg.append(f'<circle cx="{cx1}" cy="{cy1}" r="1.5"><animate attributeName="cx" values="{cx1};{cx2};{cx3};{cx1}" dur="14.2s" repeatCount="indefinite"/><animate attributeName="cy" values="{cy1};{cy2};{cy3};{cy1}" dur="14.2s" repeatCount="indefinite"/></circle>')
        svg.append('</g>')
        
    svg.append('</svg>')
    return "\n".join(svg)

def main():
    os.makedirs("output", exist_ok=True)
    
    user_details = {
        'Name': 'Naman Vijay Upadhyay',
        'Role': 'AI/ML & Backend Developer',
        'Location': 'Pune, Maharashtra, India',
        'Status': 'Building AI Solutions',
        'Languages': 'Python, C, C++, SQL',
        'Frontend': 'HTML, CSS, Streamlit, React',
        'Backend': 'FastAPI, REST APIs',
        'Database': 'MySQL, PostgreSQL, Redis',
        'Infra': 'Docker, AWS EC2',
    }
    
    source_photo = "assets/source_photo.jpg.jpeg"
    logos = [
        "assets/Ai_image_logo.png",
        "assets/Coder_image_logo.jpg",
        "assets/Coding_image_logo.png"
    ]
    
    print("--- Dark Mode ---")
    img_dark = process_photo(source_photo, dark_mode=True)
    mask_dark = floyd_steinberg_dither(img_dark)
    
    print("--- Light Mode ---")
    img_light = process_photo(source_photo, dark_mode=False)
    mask_light = floyd_steinberg_dither(img_light)
    
    print("--- Logos ---")
    num_pts = 400
    pts1 = get_logo_points(logos[0], num_points=num_pts)
    pts2 = match_points(pts1, get_logo_points(logos[1], num_points=num_pts))
    pts3 = match_points(pts2, get_logo_points(logos[2], num_points=num_pts))
    
    svg_dark = generate_svg(True, mask_dark, [pts1, pts2, pts3], user_details)
    svg_light = generate_svg(False, mask_light, [pts1, pts2, pts3], user_details)
    
    with open("output/dark.svg", "w") as f:
        f.write(svg_dark)
    with open("output/light.svg", "w") as f:
        f.write(svg_light)
        
    print("Done! SVGs generated in output/")

if __name__ == "__main__":
    main()
