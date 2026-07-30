import os
import html as html_module
import sys
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import math

def process_photo_to_points(photo_path, num_points=1500, dark_mode=True):
    print("Processing photo to points...")
    try:
        from rembg import remove
        img = Image.open(photo_path).convert("RGBA")
        if dark_mode:
            img = remove(img)
            
        bg_color = (0, 0, 0, 255) if dark_mode else (255, 255, 255, 255)
        bg = Image.new("RGBA", img.size, bg_color)
        bg.paste(img, mask=img.split()[3] if dark_mode else None)
        img = bg.convert("L")
        
        width, height = img.size
        min_dim = min(width, height)
        img = img.crop((width//2 - min_dim//2, 0, width//2 + min_dim//2, min_dim))
        
        # Resize small so dithering doesn't create too many points
        img = img.resize((150, 150), Image.Resampling.LANCZOS)
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
        
        arr = np.array(img, dtype=float)
        h, w = arr.shape
        out = np.zeros_like(arr, dtype=bool)
        
        for y in range(h):
            x_range = range(w) if y % 2 == 0 else range(w-1, -1, -1)
            for x in x_range:
                old_val = arr[y, x]
                new_val = 255 if old_val > 128 else 0
                out[y, x] = (new_val == 0)
                err = old_val - new_val
                dir_x = 1 if y % 2 == 0 else -1
                if 0 <= x + dir_x < w: arr[y, x + dir_x] += err * 7 / 16
                if y + 1 < h:
                    if 0 <= x - dir_x < w: arr[y + 1, x - dir_x] += err * 3 / 16
                    arr[y + 1, x] += err * 5 / 16
                    if 0 <= x + dir_x < w: arr[y + 1, x + dir_x] += err * 1 / 16
                    
        y_idx, x_idx = np.where(out)
        pts = np.column_stack((x_idx, y_idx))
        
        # Scale and offset photo points to fit left side
        pts = pts * 2.5 + np.array([50, 50])
        
        # Sample exactly num_points
        if len(pts) > num_points:
            indices = np.random.choice(len(pts), num_points, replace=False)
            pts = pts[indices]
        elif len(pts) < num_points:
            padding = np.zeros((num_points - len(pts), 2))
            pts = np.vstack([pts, padding])
            
        return pts
    except Exception as e:
        print(f"Error processing photo: {e}")
        return np.zeros((num_points, 2))

def get_logo_points(logo_path, num_points=1500, scale=1.5, offset=(100, 100)):
    print(f"Processing logo: {logo_path}")
    try:
        img = Image.open(logo_path).convert("L")
        img = img.resize((150, 150), Image.Resampling.LANCZOS)
        img = ImageOps.invert(img)
        arr = np.array(img)
        y_idx, x_idx = np.where(arr > 128)
        
        if len(y_idx) == 0:
            raise ValueError("No points found")
            
        indices = np.random.choice(len(y_idx), min(num_points, len(y_idx)), replace=False)
        pts = np.column_stack((x_idx[indices], y_idx[indices]))
        
        pts = pts * scale + np.array(offset)
        
        if len(pts) < num_points:
            padding = np.zeros((num_points - len(pts), 2))
            pts = np.vstack([pts, padding])
            
        return pts
    except Exception as e:
        print(f"Error with logo {logo_path}: {e}")
        angles = np.random.uniform(0, 2*np.pi, num_points)
        r = np.random.uniform(0, 50, num_points)
        return np.column_stack((r*np.cos(angles), r*np.sin(angles))) + np.array(offset)

def match_points(pts1, pts2):
    print("Matching points (Optimal Transport)...")
    dists = cdist(pts1, pts2)
    row_ind, col_ind = linear_sum_assignment(dists)
    return pts2[col_ind]

def generate_svg(dark_mode, pts_list, user_details):
    print(f"Generating {'dark' if dark_mode else 'light'} SVG...")
    
    bg_color = "#0A101F" if dark_mode else "#F8FAFC"
    box_bg = "rgba(255,255,255,0.03)" if dark_mode else "rgba(0,0,0,0.03)"
    box_border = "rgba(255,255,255,0.1)" if dark_mode else "rgba(0,0,0,0.1)"
    text_primary = "#FFFFFF" if dark_mode else "#0F172A"
    text_secondary = "#94A3B8" if dark_mode else "#475569"
    accent_color = "#22D3EE" if dark_mode else "#0ea5e9"
    point_color = "#A78BFA" if dark_mode else "#8b5cf6"
    
    width, height = 1180, 610
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    svg.append(f'<defs>')
    svg.append(f'<linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="{accent_color}"/><stop offset="100%" stop-color="{point_color}"/></linearGradient>')
    svg.append(f'</defs>')
    svg.append(f'<rect width="{width}" height="{height}" fill="{bg_color}" />')
    
    # Enhanced Text Box
    svg.append(f'<rect x="480" y="40" width="650" height="530" rx="16" fill="{box_bg}" stroke="url(#grad)" stroke-width="2.5" />')
    
    svg.append(f'<text x="520" y="90" font-family="Arial, sans-serif" font-weight="bold" font-size="28" fill="{text_primary}">SYSTEM <tspan fill="url(#grad)">INITIALIZED</tspan></text>')
    svg.append(f'<line x1="520" y1="110" x2="1090" y2="110" stroke="{box_border}" stroke-width="1"/>')
    
    y_offset = 150
    for key, val in user_details.items():
        escaped_val = html_module.escape(str(val))
        svg.append(f'<text x="520" y="{y_offset}" font-family="Arial, sans-serif" font-weight="600" font-size="16" fill="{text_secondary}">{key.upper()}</text>')
        
        # Determine bar width based on arbitrary metric or text length
        bar_width = min(400, max(100, len(val) * 12))
        
        # "Progress" Bar background
        svg.append(f'<rect x="680" y="{y_offset - 12}" width="400" height="14" rx="4" fill="{box_bg}" />')
        # "Progress" Bar fill
        svg.append(f'<rect x="680" y="{y_offset - 12}" width="{bar_width}" height="14" rx="4" fill="url(#grad)" opacity="0.8" />')
        
        svg.append(f'<text x="690" y="{y_offset}" font-family="monospace" font-weight="bold" font-size="13" fill="{text_primary}">{escaped_val}</text>')
        y_offset += 45
        
    # Morphing Points
    svg.append(f'<g fill="{point_color}" shape-rendering="crispEdges">')
    
    # pts_list contains: [portrait, logo1, logo2, logo3]
    # Sequence: P -> P -> L1 -> L1 -> L2 -> L2 -> L3 -> L3 -> P
    # We want hold periods and transition periods.
    # KeyTimes: 0 (P), 0.15 (P), 0.25 (L1), 0.40 (L1), 0.50 (L2), 0.65 (L2), 0.75 (L3), 0.90 (L3), 1.0 (P)
    kt = "0; 0.15; 0.25; 0.40; 0.50; 0.65; 0.75; 0.90; 1.0"
    
    p0, p1, p2, p3 = pts_list
    for i in range(len(p0)):
        cx0, cy0 = round(p0[i,0],1), round(p0[i,1],1)
        cx1, cy1 = round(p1[i,0],1), round(p1[i,1],1)
        cx2, cy2 = round(p2[i,0],1), round(p2[i,1],1)
        cx3, cy3 = round(p3[i,0],1), round(p3[i,1],1)
        
        vx = f"{cx0}; {cx0}; {cx1}; {cx1}; {cx2}; {cx2}; {cx3}; {cx3}; {cx0}"
        vy = f"{cy0}; {cy0}; {cy1}; {cy1}; {cy2}; {cy2}; {cy3}; {cy3}; {cy0}"
        
        svg.append(f'<circle cx="{cx0}" cy="{cy0}" r="1.5">')
        svg.append(f'<animate attributeName="cx" values="{vx}" keyTimes="{kt}" dur="20s" repeatCount="indefinite"/>')
        svg.append(f'<animate attributeName="cy" values="{vy}" keyTimes="{kt}" dur="20s" repeatCount="indefinite"/>')
        svg.append(f'</circle>')
        
    svg.append('</g>')
    svg.append('</svg>')
    return "\n".join(svg)

def main():
    os.makedirs("output", exist_ok=True)
    
    user_details = {
        'Name': 'Naman Vijay Upadhyay',
        'Role': 'AI/ML & Backend Dev',
        'Location': 'Pune, India',
        'Status': 'Building AI Solutions',
        'Languages': 'Python, C, C++, SQL',
        'Frontend': 'HTML, CSS, Streamlit, React',
        'Backend': 'FastAPI, REST APIs',
        'Database': 'MySQL, PostgreSQL',
        'Infra': 'Docker, AWS EC2',
    }
    
    source_photo = "assets/source_photo.jpg.jpeg"
    logos = [
        "assets/Ai_image_logo.png",
        "assets/Coder_image_logo.jpg",
        "assets/Coding_image_logo.png"
    ]
    
    num_pts = 1000  # Keep reasonable to stay under GitHub's 1MB limit
    
    # Process portrait
    pts_portrait_dark = process_photo_to_points(source_photo, num_points=num_pts, dark_mode=True)
    pts_portrait_light = process_photo_to_points(source_photo, num_points=num_pts, dark_mode=False)
    
    # Process logos (offset to center them in the portrait area)
    # The portrait is offset 50,50 and scaled to ~ 150*2.5 = 375 => Center is roughly 230, 230
    logo_offset = (150, 150)
    pts1 = get_logo_points(logos[0], num_points=num_pts, scale=1.5, offset=logo_offset)
    pts2 = match_points(pts1, get_logo_points(logos[1], num_points=num_pts, scale=1.5, offset=logo_offset))
    pts3 = match_points(pts2, get_logo_points(logos[2], num_points=num_pts, scale=1.5, offset=logo_offset))
    
    # Match logos back to portrait to complete loop smoothly
    pts_portrait_dark_ordered = match_points(pts3, pts_portrait_dark)
    pts_portrait_light_ordered = match_points(pts3, pts_portrait_light)
    
    svg_dark = generate_svg(True, [pts_portrait_dark_ordered, pts1, pts2, pts3], user_details)
    svg_light = generate_svg(False, [pts_portrait_light_ordered, pts1, pts2, pts3], user_details)
    
    with open("output/dark.svg", "w") as f:
        f.write(svg_dark)
    with open("output/light.svg", "w") as f:
        f.write(svg_light)
        
    print("Done! SVGs generated in output/")

if __name__ == "__main__":
    main()
