import os
import html as html_module
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def get_logo_points(logo_path, num_points=1000, scale=2.0, offset=(100, 100)):
    print(f"Processing logo: {logo_path}")
    try:
        img = Image.open(logo_path).convert("RGBA")
        
        # Create a white background and paste the image over it using the alpha channel as a mask
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3] if 'A' in img.mode else None)
        
        # Convert to grayscale and resize
        img = bg.convert("L").resize((150, 150), Image.Resampling.LANCZOS)
        
        # Invert so dark pixels become high values
        img = ImageOps.invert(img)
        arr = np.array(img)
        
        # Threshold to find dark pixels of the logo
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

def generate_svg(dark_mode, pts_sequence, user_details):
    print(f"Generating {'dark' if dark_mode else 'light'} SVG...")
    
    bg_color = "#0A101F" if dark_mode else "#F8FAFC"
    box_bg = "rgba(255,255,255,0.03)" if dark_mode else "rgba(0,0,0,0.03)"
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
    svg.append(f'<line x1="520" y1="110" x2="1090" y2="110" stroke="url(#grad)" opacity="0.3" stroke-width="1"/>')
    
    y_offset = 150
    for key, val in user_details.items():
        escaped_val = html_module.escape(str(val))
        svg.append(f'<text x="520" y="{y_offset}" font-family="Arial, sans-serif" font-weight="600" font-size="16" fill="{text_secondary}">{key.upper()}</text>')
        
        bar_width = min(400, max(100, len(val) * 12))
        
        svg.append(f'<rect x="680" y="{y_offset - 12}" width="400" height="14" rx="4" fill="{box_bg}" />')
        svg.append(f'<rect x="680" y="{y_offset - 12}" width="{bar_width}" height="14" rx="4" fill="url(#grad)" opacity="0.8" />')
        
        svg.append(f'<text x="690" y="{y_offset}" font-family="monospace" font-weight="bold" font-size="13" fill="{text_primary}">{escaped_val}</text>')
        y_offset += 45
        
    # Morphing Points
    svg.append(f'<g fill="{point_color}" shape-rendering="crispEdges">')
    
    num_states = len(pts_sequence)
    # We want each state to hold for a bit, then transition.
    # For N states, we have 2N keyframes (hold, move, hold, move...)
    # The last state transitions back to the first state.
    # Total duration = N * 5 seconds
    dur = num_states * 5
    
    # Generate keyTimes
    # Example for N=3: 0, 0.2, 0.33, 0.53, 0.66, 0.86, 1.0
    key_times = []
    time_step = 1.0 / num_states
    hold_ratio = 0.6  # 60% of the time is holding, 40% moving
    
    for i in range(num_states):
        start_hold = i * time_step
        end_hold = start_hold + (time_step * hold_ratio)
        key_times.extend([f"{start_hold:.3f}", f"{end_hold:.3f}"])
    key_times.append("1.000")
    kt_str = "; ".join(key_times)
    
    for i in range(len(pts_sequence[0])):
        vx_vals = []
        vy_vals = []
        for state_idx in range(num_states):
            cx = round(pts_sequence[state_idx][i, 0], 1)
            cy = round(pts_sequence[state_idx][i, 1], 1)
            vx_vals.extend([str(cx), str(cx)])
            vy_vals.extend([str(cy), str(cy)])
        
        # Add the first state at the end to complete the loop
        cx0 = round(pts_sequence[0][i, 0], 1)
        cy0 = round(pts_sequence[0][i, 1], 1)
        vx_vals.append(str(cx0))
        vy_vals.append(str(cy0))
        
        vx = "; ".join(vx_vals)
        vy = "; ".join(vy_vals)
        
        svg.append(f'<circle cx="{cx0}" cy="{cy0}" r="1.5">')
        svg.append(f'<animate attributeName="cx" values="{vx}" keyTimes="{kt_str}" dur="{dur}s" repeatCount="indefinite"/>')
        svg.append(f'<animate attributeName="cy" values="{vy}" keyTimes="{kt_str}" dur="{dur}s" repeatCount="indefinite"/>')
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
    
    # Get all images from assets folder
    assets_dir = "assets"
    valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}
    logos = [os.path.join(assets_dir, f) for f in os.listdir(assets_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not logos:
        print("No images found in assets folder!")
        return

    print(f"Found {len(logos)} images to animate: {logos}")
    
    num_pts = 1000
    
    # Scale logos up to fill the left space
    # Portrait was 150x150 scaled by 2.5 + offset(50,50) => 375x375
    # So we use scale=2.5 and offset=(50, 50) for the logos
    logo_offset = (50, 50)
    logo_scale = 2.5
    
    pts_sequence = []
    
    # Process first logo
    current_pts = get_logo_points(logos[0], num_points=num_pts, scale=logo_scale, offset=logo_offset)
    pts_sequence.append(current_pts)
    
    # Process the rest sequentially and match
    for logo in logos[1:]:
        next_pts = get_logo_points(logo, num_points=num_pts, scale=logo_scale, offset=logo_offset)
        matched_pts = match_points(current_pts, next_pts)
        pts_sequence.append(matched_pts)
        current_pts = matched_pts
        
    # Match the last logo back to the first to complete the loop
    pts_sequence[0] = match_points(current_pts, pts_sequence[0])
    
    # Generate SVGs
    svg_dark = generate_svg(True, pts_sequence, user_details)
    svg_light = generate_svg(False, pts_sequence, user_details)
    
    with open("output/dark.svg", "w") as f:
        f.write(svg_dark)
    with open("output/light.svg", "w") as f:
        f.write(svg_light)
        
    print("Done! SVGs generated in output/")

if __name__ == "__main__":
    main()
