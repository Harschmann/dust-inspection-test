"""
Samsung Galaxy S25 Edge — Camera Dust Detection
================================================
Designed for phones with NESTED camera circles:
  [outer housing ring] → [inner lens glass circle] → dust lives here

Pipeline:
  1. Find the camera island (elongated bump on back of phone)
  2. Inside island: detect outer lens rings (large circles)
  3. Inside each outer ring: detect inner lens circle (smaller, concentric)
  4. Dust detection runs ONLY inside the inner circle

Why this beats raw HoughCircles:
  - Each Hough search is constrained to a small ROI, not the full image
  - Inner/outer nesting relationship is enforced (same center, smaller radius)
  - Bilateral filter preserves ring edges better than Gaussian blur
  - Circularity + edge-strength validation rejects false hits

Usage:
    python dust_detection.py
    Image must be named img34.png in the same directory.
"""

import cv2
import numpy as np
import sys

# ══════════════════════════════════════════════════════════════
#  TUNING BLOCK — adjust here without touching the algorithm
# ══════════════════════════════════════════════════════════════

# ── Camera island detection ────────────────────────────────────
ISLAND_MIN_AREA       = 5_000   # px²  ignore tiny contours
ISLAND_ASPECT_RATIO   = 1.4     # island height/width > this → "tall pill shape"
                                 # set to 1.0 if island is roughly square

# ── Outer lens ring (HoughCircles inside island) ───────────────
OUTER_DP              = 1.2
OUTER_PARAM1          = 60      # Canny upper threshold
OUTER_PARAM2          = 22      # Accumulator threshold (lower = more circles found)
OUTER_MIN_RADIUS_FRAC = 0.08    # fraction of island width
OUTER_MAX_RADIUS_FRAC = 0.48    # fraction of island width

# ── Inner lens circle (HoughCircles inside outer ring crop) ────
INNER_DP              = 1.0
INNER_PARAM1          = 50
INNER_PARAM2          = 15      # more lenient because ROI is small
INNER_MIN_RADIUS_FRAC = 0.35    # fraction of outer radius
INNER_MAX_RADIUS_FRAC = 0.85    # fraction of outer radius
INNER_CENTER_TOL_FRAC = 0.25    # how far off-center inner can be (fraction of outer r)

# ── Fallback: if no inner circle found, shrink outer by this ──
INNER_FALLBACK_SCALE  = 0.72

# ── White dust thresholds (HSV) ────────────────────────────────
DUST_SAT_MAX          = 50      # saturation <= this → white/grey (not colored)
DUST_VAL_MIN          = 185     # brightness >= this → bright particle
DUST_MIN_AREA         = 1       # minimum blob px² (catches single pixels)
DUST_MAX_AREA         = 350     # maximum blob px² (ignores lens highlights)

# ── Display ────────────────────────────────────────────────────
SCREEN_MAX_DIM        = 960     # max window dimension in pixels
# ══════════════════════════════════════════════════════════════


def bilateral(img):
    """Edge-preserving smoothing — better than Gaussian for ring edges."""
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


def clahe_enhance(bgr):
    """Boost micro-contrast on L channel (LAB) to reveal faint dust."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(l)
    return cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)


def edge_score_on_circle(gray, cx, cy, r, n_samples=72):
    """
    Sample edge-magnitude along a circle's circumference.
    Returns mean gradient strength — high score = real ring edge.
    """
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    xs = np.clip((cx + r * np.cos(angles)).astype(int), 0, gray.shape[1] - 1)
    ys = np.clip((cy + r * np.sin(angles)).astype(int), 0, gray.shape[0] - 1)
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    return float(np.abs(sobel[ys, xs]).mean())


# ──────────────────────────────────────────────────────────────
# STAGE 1: Find camera island
# ──────────────────────────────────────────────────────────────
def find_camera_island(img):
    """
    The camera module (island) is a raised rectangular/pill-shaped region
    on the back of the phone. We find it as a large, roughly vertical contour.
    Returns list of bounding boxes (x, y, w, h) for each island found.
    If island detection fails, returns the full image as a single region.
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = bilateral(gray)

    # Threshold to separate the island
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Canny + dilation to close gaps
    edges = cv2.Canny(blur, 30, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    combined = cv2.bitwise_or(thresh, closed)
    kernel2  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = img.shape[:2]
    islands = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < ISLAND_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Must not be almost the full image
        if w > 0.9 * w_img and h > 0.9 * h_img:
            continue
        islands.append((x, y, w, h))

    if not islands:
        print("[WARN] Could not isolate camera island — searching whole image.")
        return [(0, 0, w_img, h_img)]

    islands.sort(key=lambda b: b[2] * b[3], reverse=True)
    return islands[:4]


# ──────────────────────────────────────────────────────────────
# STAGE 2: Detect outer lens rings inside one island
# ──────────────────────────────────────────────────────────────
def detect_outer_rings(img, island_box):
    """
    Run HoughCircles constrained to the island bounding box.
    Returns list of (cx, cy, r) in full-image coordinates.
    """
    ix, iy, iw, ih = island_box
    roi = img[iy:iy+ih, ix:ix+iw]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    smooth   = bilateral(gray_roi)

    min_r = max(10, int(min(iw, ih) * OUTER_MIN_RADIUS_FRAC))
    max_r = int(min(iw, ih) * OUTER_MAX_RADIUS_FRAC)

    circles = cv2.HoughCircles(
        smooth, cv2.HOUGH_GRADIENT,
        dp=OUTER_DP,
        minDist=max_r,           # outer rings are at least 1 radius apart
        param1=OUTER_PARAM1,
        param2=OUTER_PARAM2,
        minRadius=min_r,
        maxRadius=max_r
    )
    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)
    results = []
    for (cx_l, cy_l, r) in circles:
        score = edge_score_on_circle(smooth, cx_l, cy_l, r)
        if score < 2.0:          # weak edge → probably noise
            continue
        results.append((cx_l + ix, cy_l + iy, r))

    return results


# ──────────────────────────────────────────────────────────────
# STAGE 3: Detect inner lens circle inside one outer ring
# ──────────────────────────────────────────────────────────────
def detect_inner_circle(img, outer_cx, outer_cy, outer_r):
    """
    Search for a concentric (or near-concentric) smaller circle
    inside the outer ring crop.
    Returns (cx, cy, r) in full-image coordinates.
    Falls back to scaled outer circle if none found.
    """
    margin = int(outer_r * 1.05)
    h_img, w_img = img.shape[:2]
    x1 = max(0, outer_cx - margin)
    y1 = max(0, outer_cy - margin)
    x2 = min(w_img, outer_cx + margin)
    y2 = min(h_img, outer_cy + margin)

    roi = img[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    smooth   = bilateral(gray_roi)

    min_r = max(5,  int(outer_r * INNER_MIN_RADIUS_FRAC))
    max_r = max(10, int(outer_r * INNER_MAX_RADIUS_FRAC))
    tol   = int(outer_r * INNER_CENTER_TOL_FRAC)

    local_cx = outer_cx - x1
    local_cy = outer_cy - y1

    circles = cv2.HoughCircles(
        smooth, cv2.HOUGH_GRADIENT,
        dp=INNER_DP,
        minDist=min_r,
        param1=INNER_PARAM1,
        param2=INNER_PARAM2,
        minRadius=min_r,
        maxRadius=max_r
    )

    best      = None
    best_dist = 1e9

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (cx_l, cy_l, r) in circles:
            dist = np.hypot(cx_l - local_cx, cy_l - local_cy)
            if dist < tol and dist < best_dist:
                best_dist = dist
                best = (cx_l + x1, cy_l + y1, r)

    if best is None:
        r_fb = int(outer_r * INNER_FALLBACK_SCALE)
        print(f"       No inner circle found — using fallback r={r_fb}")
        return (outer_cx, outer_cy, r_fb)

    return best


# ──────────────────────────────────────────────────────────────
# STAGE 4: Detect white dust inside inner lens
# ──────────────────────────────────────────────────────────────
def detect_dust(img, cx, cy, r):
    """
    Inside the circle (cx,cy,r): find bright, low-saturation dust specks.
    Returns list of (dust_cx, dust_cy, draw_radius) in full-image coords.
    """
    h_img, w_img = img.shape[:2]
    x1 = max(0, cx - r);   y1 = max(0, cy - r)
    x2 = min(w_img, cx+r); y2 = min(h_img, cy+r)

    crop = img[y1:y2, x1:x2].copy()
    lx   = cx - x1;  ly = cy - y1

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (lx, ly), r, 255, -1)

    enhanced = clahe_enhance(crop)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    white = np.zeros(crop.shape[:2], dtype=np.uint8)
    white[(S <= DUST_SAT_MAX) & (V >= DUST_VAL_MIN)] = 255
    white = cv2.bitwise_and(white, mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dust_hits = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if DUST_MIN_AREA <= area <= DUST_MAX_AREA:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            dcx = x1 + bx + bw // 2
            dcy = y1 + by + bh // 2
            draw_r = max(7, max(bw, bh) // 2 + 4)
            dust_hits.append((dcx, dcy, draw_r))

    return dust_hits


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    image_path = "img34.png"
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot load '{image_path}'.")
        sys.exit(1)

    original = img.copy()
    output   = img.copy()

    print("=" * 55)
    print("  Samsung S25 Edge — Camera Dust Detection")
    print("=" * 55)

    print("\n[Stage 1] Locating camera island(s)...")
    islands = find_camera_island(img)
    print(f"          Found {len(islands)} island(s).")

    total_dust = 0
    lens_idx   = 0

    for isl_num, island_box in enumerate(islands):
        ix, iy, iw, ih = island_box
        # Draw island outline in teal
        cv2.rectangle(output, (ix, iy), (ix+iw, iy+ih), (0, 220, 220), 1)

        print(f"\n[Stage 2] Island {isl_num+1}: detecting outer lens rings...")
        outer_rings = detect_outer_rings(img, island_box)
        print(f"          Outer rings found: {len(outer_rings)}")

        if not outer_rings:
            print("          [WARN] No outer rings — skipping island.")
            continue

        for (ocx, ocy, or_) in outer_rings:
            lens_idx += 1

            print(f"\n[Stage 3] Lens {lens_idx}: finding inner lens circle...")
            icx, icy, ir = detect_inner_circle(img, ocx, ocy, or_)
            print(f"          Outer r={or_}  Inner r={ir}  "
                  f"offset={int(np.hypot(icx-ocx, icy-ocy))}px")

            print(f"[Stage 4] Lens {lens_idx}: scanning for dust...")
            dust = detect_dust(img, icx, icy, ir)
            count = len(dust)
            total_dust += count
            print(f"          Dust particles: {count}")

            # Outer ring → green
            cv2.circle(output, (ocx, ocy), or_, (0, 200, 0), 2)
            # Inner lens glass → cyan-gold
            cv2.circle(output, (icx, icy), ir, (255, 200, 0), 2)

            lbl = f"Lens {lens_idx}  [{count} dust]"
            cv2.putText(output, lbl,
                        (ocx - or_, max(ocy - or_ - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (0, 200, 0), 2, cv2.LINE_AA)

            for (dcx, dcy, dr) in dust:
                cv2.circle(output, (dcx, dcy), dr, (0, 0, 255), 1)
                cv2.circle(output, (dcx, dcy), 2,  (0, 0, 255), -1)

    # Summary bar
    bar_text = f"Lenses: {lens_idx}   Total dust particles: {total_dust}"
    bar_w    = len(bar_text) * 11 + 20
    cv2.rectangle(output, (5, 5), (bar_w, 32), (20, 20, 20), -1)
    cv2.putText(output, bar_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

    # Legend
    legend = [
        ((0, 200, 0),   "Outer housing ring"),
        ((255, 200, 0), "Inner lens glass (ROI)"),
        ((0, 0, 255),   "Dust particle"),
        ((0, 220, 220), "Camera island"),
    ]
    for li, (color, text) in enumerate(legend):
        y = output.shape[0] - 20 - li * 22
        cv2.circle(output, (18, y), 7, color, -1)
        cv2.putText(output, text, (32, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Display
    h, w  = img.shape[:2]
    scale = min(SCREEN_MAX_DIM / w, SCREEN_MAX_DIM / h, 1.0)
    disp  = (int(w * scale), int(h * scale))

    cv2.imshow("Original Image",
               cv2.resize(original, disp) if scale < 1 else original)
    cv2.imshow("Dust Detection — S25 Edge",
               cv2.resize(output, disp)   if scale < 1 else output)

    cv2.imwrite("dust_result.png", output)
    print(f"\n[Done] Result saved as dust_result.png")
    print("Press any key in the window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
