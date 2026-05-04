"""
Samsung Galaxy S25 Edge — Dust Detection (Binary Threshold Approach)
=====================================================================

Core insight (user-provided):
  Dust is WHITE and BRIGHT. Everything else — blue plastic, grey rings,
  golden borders, black gaps — is either colored (high saturation) or dark.
  So: threshold the whole image on brightness + saturation → dust turns white,
  everything else turns black. No complex color logic needed.

ROI approach (simplified):
  The actual glass/sensor area is the DARKEST roughly-circular region inside
  the camera island. We find it by looking for dark circular blobs, not by
  chaining multiple HoughCircles calls. This avoids the nested-Hough failures.

Three windows:
  1. Original image
  2. Binary mask — whole image (black bg, white dust everywhere)
  3. Result — original with dust circles marked, ROI boundary shown
"""

import cv2
import numpy as np
import sys

# ══════════════════════════════════════════════════════════════════════
#  TUNING — change these values, nothing else
# ══════════════════════════════════════════════════════════════════════

# ── Dust threshold (applied to whole image) ───────────────────────────
# Dust = HIGH brightness AND LOW saturation (white/near-white)
DUST_VAL_MIN   = 180    # V channel min  (0-255). Lower → catch dimmer dust
DUST_SAT_MAX   = 60     # S channel max  (0-255). Higher → allow slightly tinted dust

# ── Noise removal after threshold ────────────────────────────────────
MORPH_OPEN_PX  = 2      # morphological open kernel size (removes salt noise)
MORPH_CLOSE_PX = 3      # morphological close kernel size (fills dust blob gaps)

# ── Dust blob size filter ─────────────────────────────────────────────
DUST_MIN_AREA  = 1      # px² — catches single bright pixels
DUST_MAX_AREA  = 400    # px² — rejects large lens reflections / highlights

# ── ROI: camera glass detection ──────────────────────────────────────
# The glass sensor area is the darkest large circular region in the image.
# These control what counts as "dark enough to be camera glass".
GLASS_DARK_THRESH   = 80    # pixels below this intensity are "dark" (glass bg)
GLASS_MIN_RADIUS    = 30    # px — ignore tiny dark blobs
GLASS_MAX_RADIUS    = 500   # px — ignore huge dark regions (phone body)
GLASS_MIN_FILL      = 0.35  # fraction of circle area that must be dark (circularity check)
GLASS_HOUGH_PARAM2  = 20    # HoughCircles accumulator threshold (↓ = find more)

# ── Display ───────────────────────────────────────────────────────────
SCREEN_W = 960   # max display width per window
SCREEN_H = 700   # max display height per window

# ══════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────
# STEP 1: Global binary threshold — dust = white, everything else = black
# ─────────────────────────────────────────────────────────────────────
def make_dust_binary(img):
    """
    Convert entire image to binary:
      WHITE = bright pixel with low saturation  → dust candidate
      BLACK = everything else (colored rings, dark glass, blue plastic, etc.)

    Uses HSV so saturation and brightness are independent channels.
    CLAHE on the L channel first to boost contrast of very faint dust.
    """
    # 1. Boost contrast on luminance only (LAB space), don't touch hue/sat
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 2. HSV threshold
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    binary = np.zeros(img.shape[:2], dtype=np.uint8)
    binary[(V >= DUST_VAL_MIN) & (S <= DUST_SAT_MAX)] = 255

    # 3. Remove single-pixel salt noise
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (MORPH_OPEN_PX, MORPH_OPEN_PX))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    # 4. Fill tiny gaps within a dust blob
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_CLOSE_PX, MORPH_CLOSE_PX))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    return binary


# ─────────────────────────────────────────────────────────────────────
# STEP 2: ROI detection — find the dark glass circles (sensor areas)
# ─────────────────────────────────────────────────────────────────────
def find_glass_rois(img):
    """
    The sensor glass is the darkest, roughly circular region inside the camera.
    Strategy:
      a. Threshold the grayscale image for DARK pixels only.
      b. Run HoughCircles on the dark-pixel map.
      c. For each candidate circle, check that enough of its interior is dark
         (this rejects bright circular reflections and edge artefacts).

    Returns list of (cx, cy, r) in image coordinates.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bilateral filter preserves sharp ring edges
    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=60, sigmaSpace=60)

    # Detect circles on the full smoothed image
    circles = cv2.HoughCircles(
        smooth,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=GLASS_MIN_RADIUS * 2,
        param1=60,
        param2=GLASS_HOUGH_PARAM2,
        minRadius=GLASS_MIN_RADIUS,
        maxRadius=GLASS_MAX_RADIUS
    )

    if circles is None:
        print("  [ROI] No circles found — will mark dust on full image.")
        return []

    circles = np.round(circles[0]).astype(int)
    h_img, w_img = img.shape[:2]

    # Dark pixel map for fill-ratio check
    _, dark_map = cv2.threshold(smooth, GLASS_DARK_THRESH, 255, cv2.THRESH_BINARY_INV)

    valid = []
    for (cx, cy, r) in circles:
        # Create a circular mask and measure dark pixel fill
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        circle_area = np.pi * r * r
        dark_pixels  = cv2.countNonZero(cv2.bitwise_and(dark_map, mask))
        fill_ratio   = dark_pixels / circle_area

        if fill_ratio >= GLASS_MIN_FILL:
            valid.append((cx, cy, r, fill_ratio))
            print(f"  [ROI] Circle accepted  center=({cx},{cy}) r={r}  "
                  f"dark_fill={fill_ratio:.2f}")
        else:
            print(f"  [ROI] Circle rejected  center=({cx},{cy}) r={r}  "
                  f"dark_fill={fill_ratio:.2f} < {GLASS_MIN_FILL}")

    # Sort by radius descending (main lens first)
    valid.sort(key=lambda x: x[2], reverse=True)
    return [(cx, cy, r) for cx, cy, r, _ in valid]


# ─────────────────────────────────────────────────────────────────────
# STEP 3: Filter dust blobs — keep only those inside ROI circles
# ─────────────────────────────────────────────────────────────────────
def filter_dust_in_rois(binary, rois, img_shape):
    """
    From the global binary mask, keep only dust blobs that fall
    inside one of the detected ROI circles.
    If no ROIs were found, uses the full binary (no masking).
    Returns:
      - roi_binary  : binary mask restricted to ROI areas
      - dust_points : list of (cx, cy, draw_r) for each confirmed dust blob
    """
    h, w = img_shape[:2]

    if not rois:
        # No ROI — work on full image
        roi_mask   = np.full((h, w), 255, dtype=np.uint8)
    else:
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        for (cx, cy, r) in rois:
            cv2.circle(roi_mask, (cx, cy), r, 255, -1)

    roi_binary = cv2.bitwise_and(binary, roi_mask)

    # Find blobs and filter by area
    cnts, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    dust_points = []
    confirmed_binary = np.zeros((h, w), dtype=np.uint8)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if DUST_MIN_AREA <= area <= DUST_MAX_AREA:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            dcx    = bx + bw // 2
            dcy    = by + bh // 2
            draw_r = max(7, max(bw, bh) // 2 + 4)
            dust_points.append((dcx, dcy, draw_r))
            cv2.drawContours(confirmed_binary, [cnt], -1, 255, -1)

    return confirmed_binary, dust_points


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def fit_to_screen(img):
    h, w = img.shape[:2]
    scale = min(SCREEN_W / w, SCREEN_H / h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def draw_legend(img, items):
    """items = list of (bgr_color, label)"""
    for i, (color, label) in enumerate(items):
        y = img.shape[0] - 15 - i * 22
        cv2.circle(img, (16, y), 7, color, -1)
        cv2.putText(img, label, (30, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    image_path = "img34.png"
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot load '{image_path}'")
        sys.exit(1)

    print("=" * 60)
    print("  Samsung S25 Edge — Dust Detection (Binary Threshold)")
    print("=" * 60)

    # ── Step 1: global binary dust mask ──────────────────────────
    print("\n[Step 1] Building global binary dust mask...")
    binary_global = make_dust_binary(img)
    non_zero = cv2.countNonZero(binary_global)
    print(f"         White pixels before ROI filter: {non_zero}")

    # ── Step 2: find glass ROIs ───────────────────────────────────
    print("\n[Step 2] Finding camera glass ROI circles...")
    rois = find_glass_rois(img)
    print(f"         Valid ROI circles found: {len(rois)}")

    # ── Step 3: filter dust to ROI ────────────────────────────────
    print("\n[Step 3] Filtering dust blobs inside ROIs...")
    binary_roi, dust_points = filter_dust_in_rois(binary_global, rois, img.shape)
    print(f"         Dust particles confirmed: {len(dust_points)}")

    # ── Build result overlay ──────────────────────────────────────
    result = img.copy()

    # Draw ROI circles
    for idx, (cx, cy, r) in enumerate(rois):
        cv2.circle(result, (cx, cy), r, (0, 220, 100), 2)
        cv2.putText(result, f"Lens {idx+1}", (cx - r, cy - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 100), 2, cv2.LINE_AA)

    # Draw dust markers
    for (dcx, dcy, dr) in dust_points:
        cv2.circle(result, (dcx, dcy), dr, (0, 0, 255), 1)
        cv2.circle(result, (dcx, dcy), 2,  (0, 0, 255), -1)

    # Summary bar
    summary = f"ROI circles: {len(rois)}   Dust particles: {len(dust_points)}"
    bar_w = len(summary) * 11 + 20
    cv2.rectangle(result, (5, 5), (bar_w, 30), (10, 10, 10), -1)
    cv2.putText(result, summary, (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

    draw_legend(result, [
        ((0, 220, 100), "Glass ROI boundary"),
        ((0, 0, 255),   "Dust particle"),
    ])

    # ── Build binary display image ────────────────────────────────
    # Stack global binary and ROI-filtered binary side by side for comparison
    # Convert to BGR so we can add color annotations
    bin_global_bgr = cv2.cvtColor(binary_global, cv2.COLOR_GRAY2BGR)
    bin_roi_bgr    = cv2.cvtColor(binary_roi,    cv2.COLOR_GRAY2BGR)

    # Draw ROI circles on the ROI binary image in green
    for (cx, cy, r) in rois:
        cv2.circle(bin_roi_bgr, (cx, cy), r, (0, 200, 0), 2)

    # Label the two halves
    cv2.putText(bin_global_bgr, "Global (full image)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
    cv2.putText(bin_roi_bgr, "ROI-filtered",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)

    # Resize both to same height before stacking
    h1, w1 = bin_global_bgr.shape[:2]
    scale_h = SCREEN_H / h1
    new_w   = int(w1 * scale_h)
    new_h   = SCREEN_H

    bin_g_resized = cv2.resize(bin_global_bgr, (new_w, new_h))
    bin_r_resized = cv2.resize(bin_roi_bgr,    (new_w, new_h))
    binary_display = np.hstack([bin_g_resized, bin_r_resized])

    # ── Show windows ──────────────────────────────────────────────
    cv2.imshow("1 — Original Image",      fit_to_screen(img))
    cv2.imshow("2 — Binary Dust Mask",    binary_display)
    cv2.imshow("3 — Detection Result",    fit_to_screen(result))

    # ── Save outputs ──────────────────────────────────────────────
    cv2.imwrite("dust_result.png",        result)
    cv2.imwrite("dust_binary_global.png", binary_global)
    cv2.imwrite("dust_binary_roi.png",    binary_roi)
    print("\n[Saved] dust_result.png  |  dust_binary_global.png  |  dust_binary_roi.png")
    print("Press any key in any window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
