"""
Camera Detection — Step by Step
================================
Stage 1: Find the capsule-shaped camera module
Stage 2: Find the N largest circles INSIDE the capsule only
          (N is auto or manually set — no circles outside capsule possible)

Tune only the values in the TUNING BLOCK.
"""

import cv2
import numpy as np
import sys

# ══════════════════════════════════════════════════════════════
#  TUNING BLOCK
# ══════════════════════════════════════════════════════════════

# ── Stage 1: Capsule detection ────────────────────────────────
CAPSULE_CLOSE_ITER  = 4      # morphological close iterations to merge gaps
                              # raise if capsule outline breaks into pieces
CAPSULE_MIN_AREA    = 8_000  # px² — ignore tiny contours
CAPSULE_SOLIDITY    = 0.55   # contour area / convex hull area
                              # capsule is solid, so this should be high
                              # lower if the capsule contour has dents

# ── Stage 2: Circle detection inside capsule ─────────────────
N_CAMERAS           = 2      # how many camera lenses to expect (take N largest)
                              # S25 Edge = 2,  other models = 3 or 4
CIRCLE_PARAM2       = 25     # HoughCircles sensitivity (lower = finds more)
CIRCLE_MIN_R_FRAC   = 0.05   # min circle radius as fraction of capsule short side
CIRCLE_MAX_R_FRAC   = 0.50   # max circle radius as fraction of capsule short side

# ── Display ───────────────────────────────────────────────────
MAX_W, MAX_H        = 1200, 800

# ══════════════════════════════════════════════════════════════

IMAGE = "img34.png"
COLORS = [
    (0, 255, 0),    # green
    (0, 200, 255),  # yellow
    (255, 100, 0),  # blue
    (0, 100, 255),  # orange
]


def resize_display(img, max_w=MAX_W, max_h=MAX_H):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# ──────────────────────────────────────────────────────────────
# Stage 1: Find capsule
# ──────────────────────────────────────────────────────────────
def find_capsule(img):
    """
    Detects the oval/pill-shaped camera module.

    The capsule is large, solid, and has a distinct boundary from the
    phone back. We find it by:
      1. Canny edge map
      2. Large morphological close → merges ring edges into one filled blob
      3. Pick the largest contour that is not the full image frame
      4. Validate it looks like a solid blob (solidity check)

    Returns (contour, bounding_box_xyxy, mask_image) or None.
    """
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur   = cv2.bilateralFilter(gray, 9, 75, 75)
    edges  = cv2.Canny(blur, 20, 80)

    # Close kernel: large enough to bridge the gap between lens rings
    # and merge them into one solid capsule blob
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k,
                              iterations=CAPSULE_CLOSE_ITER)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = img.shape[:2]
    full_area    = h_img * w_img
    candidates   = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < CAPSULE_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Reject if it's practically the whole image
        if (w * h) > 0.80 * full_area:
            continue

        # Solidity = how filled the contour is (capsule should be solid)
        hull     = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if solidity < CAPSULE_SOLIDITY:
            continue

        candidates.append((area, cnt, x, y, w, h, solidity))

    if not candidates:
        return None

    # Take the largest valid candidate
    candidates.sort(reverse=True)
    _, best_cnt, x, y, w, h, sol = candidates[0]

    # Build a mask for the capsule interior (used to restrict circle search)
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(mask, [best_cnt], -1, 255, -1)

    print(f"  Capsule found: x={x} y={y} w={w} h={h}  solidity={sol:.2f}")
    return best_cnt, (x, y, x + w, y + h), mask


# ──────────────────────────────────────────────────────────────
# Stage 2: Find N largest circles inside capsule
# ──────────────────────────────────────────────────────────────
def find_lens_circles(img, capsule_box, capsule_mask):
    """
    Run HoughCircles on the cropped capsule ROI only.
    Returns the N_CAMERAS largest circles in full-image coordinates.
    """
    x1, y1, x2, y2 = capsule_box
    roi_bgr  = img[y1:y2, x1:x2]
    roi_mask = capsule_mask[y1:y2, x1:x2]

    gray   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # Apply capsule mask so edges outside don't confuse Hough
    gray   = cv2.bitwise_and(gray, roi_mask)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    short_side = min(x2 - x1, y2 - y1)
    min_r = max(15, int(short_side * CIRCLE_MIN_R_FRAC))
    max_r = int(short_side * CIRCLE_MAX_R_FRAC)

    circles = cv2.HoughCircles(
        smooth,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_r * 2,       # at least 2× min_r apart
        param1=60,
        param2=CIRCLE_PARAM2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        print("  No circles found inside capsule. Lower CIRCLE_PARAM2.")
        return []

    circles = np.round(circles[0]).astype(int)

    # Sort by radius descending, take top N_CAMERAS
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    circles = circles[:N_CAMERAS]

    # Shift back to full-image coordinates
    result = []
    for (cx, cy, r) in circles:
        result.append((cx + x1, cy + y1, r))
        print(f"  Lens circle: centre=({cx+x1},{cy+y1})  radius={r}px")

    return result


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    img = cv2.imread(IMAGE)
    if img is None:
        print(f"Cannot load {IMAGE}")
        sys.exit(1)

    result = img.copy()

    # ── Stage 1 ──────────────────────────────────────────────
    print("\n[Stage 1] Detecting capsule...")
    cap = find_capsule(img)

    if cap is None:
        print("  Capsule not found. Try:")
        print("  • Lower  CAPSULE_MIN_AREA  (if capsule is small in the image)")
        print("  • Lower  CAPSULE_SOLIDITY  (if capsule outline has gaps)")
        print("  • Raise  CAPSULE_CLOSE_ITER (if edges don't close into one blob)")
        # Fall back: search whole image
        h, w = img.shape[:2]
        capsule_box  = (0, 0, w, h)
        capsule_mask = np.full((h, w), 255, dtype=np.uint8)
        capsule_cnt  = None
    else:
        capsule_cnt, capsule_box, capsule_mask = cap
        # Draw capsule outline
        cv2.drawContours(result, [capsule_cnt], -1, (0, 220, 220), 2)
        x1, y1, x2, y2 = capsule_box
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 220, 220), 1)
        cv2.putText(result, "Camera Module", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 2)

    # ── Stage 2 ──────────────────────────────────────────────
    print(f"\n[Stage 2] Finding {N_CAMERAS} largest lens circles inside capsule...")
    lenses = find_lens_circles(img, capsule_box, capsule_mask)

    for i, (cx, cy, r) in enumerate(lenses):
        color = COLORS[i % len(COLORS)]
        cv2.circle(result, (cx, cy), r, color, 2)
        cv2.circle(result, (cx, cy), 4, color, -1)
        cv2.putText(result, f"Lens {i+1}  r={r}px",
                    (cx - r, cy - r - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

    # Summary
    txt = f"Capsule: {'OK' if cap else 'FALLBACK'}   Lenses: {len(lenses)}"
    cv2.rectangle(result, (4, 4), (len(txt) * 11 + 14, 28), (10, 10, 10), -1)
    cv2.putText(result, txt, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)

    # ── Display ───────────────────────────────────────────────
    cv2.imshow("Original",      resize_display(img))
    cv2.imshow("Outer Rings",   resize_display(result))

    cv2.imwrite("stage1_result.png", result)
    print("\nSaved: stage1_result.png")
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
