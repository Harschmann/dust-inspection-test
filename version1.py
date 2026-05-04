"""
Samsung Camera Dust Detection
==============================
Detects white dust particles inside mobile camera lenses.
- Handles multiple lenses per image
- Ignores colored elements (blue lining, golden ring) using HSV saturation filtering
- Detects very small white dust particles using CLAHE + blob analysis

Usage:
    python dust_detection.py
    (image must be named img34.png in the same directory)
"""

import cv2
import numpy as np
import sys


# ─────────────────────────────────────────────
# TUNING PARAMETERS  (adjust these if needed)
# ─────────────────────────────────────────────
HOUGH_DP           = 1.2    # Inverse resolution ratio for HoughCircles
HOUGH_MIN_DIST     = 80     # Min pixel distance between detected lens centers
HOUGH_PARAM1       = 80     # Canny upper threshold inside HoughCircles
HOUGH_PARAM2       = 28     # Accumulator threshold (lower = detect more circles)
LENS_MIN_RADIUS    = 40     # Minimum lens radius in pixels
LENS_MAX_RADIUS    = 350    # Maximum lens radius in pixels

# White dust thresholds in HSV
DUST_SAT_MAX       = 55     # Saturation ≤ this → "white/grey" (not colored)
DUST_VAL_MIN       = 190    # Value (brightness) ≥ this → bright particle
DUST_MIN_AREA      = 1      # Minimum dust blob area in px² (catches single pixels)
DUST_MAX_AREA      = 400    # Maximum dust blob area in px² (ignores large reflections)

INNER_RADIUS_SCALE = 0.92   # Crop slightly inside detected edge to avoid outer ring artifacts
# ─────────────────────────────────────────────


def enhance_for_dust(bgr_roi):
    """
    Apply CLAHE on the L channel of LAB to boost tiny dust contrast,
    then convert to HSV for color-aware white detection.
    """
    lab = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return enhanced


def detect_lenses(gray_img):
    """
    Detect circular camera lenses using HoughCircles.
    Returns list of (cx, cy, r) tuples.
    """
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=LENS_MIN_RADIUS,
        maxRadius=LENS_MAX_RADIUS
    )
    if circles is None:
        return []
    circles = np.round(circles[0, :]).astype(int)
    return [(c[0], c[1], c[2]) for c in circles]


def detect_dust_in_lens(img, cx, cy, r):
    """
    Given the full image and a detected lens circle (cx, cy, r):
      1. Crop the bounding box of the lens.
      2. Mask to inner circle only (ignoring outer ring artifacts).
      3. Enhance contrast with CLAHE.
      4. Threshold for white/grey particles in HSV.
      5. Remove noise with morphological opening.
      6. Return list of dust contours (in full-image coordinates).
    """
    inner_r = int(r * INNER_RADIUS_SCALE)

    # Bounding box (clamped to image edges)
    h_img, w_img = img.shape[:2]
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w_img, cx + r)
    y2 = min(h_img, cy + r)

    lens_bgr = img[y1:y2, x1:x2].copy()

    # Local center of the cropped region
    local_cx = cx - x1
    local_cy = cy - y1

    # Create circular mask (inner circle only)
    circle_mask = np.zeros(lens_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, (local_cx, local_cy), inner_r, 255, -1)

    # Enhance contrast
    enhanced = enhance_for_dust(lens_bgr)

    # Convert to HSV
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # White dust mask:
    #   - Low saturation  → not colored (not blue lining, not gold)
    #   - High value      → bright particle
    white_mask = np.zeros(lens_bgr.shape[:2], dtype=np.uint8)
    white_mask[(S <= DUST_SAT_MAX) & (V >= DUST_VAL_MIN)] = 255

    # Restrict to lens interior
    white_mask = cv2.bitwise_and(white_mask, circle_mask)

    # Morphological opening: remove single-pixel salt noise but keep small blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of dust particles
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dust_contours_full = []  # contours in full-image coordinates
    dust_boxes = []          # (x, y, w, h) in full-image coordinates

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if DUST_MIN_AREA <= area <= DUST_MAX_AREA:
            # Shift contour back to full-image coordinates
            shifted = cnt + np.array([[[x1, y1]]])
            dust_contours_full.append(shifted)
            bx, by, bw, bh = cv2.boundingRect(shifted)
            dust_boxes.append((bx, by, bw, bh))

    return dust_contours_full, dust_boxes, (x1, y1, x2, y2), white_mask, circle_mask


def main():
    image_path = "img34.png"
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load '{image_path}'. Make sure it is in the same folder.")
        sys.exit(1)

    original = img.copy()
    output   = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("[INFO] Detecting camera lenses...")
    lenses = detect_lenses(gray)

    if not lenses:
        print("[WARN] No lenses detected. Try lowering HOUGH_PARAM2 or adjusting radius range.")
    else:
        print(f"[INFO] Detected {len(lenses)} lens(es).")

    total_dust = 0

    for idx, (cx, cy, r) in enumerate(lenses):
        print(f"\n[Lens {idx + 1}]  center=({cx},{cy})  radius={r}px")

        dust_contours, dust_boxes, bbox, white_mask, circle_mask = detect_dust_in_lens(
            img, cx, cy, r
        )

        count = len(dust_contours)
        total_dust += count
        print(f"         Dust particles found: {count}")

        # ── Draw on output image ──────────────────────────────────────────

        # Green circle = detected lens boundary
        cv2.circle(output, (cx, cy), r, (0, 220, 0), 2)

        # Lens label
        label = f"Lens {idx + 1}  [{count} dust]"
        label_y = max(cy - r - 10, 20)
        cv2.putText(output, label, (cx - r, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)

        # Red boxes / circles around each dust particle
        for (bx, by, bw, bh) in dust_boxes:
            # Draw a small red circle centered on the dust blob
            dust_cx = bx + bw // 2
            dust_cy = by + bh // 2
            draw_r  = max(6, max(bw, bh) // 2 + 3)   # ensure visible even for 1-px blobs
            cv2.circle(output, (dust_cx, dust_cy), draw_r, (0, 0, 255), 1)
            cv2.circle(output, (dust_cx, dust_cy), 2,      (0, 0, 255), -1)  # dot at center

    # ── Summary text ─────────────────────────────────────────────────────
    summary = f"Lenses: {len(lenses)}   Total dust: {total_dust}"
    cv2.rectangle(output, (5, 5), (len(summary) * 11 + 10, 30), (0, 0, 0), -1)
    cv2.putText(output, summary, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # ── Show windows ──────────────────────────────────────────────────────
    # Resize if image is very large
    screen_max = 900
    h, w = img.shape[:2]
    scale = min(screen_max / w, screen_max / h, 1.0)
    if scale < 1.0:
        disp_size = (int(w * scale), int(h * scale))
        original_disp = cv2.resize(original, disp_size)
        output_disp   = cv2.resize(output,   disp_size)
    else:
        original_disp = original
        output_disp   = output

    cv2.imshow("Original Image", original_disp)
    cv2.imshow("Dust Detection Result", output_disp)

    # Save the result
    cv2.imwrite("dust_result.png", output)
    print(f"\n[INFO] Result saved as 'dust_result.png'")
    print("[INFO] Press any key in an OpenCV window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
