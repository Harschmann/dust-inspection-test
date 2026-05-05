"""
Camera Lens Detection via Color Segmentation
=============================================
Step 1 : Isolate the blue/cyan camera module body using HSV range
Step 2 : Find dark (black) circular blobs INSIDE that region — those are lenses
Step 3 : Draw + crop each lens

TUNING GUIDE  (only touch the TUNING BLOCK)
───────────────────────────────────────────
Blue/cyan module body
  HUE_LOW  / HUE_HIGH   → hue range  (cyan is ~85-100, blue ~100-130)
                           open a color picker, check your exact shade
  SAT_LOW               → how vivid the color must be (raise to reject grey)
  VAL_LOW               → how bright it must be (lower if module looks dark)

Dark lens circles inside module
  DARK_THRESH           → pixels darker than this are "lens" (raise if lens
                           looks grey rather than black, e.g. try 80-120)
  MIN_RADIUS            → smallest lens radius to accept  (pixels)
  MAX_RADIUS            → largest  lens radius to accept  (pixels)
  CIRCLE_PARAM2         → HoughCircles sensitivity inside the dark mask
                           lower  → finds more  circles (may add false ones)
                           higher → finds fewer circles (may miss one)
  N_LENSES              → how many lenses to keep (takes N largest circles)
"""

import cv2
import numpy as np
import sys

# ══════════════════════════════════════════════════════════════
#  TUNING BLOCK
# ══════════════════════════════════════════════════════════════

# ── Step 1: blue/cyan module color range (HSV) ────────────────
HUE_LOW   = 85    # hue lower bound  (0-179 in OpenCV)
HUE_HIGH  = 130   # hue upper bound
SAT_LOW   = 60    # minimum saturation (keeps vivid colors, rejects greys)
VAL_LOW   = 40    # minimum brightness (keeps even darkish blues)

# ── Step 2: dark lens blob inside the module ──────────────────
DARK_THRESH   = 60    # grayscale value — pixels below this = "dark lens"
MIN_RADIUS    = 30    # minimum circle radius in pixels
MAX_RADIUS    = 400   # maximum circle radius in pixels
CIRCLE_PARAM2 = 20    # HoughCircles accumulator threshold
N_LENSES      = 2     # keep this many largest circles  (2 for S25 Edge)

# ══════════════════════════════════════════════════════════════

IMAGE = "img34.png"

def fit(im, max_w=1000, max_h=750):
    h, w = im.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(im, (int(w * scale), int(h * scale))) if scale < 1 else im


img = cv2.imread(IMAGE)
if img is None:
    print(f"Cannot load {IMAGE}")
    sys.exit(1)

print(f"Image size: {img.shape[1]} w  x  {img.shape[0]} h")

# ── Step 1: isolate blue/cyan module ─────────────────────────
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([HUE_LOW,  SAT_LOW, VAL_LOW])
upper = np.array([HUE_HIGH, 255,     255    ])
module_mask = cv2.inRange(hsv, lower, upper)

# Clean up small specks and fill holes in the module mask
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5 ))
module_mask = cv2.morphologyEx(module_mask, cv2.MORPH_CLOSE, k_close, iterations=3)
module_mask = cv2.morphologyEx(module_mask, cv2.MORPH_OPEN,  k_open,  iterations=1)

pixels_found = cv2.countNonZero(module_mask)
print(f"\n[Step 1] Blue/cyan mask: {pixels_found} pixels matched")
if pixels_found == 0:
    print("  → Nothing matched. Adjust HUE_LOW / HUE_HIGH / SAT_LOW")

# Visualise module mask as a coloured overlay on original
module_vis = img.copy()
module_vis[module_mask > 0] = (module_vis[module_mask > 0] * 0.5
                               + np.array([0, 255, 128]) * 0.5).astype(np.uint8)

# ── Step 2: find dark circles inside the module mask ─────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Dark pixel map, restricted to the blue module region
_, dark = cv2.threshold(gray, DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
dark_in_module = cv2.bitwise_and(dark, module_mask)

# Smooth before Hough (bilateral keeps edges sharp)
smooth = cv2.bilateralFilter(dark_in_module, 9, 75, 75)

circles = cv2.HoughCircles(
    smooth,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=MIN_RADIUS * 2,
    param1=50,
    param2=CIRCLE_PARAM2,
    minRadius=MIN_RADIUS,
    maxRadius=MAX_RADIUS,
)

result = img.copy()

# Draw module contour
cnts, _ = cv2.findContours(module_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(result, cnts, -1, (0, 255, 200), 2)

COLORS = [(0, 255, 0), (0, 180, 255), (255, 80, 0), (0, 80, 255)]

crops = []
if circles is None:
    print("\n[Step 2] No circles found inside module.")
    print("  → Try: lower DARK_THRESH, lower CIRCLE_PARAM2, or adjust MIN/MAX_RADIUS")
else:
    circles = np.round(circles[0]).astype(int)
    # Sort by radius descending, keep top N_LENSES
    circles = sorted(circles, key=lambda c: c[2], reverse=True)[:N_LENSES]

    print(f"\n[Step 2] Lens circles found: {len(circles)}")
    h_img, w_img = img.shape[:2]

    for i, (cx, cy, r) in enumerate(circles):
        color = COLORS[i % len(COLORS)]
        print(f"  Lens {i+1}: centre=({cx},{cy})  radius={r}px")

        # Draw on result
        cv2.circle(result, (cx, cy), r, color, 2)
        cv2.circle(result, (cx, cy), 4, color, -1)
        cv2.putText(result, f"Lens {i+1}  r={r}",
                    (cx - r, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Crop the lens (bounding square, clamped to image)
        x1 = max(0, cx - r);   y1 = max(0, cy - r)
        x2 = min(w_img, cx+r); y2 = min(h_img, cy+r)
        crops.append((i+1, img[y1:y2, x1:x2].copy()))

    # Show each cropped lens
    for idx, crop in crops:
        win = f"Lens {idx} crop"
        cv2.imshow(win, fit(crop, 400, 400))

# ── Display ───────────────────────────────────────────────────
cv2.imshow("1 - Original",          fit(img))
cv2.imshow("2 - Blue module mask",  fit(module_vis))
cv2.imshow("3 - Lens detection",    fit(result))

cv2.imwrite("out_module_mask.png", module_vis)
cv2.imwrite("out_lens_detect.png", result)
print("\nSaved: out_module_mask.png  |  out_lens_detect.png")
print("Press any key to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
