"""
Step 1: Detect outer camera rings only.
Tune the 5 parameters in the TUNING BLOCK until both rings are found correctly.
Nothing else happens here — no dust, no inner circles, no island detection.
"""

import cv2
import numpy as np
import sys

# ══════════════════════════════════════════════════════
#  TUNING BLOCK  ← only change values here
# ══════════════════════════════════════════════════════
MIN_RADIUS  = 50    # smallest ring radius in pixels
MAX_RADIUS  = 300   # largest  ring radius in pixels
MIN_DIST    = 80    # minimum distance between two ring centres (px)
PARAM1      = 60    # Canny edge threshold inside HoughCircles
PARAM2      = 30    # accumulator threshold — LOWER = finds more circles
                    #   too low  → false rings appear
                    #   too high → real rings get missed
                    #   start at 30, step by ±5 until both lenses show up
# ══════════════════════════════════════════════════════

IMAGE = "img34.png"

img = cv2.imread(IMAGE)
if img is None:
    print(f"Cannot load {IMAGE}")
    sys.exit(1)

gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

circles = cv2.HoughCircles(
    smooth,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=MIN_DIST,
    param1=PARAM1,
    param2=PARAM2,
    minRadius=MIN_RADIUS,
    maxRadius=MAX_RADIUS,
)

result = img.copy()

if circles is None:
    print("No circles found. Lower PARAM2 or adjust MIN/MAX_RADIUS.")
else:
    circles = np.round(circles[0]).astype(int)
    print(f"Found {len(circles)} circle(s):")
    for i, (cx, cy, r) in enumerate(circles):
        print(f"  Circle {i+1}: centre=({cx},{cy})  radius={r}px")
        cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)   # ring
        cv2.circle(result, (cx, cy), 4, (0, 0, 255), -1)  # centre dot
        cv2.putText(result, f"r={r}", (cx - r, cy - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

# Resize for display if image is large
h, w   = result.shape[:2]
scale  = min(960 / w, 700 / h, 1.0)
disp   = (int(w * scale), int(h * scale))

cv2.imshow("Original",     cv2.resize(img,    disp))
cv2.imshow("Outer Rings",  cv2.resize(result, disp))
cv2.waitKey(0)
cv2.destroyAllWindows()
