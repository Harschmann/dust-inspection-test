import cv2
import sys

# ══════════════════════════════════════════════════════════════
#  CROP PARAMETERS  ← tweak these
# ══════════════════════════════════════════════════════════════

X = 100   # left edge of the rectangle (pixels from left of image)
Y = 100   # top  edge of the rectangle (pixels from top  of image)
W = 400   # width  of the rectangle
H = 300   # height of the rectangle

# ══════════════════════════════════════════════════════════════

img = cv2.imread("img34.png")
if img is None:
    print("Cannot load img34.png")
    sys.exit(1)

print(f"Image size: {img.shape[1]} x {img.shape[0]}  (width x height)")
print(f"Crop box  : X={X}  Y={Y}  W={W}  H={H}")

# Draw rectangle on a copy of the original
annotated = img.copy()
cv2.rectangle(annotated, (X, Y), (X + W, Y + H), (0, 255, 0), 2)

# Crop
crop = img[Y : Y + H, X : X + W]

# Resize for display (keeps aspect ratio, max 900px wide)
def fit(im, max_w=900, max_h=700):
    h, w = im.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1:
        return cv2.resize(im, (int(w * scale), int(h * scale)))
    return im

cv2.imshow("Original  (green box = crop area)", fit(annotated))
cv2.imshow("Cropped region", fit(crop))
cv2.waitKey(0)
cv2.destroyAllWindows()
