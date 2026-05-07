import os
import cv2
import math
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# ----------------------------
# Config
# ----------------------------
DICT = cv2.aruco.DICT_6X6_250

# How many markers to generate (you can also directly set marker_ids)
X = 6
marker_ids = list(range(X))  # e.g. [0,1,2,3,4,5] 
marker_ids = [10, 11, 12, 13, 14, 15]  # <-- or set explicit IDs

# PDF output
OUTPUT_PDF = "aruco_print_sizes_A4.pdf"
MARKER_DIR = "aruco_markers_png"

# Smallest page layout: all markers on one page
SMALL_GRID_MARKER_SIZE_CM = 6.0

# Intermediate single-marker page sizes (physical size on paper)
INTERMEDIATE_SIZES_CM = [8, 10, 12, 14, 16]

# Margins for single-marker pages
PAGE_MARGIN_CM = 1.0

# Generate marker images at high resolution so printing stays crisp.
# (The PDF will scale these images to requested physical sizes.)
MARKER_IMAGE_PX = 2400

# ----------------------------
# Marker generation
# ----------------------------
def generate_aruco_pngs(marker_ids, out_dir, marker_px=2400, aruco_dict_id=DICT):
    os.makedirs(out_dir, exist_ok=True)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)

    paths = {}
    for mid in marker_ids:
        img = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_px)
        path = os.path.join(out_dir, f"aruco_{mid}.png")
        cv2.imwrite(path, img)
        paths[mid] = path
    return paths

# ----------------------------
# PDF helpers
# ----------------------------
def draw_centered_image(c, img_path, center_x, center_y, w_pt, h_pt):
    img = ImageReader(img_path)
    x = center_x - w_pt / 2.0
    y = center_y - h_pt / 2.0
    c.drawImage(img, x, y, width=w_pt, height=h_pt, preserveAspectRatio=True, mask="auto")

def add_footer_label(c, text, page_w, page_h):
    c.setFont("Helvetica", 9)
    c.drawCentredString(page_w / 2.0, 0.75 * cm, text)

# ----------------------------
# Build PDF
# ----------------------------
def build_pdf(marker_png_paths, output_pdf):
    page_w, page_h = A4
    c = canvas.Canvas(output_pdf, pagesize=A4)

    # ---- Page 1: all markers on one page, ~6cm each ----
    marker_size_pt = SMALL_GRID_MARKER_SIZE_CM * cm

    n = len(marker_png_paths)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Compute grid area with some padding
    top_margin = 2.0 * cm
    bottom_margin = 2.0 * cm
    side_margin = 1.5 * cm
    gap = 0.8 * cm

    usable_w = page_w - 2 * side_margin
    usable_h = page_h - top_margin - bottom_margin

    # Adjust marker size down if needed to fit grid
    max_marker_w = (usable_w - (cols - 1) * gap) / cols
    max_marker_h = (usable_h - (rows - 1) * gap) / rows
    marker_size_pt = min(marker_size_pt, max_marker_w, max_marker_h)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(side_margin, page_h - 1.3 * cm, f"ArUco markers (grid) - {SMALL_GRID_MARKER_SIZE_CM:.1f} cm each")

    mids = list(marker_png_paths.keys())
    for idx, mid in enumerate(mids):
        r = idx // cols
        col = idx % cols

        x = side_margin + col * (marker_size_pt + gap) + marker_size_pt / 2.0
        y_top = page_h - top_margin
        y = y_top - r * (marker_size_pt + gap) - marker_size_pt / 2.0

        draw_centered_image(c, marker_png_paths[mid], x, y, marker_size_pt, marker_size_pt)

        # label under each marker
        c.setFont("Helvetica", 10)
        c.drawCentredString(x, y - marker_size_pt / 2.0 - 0.35 * cm, f"ID {mid}")

    add_footer_label(c, "Grid page: print at 100% scale (no 'fit to page')", page_w, page_h)
    c.showPage()

    # ---- Intermediate sizes: one marker per page, several sizes ----
    margin_pt = PAGE_MARGIN_CM * cm
    center_x = page_w / 2.0
    center_y = page_h / 2.0 + 0.5 * cm  # slight upward shift to make room for footer

    for size_cm in INTERMEDIATE_SIZES_CM:
        size_pt = size_cm * cm
        for mid, img_path in marker_png_paths.items():
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin_pt, page_h - 1.3 * cm, f"ArUco ID {mid} - {size_cm:.1f} cm")

            draw_centered_image(c, img_path, center_x, center_y, size_pt, size_pt)

            add_footer_label(c, f"ID {mid} @ {size_cm:.1f} cm - print at 100% scale", page_w, page_h)
            c.showPage()

    # ---- Largest size: one marker per page, as big as possible ----
    max_size_pt = min(page_w - 2 * margin_pt, page_h - 2 * margin_pt - 1.5 * cm)  # leave footer space

    for mid, img_path in marker_png_paths.items():
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin_pt, page_h - 1.3 * cm, f"ArUco ID {mid} - MAX on A4")

        draw_centered_image(c, img_path, center_x, center_y, max_size_pt, max_size_pt)

        max_size_cm = max_size_pt / cm
        add_footer_label(c, f"ID {mid} @ ~{max_size_cm:.1f} cm (max) - print at 100% scale", page_w, page_h)
        c.showPage()

    c.save()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    marker_png_paths = generate_aruco_pngs(marker_ids, MARKER_DIR, marker_px=MARKER_IMAGE_PX, aruco_dict_id=DICT)
    build_pdf(marker_png_paths, OUTPUT_PDF)
    print(f"Saved PDF: {OUTPUT_PDF}")
    print(f"Marker PNGs in: {MARKER_DIR}")

