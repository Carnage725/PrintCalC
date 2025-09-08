import argparse
import time
import fitz  # PyMuPDF
import numpy as np

def analyze_pdf_whole(pdf_path, dpi=300, white_thresh=245, chroma_thresh=0.05):
    """
    Analyze all pages of a PDF and return only:
      - area_white_pct: % of total page area that is white
      - area_color_pct: % of total page area that is colored (non-white & chromatic)
      - area_black_pct: % of total page area that is grayscale/black (non-white & achromatic)
    """
    doc = fitz.open(pdf_path)

    total_px = 0
    white_px = 0
    color_px = 0
    gray_px  = 0

    for page in doc:
        # Render page to RGB
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                              colorspace=fitz.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)

        h, w, _ = img.shape
        total_px += h * w

        # White mask in RGB space
        r = img[:, :, 0].astype(np.uint16)
        g = img[:, :, 1].astype(np.uint16)
        b = img[:, :, 2].astype(np.uint16)

        white_mask = (r >= white_thresh) & (g >= white_thresh) & (b >= white_thresh)
        wp = int(white_mask.sum())
        white_px += wp

        # If fully white, next page
        if wp == h * w:
            continue

        # Work on non-white pixels only
        nw = ~white_mask
        rf = (r[nw].astype(np.float32)) / 255.0
        gf = (g[nw].astype(np.float32)) / 255.0
        bf = (b[nw].astype(np.float32)) / 255.0

        # RGB -> CMYK (device conversion; ICC not applied)
        max_rgb = np.maximum.reduce([rf, gf, bf])
        k = 1.0 - max_rgb
        denom = np.maximum(1.0 - k, 1e-8)  # avoid div-by-zero
        c = (1.0 - rf - k) / denom
        m = (1.0 - gf - k) / denom
        y = (1.0 - bf - k) / denom

        # Color vs grayscale via CMY chroma
        max_cmy = np.maximum.reduce([c, m, y])
        min_cmy = np.minimum.reduce([c, m, y])
        chroma = max_cmy - min_cmy
        color_mask = chroma > chroma_thresh

        color_px += int(color_mask.sum())
        gray_px  += int((~color_mask).sum())

    # Final metrics over the whole document
    if total_px == 0:
        return (0.0, 0.0, 0.0)

    area_white_pct = (white_px * 100.0) / total_px
    area_color_pct = (color_px * 100.0) / total_px
    area_black_pct = (gray_px  * 100.0) / total_px

    return (area_white_pct, area_color_pct, area_black_pct)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure white, color, and black/grayscale area for a whole PDF.")
    ap.add_argument("pdf", help="Path to PDF")
    ap.add_argument("--dpi", type=int, default=300, help="Render DPI (lower for speed, higher for thin details)")
    ap.add_argument("--white", type=int, default=245, help="RGB white threshold 0–255 (raise if page background is off-white)")
    ap.add_argument("--chroma", type=float, default=0.05, help="CMY chroma threshold (0–1) to classify colored vs grayscale")
    args = ap.parse_args()

    t0 = time.perf_counter()
    area_white, area_color, area_black = analyze_pdf_whole(
        args.pdf, dpi=args.dpi, white_thresh=args.white, chroma_thresh=args.chroma
    )
    t1 = time.perf_counter()

    print("\n=== Document Summary (whole PDF) ===\n")
    print(f"  White (paper):         {area_white:.2f}%")
    print(f"  Color (non-white):     {area_color:.2f}%")
    print(f"  Black/Grayscale:       {area_black:.2f}%")
    print(f"\nProcessing time: {t1 - t0:.2f} seconds")
