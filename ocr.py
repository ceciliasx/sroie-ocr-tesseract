"""
Tesseract OCR starter pipeline.

Usage:
    python ocr.py path/to/image.jpg
    python ocr.py image.png --lang eng+ind --psm 6 --show-steps
    python ocr.py image.png --no-preprocess      # compare against raw OCR

Outputs (in ./output/<image-name>/):
    text.txt       - extracted text
    words.json     - each word with confidence + bounding box
    boxes.png      - image with word boxes drawn (green = confident, red = low)
    steps/*.png    - preprocessing stages (with --show-steps)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytesseract


# ---------- preprocessing ----------

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def upscale_if_small(gray, min_height=1000):
    """Tesseract likes text ~30px tall. Upscale small images."""
    h = gray.shape[0]
    if h < min_height:
        scale = min_height / h
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def denoise(gray):
    return cv2.fastNlMeansDenoising(gray, h=10)


def binarize(gray):
    """Adaptive threshold handles uneven lighting (phone photos, shadows)."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def _rotate(img, angle, border=255):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def deskew(binary, max_angle=10, step=0.5):
    """
    Projection-profile deskew: try angles, keep the one where text rows are
    sharpest (row sums of ink have the highest variance = lines aligned).
    More robust than minAreaRect, which is fooled by borders and noise.
    """
    ink = 255 - binary                              # text = high values
    small = cv2.resize(ink, None, fx=0.5, fy=0.5)   # faster search
    best_angle, best_score = 0.0, -1
    for a in np.arange(-max_angle, max_angle + step, step):
        score = np.var(_rotate(small, a, border=0).sum(axis=1))
        if score > best_score:
            best_angle, best_score = float(a), score
    if abs(best_angle) < step:
        return binary, 0.0
    return _rotate(binary, best_angle), best_angle


def preprocess(img, save_dir=None):
    steps = {}
    gray = to_gray(img);            steps["1_gray"] = gray
    gray = upscale_if_small(gray);  steps["2_upscaled"] = gray
    gray = denoise(gray);           steps["3_denoised"] = gray
    binary = binarize(gray);        steps["4_binary"] = binary
    final, angle = deskew(binary);  steps["5_deskewed"] = final
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, im in steps.items():
            cv2.imwrite(str(save_dir / f"{name}.png"), im)
    return final, angle


# ---------- OCR ----------

def run_ocr(img, lang="eng", psm=3, oem=1):
    """
    psm (page segmentation mode) matters a lot:
      3  = fully automatic (default, good for documents)
      4  = single column of variable-size text (receipts!)
      6  = single uniform block of text
      7  = single line
      11 = sparse text, find as much as possible (signs, screenshots)
    oem 1 = LSTM neural engine only
    """
    config = f"--oem {oem} --psm {psm}"
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    data = pytesseract.image_to_data(img, lang=lang, config=config,
                                     output_type=pytesseract.Output.DICT)
    words = []
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if word.strip() and conf >= 0:
            words.append({
                "text": word,
                "conf": round(conf, 1),
                "box": [data["left"][i], data["top"][i], data["width"][i], data["height"][i]],
                "line": f'{data["block_num"][i]}-{data["par_num"][i]}-{data["line_num"][i]}',
            })
    return text, words


def draw_boxes(img, words, threshold=70):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for w in words:
        x, y, bw, bh = w["box"]
        color = (0, 180, 0) if w["conf"] >= threshold else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
    return out


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Tesseract OCR pipeline")
    ap.add_argument("image")
    ap.add_argument("--lang", default="eng", help="e.g. eng, ind, eng+ind")
    ap.add_argument("--psm", type=int, default=3)
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--show-steps", action="store_true")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    path = Path(args.image)
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Could not read image: {path}")

    out_dir = Path(args.out) / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_preprocess:
        proc, angle = to_gray(img), 0.0
    else:
        proc, angle = preprocess(img, out_dir / "steps" if args.show_steps else None)

    text, words = run_ocr(proc, lang=args.lang, psm=args.psm)

    (out_dir / "text.txt").write_text(text, encoding="utf-8")
    (out_dir / "words.json").write_text(json.dumps(words, indent=2, ensure_ascii=False), encoding="utf-8")
    cv2.imwrite(str(out_dir / "boxes.png"), draw_boxes(proc, words))

    confs = [w["conf"] for w in words]
    avg = sum(confs) / len(confs) if confs else 0
    low = [w["text"] for w in words if w["conf"] < 70]
    print(f"--- {path.name} | lang={args.lang} psm={args.psm} "
          f"preprocess={not args.no_preprocess} deskew={angle:.2f}° ---")
    print(text.strip())
    print(f"\nwords: {len(words)} | avg confidence: {avg:.1f} | low-confidence: {low[:10]}")
    print(f"saved to {out_dir}/")


if __name__ == "__main__":
    main()
