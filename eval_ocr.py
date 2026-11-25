"""
Score Tesseract OCR on SROIE: raw images vs. ocr.py's clean-up.

Usage:
    python eval_ocr.py --limit 5        # quick check on 5 images
    python eval_ocr.py                  # all 347 test images
    python eval_ocr.py --psm 6          # try another page mode
    python eval_ocr.py --modes raw      # only the raw baseline

Metrics (each image's OCR text vs. the text in box/):
    CER      character error rate  (lower is better)
    WER      word error rate       (lower is better)
    word F1  bag-of-words match, ignores reading order (higher is better)

OCR text is cached in results/<split>/psm<N>/<mode>/<id>.txt, so if you stop
and re-run, it only OCRs images that aren't done yet.
"""
import argparse
import csv
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Each Tesseract call uses 1 CPU thread; we run several images at once instead.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import cv2
import jiwer
import pytesseract

from ocr import preprocess, to_gray

DATA = Path("SROIE2019")
WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if shutil.which("tesseract") is None and Path(WINDOWS_TESSERACT).exists():
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT


# ---------- ground truth ----------

def load_truth(box_file):
    """Each line is x1,y1,x2,y2,x3,y3,x4,y4,TEXT, and TEXT may contain commas."""
    lines = []
    for row in box_file.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        parts = row.split(",", 8)          # split only on the first 8 commas
        if len(parts) == 9 and parts[8].strip():
            lines.append(parts[8].strip())
    return "\n".join(lines)


def normalize(text):
    """Uppercase, and turn all spaces/newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip().upper()


# ---------- OCR (runs in worker processes) ----------

def ocr_image(img_path, mode, psm, out_file):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read {img_path}")
    if mode == "raw":
        proc = to_gray(img)
    else:
        proc, _angle = preprocess(img)
    text = pytesseract.image_to_string(proc, lang="eng", config=f"--oem 1 --psm {psm}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(text, encoding="utf-8")
    return out_file


# ---------- scoring ----------

def matched_words(ref_words, hyp_words):
    """Words found in both texts, ignoring order (duplicates counted fairly)."""
    return sum((Counter(ref_words) & Counter(hyp_words)).values())


def score(ref, hyp):
    ref_words, hyp_words = ref.split(), hyp.split()
    if hyp:
        cer, wer = jiwer.cer(ref, hyp), jiwer.wer(ref, hyp)
    else:                                  # Tesseract found nothing at all
        cer = wer = 1.0
    return {
        "ref_chars": len(ref),
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "matched_words": matched_words(ref_words, hyp_words),
        "cer": round(cer, 4),
        "wer": round(wer, 4),
    }

def read_ocr(run_dir, mode, stem, min_words):
    """'auto' = use raw OCR, unless it found fewer than min_words words; then use clean."""
    if mode == "auto":
        raw = (run_dir / "raw" / f"{stem}.txt").read_text(encoding="utf-8")
        if len(raw.split()) >= min_words:
            return raw
        return (run_dir / "clean" / f"{stem}.txt").read_text(encoding="utf-8")
    return (run_dir / mode / f"{stem}.txt").read_text(encoding="utf-8")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Evaluate Tesseract on SROIE")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    # ap.add_argument("--modes", default="raw,clean", help="comma-separated: raw, clean")
    ap.add_argument("--modes", default="raw,clean,auto", help="comma-separated: raw, clean, auto")
    ap.add_argument("--min-words", type=int, default=20, help="auto mode: fall back to clean if raw finds fewer words than this")
    ap.add_argument("--psm", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    images = sorted((DATA / args.split / "img").glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    run_dir = Path(args.out) / args.split / f"psm{args.psm}"

    # 1) OCR every image that isn't cached yet
    # "auto" reuses the raw and clean results, so only those two need OCR
    ocr_modes = {m for m in modes if m != "auto"}
    if "auto" in modes:
        ocr_modes |= {"raw", "clean"}
    jobs = [(img, mode, run_dir / mode / f"{img.stem}.txt")
            for mode in sorted(ocr_modes) for img in images]
    jobs = [j for j in jobs if not j[2].exists()]
    print(f"{len(images)} images x {len(ocr_modes)} OCR modes, {len(jobs)} to OCR "
          f"({args.workers} at a time)")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(ocr_image, img, mode, args.psm, out) for img, mode, out in jobs]
        for done, fut in enumerate(as_completed(futures), 1):
            fut.result()                   # shows any error from the worker
            if done % 25 == 0 or done == len(jobs):
                print(f"  OCR {done}/{len(jobs)}")

    # 2) Score each image
    rows = []
    for mode in modes:
        for img in images:
            ref = normalize(load_truth(DATA / args.split / "box" / f"{img.stem}.txt"))
            # hyp = normalize((run_dir / mode / f"{img.stem}.txt").read_text(encoding="utf-8"))
            hyp = normalize(read_ocr(run_dir, mode, img.stem, args.min_words))
            rows.append({"id": img.stem, "mode": mode, **score(ref, hyp)})

    csv_path = run_dir / "scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 3) Summary over all images (weighted by text length, so long receipts count more)
    print(f"\nsplit={args.split}  psm={args.psm}  images={len(images)}")
    print(f"{'mode':<6} {'CER':>7} {'WER':>7} {'word P':>7} {'word R':>7} {'word F1':>8}")
    for mode in modes:
        r = [row for row in rows if row["mode"] == mode]
        chars = sum(x["ref_chars"] for x in r)
        words = sum(x["ref_words"] for x in r)
        cer = sum(x["cer"] * x["ref_chars"] for x in r) / chars
        wer = sum(x["wer"] * x["ref_words"] for x in r) / words
        matched = sum(x["matched_words"] for x in r)
        found = sum(x["hyp_words"] for x in r)
        precision = matched / found if found else 0.0
        recall = matched / words
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{mode:<6} {cer:7.1%} {wer:7.1%} {precision:7.1%} {recall:7.1%} {f1:8.1%}")
    print(f"\nper-image scores: {csv_path}")


if __name__ == "__main__":
    main()