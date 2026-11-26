"""
Extract company, date, address and total from OCR text with simple rules,
and score them against SROIE's entities/ answers.

Usage:
    python extract_fields.py                    # auto-mode OCR text (raw, clean fallback)
    python extract_fields.py --mode raw
    python extract_fields.py --show-misses 5    # print some wrong answers per field

Needs the OCR text cached by eval_ocr.py first (results/<split>/psm<N>/...).
"""
import argparse
import csv
import json
import re
from pathlib import Path

import jiwer

from eval_ocr import DATA, read_ocr

FIELDS = ["company", "date", "address", "total"]


# ---------- date ----------

MONTHS = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
DATE_PATTERNS = [
    # 15/01/2019, 27-06-2018, 08/03/18, 09.02.2018
    re.compile(r"\b(\d{1,2})\s?[/.-]\s?(\d{1,2})\s?[/.-]\s?(\d{4}|\d{2})\b"),
    # 2018-06-27
    re.compile(r"\b(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})\b"),
    # 12 MAR 2018, 12-Mar-18
    re.compile(rf"\b(\d{{1,2}})[\s-]?({MONTHS})[A-Z]*[\s,-]?(\d{{4}}|\d{{2}})\b", re.IGNORECASE),
]


def is_valid_date(kind, m):
    """Reject things like 45/67/89 that only look like dates."""
    if kind == 0:
        a, b = int(m.group(1)), int(m.group(2))
        return 1 <= a <= 31 and 1 <= b <= 31 and min(a, b) <= 12
    if kind == 1:
        return 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31
    return 1 <= int(m.group(1)) <= 31


def find_date(lines):
    """First valid date on the receipt, top to bottom."""
    for line in lines:
        for kind, pattern in enumerate(DATE_PATTERNS):
            for m in pattern.finditer(line):
                if is_valid_date(kind, m):
                    text = m.group(0)
                    return re.sub(r"\s", "", text) if kind < 2 else text.upper()
    return ""


# ---------- total ----------

AMOUNT = re.compile(r"\d{1,3}(?:,\d{3})+\.\d{2}(?!\d)|\d+[.,]\d{2}(?!\d)")
TOTAL_WORD = re.compile(r"T[O0]TA[L1I]", re.IGNORECASE)          # also catches OCR slips like T0TAL
NOT_TOTAL = re.compile(r"SUB\s*-?\s*T[O0]TA|QTY|QUANTITY", re.IGNORECASE)


def to_number(text):
    """'1,234.50' -> 1234.5, and '0,20' (OCR read the dot as a comma) -> 0.2"""
    if re.fullmatch(r"\d+,\d{2}", text):
        text = text.replace(",", ".")
    return float(text.replace(",", ""))


def find_total(lines):
    """Largest amount on a TOTAL line (not sub-total / quantity)."""
    candidates = []
    for i, line in enumerate(lines):
        if TOTAL_WORD.search(line) and not NOT_TOTAL.search(line):
            amounts = AMOUNT.findall(line)
            if not amounts and i + 1 < len(lines):   # amount wrapped onto the next line
                amounts = AMOUNT.findall(lines[i + 1])
            if amounts:
                candidates.append(to_number(amounts[-1]))
    if not candidates:                                # fallback: biggest amount anywhere
        candidates = [to_number(a) for line in lines for a in AMOUNT.findall(line)]
    return f"{max(candidates):.2f}" if candidates else ""


# ---------- company ----------

COMPANY_HINT = re.compile(
    r"SDN\.?\s*BHD|S/B|\bBHD\b|\bBERHAD\b|ENTERPRISE|TRADING|RESTORAN|RESTAURANT|"
    r"\bMART\b|\bSTORE\b|HARDWARE|\bCAFE\b|BAKERY|PHARMACY|\bLTD\b|\bPLT\b",
    re.IGNORECASE)
SKIP_LINE = re.compile(r"INVOICE|RECEIPT|\bCOPY\b|WELCOME|\bGST\b|\bTEL\b|\bFAX\b", re.IGNORECASE)
REG_NO_TAIL = re.compile(r"\s*\(?\s*\d{4,}\s*-+\s*[A-Z]\s*\)?.*$")   # "(640899-P)" and anything after


def clean_company(line):
    return REG_NO_TAIL.sub("", line).strip(" .,:;-")


def looks_like_name(text):
    chars = text.replace(" ", "")
    letters = sum(c.isalpha() for c in chars)
    return letters >= 4 and letters / len(chars) >= 0.6


def find_company(lines, top=8):
    """Returns (company, line index). Looks only at the top of the receipt."""
    head = lines[:top]
    # 1) a line with a company-type word (SDN BHD, RESTORAN, ...)
    for i, line in enumerate(head):
        if COMPANY_HINT.search(line):
            # name split over two lines: "MR. D.I.Y. (KUCHAI)" / "SDN BHD"
            if i > 0 and len(line.split()) <= 2 and looks_like_name(head[i - 1]):
                return clean_company(head[i - 1] + " " + line), i
            return clean_company(line), i
    # 2) otherwise the first line that looks like a name
    for i, line in enumerate(head):
        name = clean_company(line)
        if looks_like_name(name) and not SKIP_LINE.search(line):
            return name, i
    return "", -1


# ---------- address ----------

ADDRESS_STOP = re.compile(
    r"\bTEL\b|\bT\s*:|\bFAX\b|\bF\s*:|PHONE|H/P|E-?MAIL|WWW|HTTP|\bGST\b|INVOICE|RECEIPT|"
    r"\bDATE\b|CASHIER", re.IGNORECASE)
REG_LINE = re.compile(r"\d{4,}\s*-+\s*[A-Z]\b|\bROC\b|CO\.?\s*-?\s*REG|COMPANY\s*(REG|NO)", re.IGNORECASE)


def find_address(lines, company_idx, max_lines=5):
    """Lines under the company name, until a phone/GST/date line."""
    parts = []
    for line in lines[company_idx + 1:]:
        if ADDRESS_STOP.search(line) or find_date([line]):
            break
        if REG_LINE.search(line):
            if parts:
                break
            continue                     # registration number right under the name
        parts.append(line)
        if len(parts) == max_lines:
            break
    return " ".join(parts)


# ---------- extract + score ----------

def extract(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    company, idx = find_company(lines)
    return {
        "company": company,
        "date": find_date(lines),
        "address": find_address(lines, idx) if idx >= 0 else "",
        "total": find_total(lines),
    }


def norm(text):
    return re.sub(r"\s+", " ", text.upper()).strip(" .,;:")


def same_total(pred, truth):
    nums = AMOUNT.findall(truth) or re.findall(r"\d+", truth)
    return bool(pred and nums) and abs(to_number(nums[-1]) - float(pred)) < 0.005


def compare(field, pred, truth):
    """Returns (exact match?, similarity 0..1)."""
    p, t = norm(pred), norm(truth)
    exact = same_total(pred, truth) if field == "total" else p == t
    similarity = max(0.0, 1 - jiwer.cer(t, p)) if p else 0.0
    return exact, similarity


def load_entities(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser(description="Rule-based field extraction on SROIE")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--psm", type=int, default=4)
    ap.add_argument("--mode", default="auto", choices=["raw", "clean", "auto"])
    ap.add_argument("--min-words", type=int, default=20)
    ap.add_argument("--show-misses", type=int, default=3)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    run_dir = Path(args.out) / args.split / f"psm{args.psm}"
    rows, skipped = [], 0
    for ent_file in sorted((DATA / args.split / "entities").glob("*.txt")):
        truth = load_entities(ent_file)
        try:
            text = read_ocr(run_dir, args.mode, ent_file.stem, args.min_words)
        except FileNotFoundError:
            text = None
        if truth is None or text is None:
            skipped += 1
            continue
        pred = extract(text)
        for field in FIELDS:
            if not truth.get(field, "").strip():
                continue
            exact, sim = compare(field, pred[field], truth[field])
            rows.append({"id": ent_file.stem, "field": field, "truth": truth[field],
                         "pred": pred[field], "exact": int(exact), "similarity": round(sim, 3)})

    csv_path = run_dir / f"fields_{args.mode}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    receipts = len({r["id"] for r in rows})
    print(f"split={args.split}  psm={args.psm}  mode={args.mode}  receipts={receipts}  skipped={skipped}")
    print(f"{'field':<9} {'exact':>7} {'similar':>8}")
    for field in FIELDS:
        r = [x for x in rows if x["field"] == field]
        exact = sum(x["exact"] for x in r) / len(r)
        sim = sum(x["similarity"] for x in r) / len(r)
        print(f"{field:<9} {exact:7.1%} {sim:8.1%}")

    if args.show_misses:
        for field in FIELDS:
            misses = [x for x in rows if x["field"] == field and not x["exact"]][: args.show_misses]
            print(f"\n--- {field} misses ---")
            for x in misses:
                print(f"{x['id']}\n  truth: {x['truth']}\n  pred : {x['pred']}")
    print(f"\nall predictions: {csv_path}")


if __name__ == "__main__":
    main()