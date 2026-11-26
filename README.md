# Receipt OCR with Tesseract on SROIE

Reads scanned receipts with **Tesseract OCR**, measures text accuracy, and extracts four fields (**company, date, address, total**) with rule-based parsing. Uses the [SROIE 2019](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2) dataset: 973 Malaysian receipts (626 train / 347 test).

The full analysis, with charts and examples, is in **[report.ipynb](report.ipynb)**.

## Pipeline

```
receipt image → (optional) clean-up → Tesseract OCR → rule-based extraction → company / date / address / total
```

1. **Image clean-up** (`ocr.py`): grayscale, upscale small images, denoise, adaptive black-and-white, deskew.
2. **OCR evaluation** (`eval_ocr.py`): runs Tesseract on every receipt in three modes and scores the text against the ground truth.
   - **raw**: grayscale only
   - **clean**: with the clean-up above
   - **auto**: raw, falling back to clean when raw finds fewer than 20 words
3. **Field extraction** (`extract_fields.py`): regex and layout rules for each field, scored against the answer key.
4. **Analysis** (`report.ipynb`): data exploration, OCR comparison, error analysis and an overfitting check.

## Results

**OCR quality** (test set, `--psm 4`)

| Mode | CER ↓ | Word F1 ↑ |
|---|---|---|
| raw | 26.2% | 70.5% |
| clean | 25.3% | 67.4% |
| **auto** | **24.4%** | **70.6%** |

**Field extraction** (exact match)

| Field | Test | Train (unseen) |
|---|---|---|
| date | 69.7% | 70.4% |
| total | 62.5% | 62.4% |
| company | 45.0% | 41.1% |
| address | 22.5% | 19.4% |

## Key findings

- **Clean-up isn't always better.** It helps on faded thermal receipts but adds noise on sharp scans (better on 30% of receipts, worse on 35%). Using it only as a fallback (auto) gives the best results.
- **Most wrong answers come from OCR, not the rules:** 61–82% of misses for company, date and address. Total is the exception, with 56% rule errors.
- **The answer key has noise.** About 5% of addresses don't match the receipt's own text, e.g. `B1750` where the receipt says `81750`.
- **There's a small amount of overfitting.** The rules were tuned on test misses. Date and total hold up on train, while company and address drop 3–4 points.

## Tools

| Tool | Used for |
|---|---|
| Python 3.12 | Everything |
| Tesseract 5.5 (LSTM engine) + `pytesseract` | OCR |
| OpenCV, NumPy, Pillow | Image clean-up and loading |
| `jiwer` | Character and word error rates |
| pandas, matplotlib, Jupyter | Analysis and charts |
| `concurrent.futures` | Running OCR on several images in parallel |

## Project structure

```
ocr.py              image clean-up + single-image OCR
eval_ocr.py         OCR on a whole split, CER/WER/word-F1 scores
extract_fields.py   rule-based field extraction + scoring
report.ipynb        analysis notebook
requirements.txt
SROIE2019/          dataset (not in repo)
results/            OCR text + scores (generated, not in repo)
```
