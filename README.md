



# Unsupervised Topic Detection 

This repository contains a **reproducible** implementation of the *unsupervised topic detection* pipeline. It follows the exact stages used in the experiments:

1) preprocessing → 2) TF–IDF featureisation → 3) k-means with elbow/Knee selection → 4) topic keyword extraction → 5) evaluation (precision/recall/F1/accuracy against reference labels when available).

The main entry-point script is **`topic_detection_unsupervised.py`**.

---

## 1) Quick Start

```bash
# (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install experiment dependencies
pip install -r requirements_unsupervised.txt

# Run on a CSV dataset
python topic_detection_unsupervised.py \
  --data path/to/switchboard.csv \
  --text_col utterance \
  --label_col label \
  --dataset_name switchboard \
  --outdir outputs/switchboard
```

**Outputs** are written under `--outdir`:
- `topics_switchboard.csv` — final topic–keyword table (top-10 words per topic, with cluster sizes).
- `predictions_switchboard.csv` — utterance-level predicted topic ids and (optional) mapped reference labels.
- `metrics_switchboard.json` — evaluation metrics (accuracy, precision, recall, F1 macro/micro, homogeneity/completeness/NMI).
- `elbow_plot.png` — WCSS vs. K plot with the selected knee \(K^\*\).
- `confusion_matrix.png` — (if labels provided).

> If your dataset has **no labels**, omit `--label_col`. The script will still produce topics and cluster assignments, and will compute intrinsic clustering scores.

---

## 2) Dataset Format

Input is a **CSV** with at least one text column containing dialogue utterances.

| Column        | Required | Description                                                      |
|---------------|----------|------------------------------------------------------------------|
| `utterance`   | yes      | The dialogue utterance (free text). Use `--text_col` to rename. |
| `label`       | no       | Reference topic label (string/int). Use `--label_col` to rename.|
| `conv_id`     | no       | Conversation/session id (optional; not used for clustering).    |

**Example (first rows):**
```csv
utterance,label,conv_id
"Do you drive to work every day?",transport,sw_0001
"Mostly bus—parking is awful downtown.",transport,sw_0001
"...",...
```

---

## 3) How It Works (Aligned with Chapter 5)

- **Preprocessing**: case-folding, punctuation/URL removal, tokenisation, stopword removal, optional bigrams; lemmatisation optional (spaCy switch).
- **Vectorisation**: TF–IDF on \(n\)-grams (default `[1,2]`), vocabulary pruning (`min_df=5`, `max_df=0.95`), L2-normalisation.
- **Clustering / K selection**: k-means++ with multiple restarts; the **elbow (knee)** is selected using curvature (Kneedle).
- **Keyword extraction**: for each cluster, rank terms by centroid-aligned TF–IDF weight and keep top-10.
- **Evaluation** (if `--label_col` is provided): accuracy, precision, recall, F1 (macro/micro), homogeneity/completeness/NMI; Hungarian matching to align clusters to labels (optional switch).

These steps mirror the thesis methodology to ensure **reproducibility**.

---

## 4) Command-line Arguments

```bash
python topic_detection_unsupervised.py --help
```

Key options:

- `--data` (str, **required**): path to the CSV file.
- `--text_col` (str, default=`utterance`): text column name.
- `--label_col` (str, optional): reference label column name (if available).
- `--dataset_name` (str, default=`dataset`): short tag used in output filenames.
- `--outdir` (str, default=`outputs`): directory to write results.
- `--min_df` (int, default=5): min document frequency for vocabulary.
- `--max_df` (float, default=0.95): max document frequency threshold.
- `--ngram_min` / `--ngram_max` (int, default=1/2): \(n\)-gram range.
- `--kmin` / `--kmax` (int, default=5/60): K search range for elbow.
- `--max_iter` (int, default=300): k-means max iterations.
- `--n_init` (int, default=20): k-means restarts.
- `--seed` (int, default=42): random seed for full reproducibility.
- `--match_labels` (flag): use Hungarian matching to align clusters to labels for scoring.
- `--use_spacy_lemma` (flag): enable spaCy lemmatisation (slower; adds dependency), language `en_core_web_sm`.

---

## 5) Reproducing Thesis Tables

After running the script per dataset (Switchboard, PersonaChat, MultiWOZ), open
`topics_<dataset>.csv`. It lists **K\*** clusters with their **top-10 keywords** and sizes, so you can directly format the tables shown in Chapter 5. The `metrics_<dataset>.json` aggregates the scores presented alongside.

---

## 6) Notes on Intrinsic Coherence (Optional)

The thesis reports external metrics; if you need **topic coherence (NPMI)** intrinsics, you can extend the script by plugging in `octis` or `gensim`’s coherence module with a reference corpus. This is not enabled by default to keep dependencies minimal.

---

## 7) Troubleshooting

- **Memory errors / slow fit**: reduce `--kmax`, lower `--ngram_max` to `1`, increase `min_df`.
- **Few topics**: widen `--kmin/--kmax` or lower `min_df` to grow vocabulary.
- **Noisy keywords**: enable `--use_spacy_lemma`, raise `min_df`, or cap `max_df` (e.g., `0.9`).

---

## 8) Environment

- Python 3.10+ recommended.
- See `requirements_unsupervised.txt`. Install via:
  ```bash
  pip install -r requirements_unsupervised.txt
  ```

---

## 9) Citation

If you use this code, please cite the thesis and the following components: scikit-learn (TF–IDF, k-means), kneed (Kneedle), pandas, numpy, and spaCy (if enabled).
