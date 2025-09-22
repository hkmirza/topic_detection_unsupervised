#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsupervised Topic Detection for Dialogue Corpora
Pipeline: Preprocess → TF-IDF → k-means with elbow selection → seeded LDA refinement
Outputs: top-N keywords per topic (Φ), document-topic distributions (Θ), and evaluation if labels exist.

This script implements the unsupervised variant of topic detectiion:
 - Term similarity with TF–IDF and L2-normalisation
 - k-means clustering; K* chosen by an elbow (curvature) heuristic
 - Seeds = top-M TF–IDF terms per cluster; used to bias LDA's eta prior
 - LDA used as an unsupervised PLDA-like refinement to obtain Θ (theta) and Φ (phi)
 - Evaluation (accuracy / precision / recall / F1) via Hungarian matching when 'label' is present

References notation:
 - U: list of utterances
 - T: number of topics (set to selected K*)
 - Θ (Theta): document–topic distribution
 - Φ (Phi): topic–word distribution
"""

import argparse
import os
import re
import json
import math
import string
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from scipy.optimize import linear_sum_assignment

import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

import nltk
from nltk.corpus import stopwords

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------
# Text preprocessing
# -----------------------

_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_NON_ALPHA_RE = re.compile(r'[^a-zA-Z\s]')
_MULTI_SPACE_RE = re.compile(r'\s+')

def ensure_nltk():
    try:
        _ = stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def simple_preprocess(text: str) -> str:
    """Lowercase, strip URLs/punct, collapse spaces, remove stopwords."""
    text = text or ""
    text = text.lower()
    text = _URL_RE.sub(' ', text)
    text = _NON_ALPHA_RE.sub(' ', text)
    text = _MULTI_SPACE_RE.sub(' ', text).strip()
    toks = [t for t in text.split() if t not in STOP]
    return ' '.join(toks)

# -----------------------
# Elbow (curvature) heuristic
# -----------------------

def choose_k_by_elbow(k_vals, inertias):
    """
    Select K* at the 'elbow' using a simple discrete curvature heuristic:
    find the point maximising the perpendicular distance to the line between endpoints.
    """
    x = np.array(k_vals, dtype=float)
    y = np.array(inertias, dtype=float)
    # Line from first to last
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    # Distances from each point to the line
    def point_line_dist(p, a, b):
        return np.abs(np.cross(b - a, p - a) / np.linalg.norm(b - a))
    dists = np.array([point_line_dist(np.array([xi, yi]), p1, p2) for xi, yi in zip(x, y)])
    idx = int(np.argmax(dists))
    return int(x[idx]), idx, dists

# -----------------------
# Topic keyword extraction from k-means centroids
# -----------------------

def top_terms_from_centroids(kmeans, feature_names, top_m):
    """Return list of top-M terms for each cluster centroid."""
    centroids = kmeans.cluster_centers_
    seeds = []
    for k in range(centroids.shape[0]):
        idx = np.argsort(centroids[k])[::-1][:top_m]
        seeds.append([feature_names[i] for i in idx])
    return seeds  # list of lists

# -----------------------
# Build LDA with eta seeding
# -----------------------

def build_seeded_eta(num_topics, dictionary, seeds, base_eta=1.0, boost=15.0):
    """
    Construct an eta prior matrix (num_topics x vocab_size).
    Seed words for topic k receive a boosted prior to bias Φ toward cluster keywords.
    """
    V = len(dictionary)
    eta = np.full((num_topics, V), base_eta, dtype=float)
    word2id = dictionary.token2id
    for t, seed_words in enumerate(seeds):
        for w in seed_words:
            if w in word2id:
                eta[t, word2id[w]] = base_eta + boost
    return eta

def dense_doc_topic(model, bow, T):
    """Convert gensim's sparse topic distribution to a dense Θ row."""
    dist = model.get_document_topics(bow, minimum_probability=0.0)
    theta = np.zeros(T, dtype=float)
    for t, p in dist:
        theta[t] = p
    return theta

# -----------------------
# Evaluation with Hungarian alignment
# -----------------------

def evaluate_with_hungarian(y_true, y_pred_topics, T):
    """
    Align discovered topics to gold labels by Hungarian matching on the confusion matrix,
    then compute accuracy / precision / recall / F1 (macro & weighted).
    """
    # Map labels to contiguous ids
    labels = sorted(pd.unique(y_true))
    lab2id = {lab: i for i, lab in enumerate(labels)}
    y_true_ids = np.array([lab2id[l] for l in y_true], dtype=int)

    # Build confusion matrix: rows=gold labels, cols=topics
    C = confusion_matrix(y_true_ids, y_pred_topics, labels=range(len(labels)))
    # If topics > labels or vice-versa, pad to square for Hungarian
    K = max(C.shape[0], C.shape[1])
    M = np.zeros((K, K), dtype=int)
    M[:C.shape[0], :C.shape[1]] = C
    # Hungarian: maximise matches => minimise negative counts
    cost = M.max() - M
    r_ind, c_ind = linear_sum_assignment(cost)

    # Topic->label mapping
    topic2label = {}
    for r, c in zip(r_ind, c_ind):
        if r < C.shape[0] and c < C.shape[1]:
            topic2label[c] = r

    # Map predicted topics to labels (unmapped topics fallback to most common label)
    default_label_id = np.argmax(C.sum(axis=1))
    mapped = np.array([topic2label.get(t, default_label_id) for t in y_pred_topics], dtype=int)

    acc = accuracy_score(y_true_ids, mapped)
    pr_macro, rc_macro, f1_macro, _ = precision_recall_fscore_support(y_true_ids, mapped, average='macro', zero_division=0)
    pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(y_true_ids, mapped, average='weighted', zero_division=0)

    report = classification_report(y_true_ids, mapped, zero_division=0, target_names=[str(l) for l in labels])
    out = {
        "accuracy": acc,
        "precision_macro": pr_macro, "recall_macro": rc_macro, "f1_macro": f1_macro,
        "precision_weighted": pr_w, "recall_weighted": rc_w, "f1_weighted": f1_w,
        "topic_to_label_mapping": {int(t): labels[lid] for t, lid in topic2label.items()},
        "classification_report": report
    }
    return out

# -----------------------
# Main experiment
# -----------------------

def run_experiment(args):
    # Load data
    df = pd.read_csv(args.csv)
    assert args.text_col in df.columns, f"Missing text column '{args.text_col}'"
    if args.label_col and args.label_col not in df.columns:
        print(f"[WARN] Label column '{args.label_col}' not found. Running without supervised evaluation.")
        args.label_col = None

    # Preprocess utterances U
    global STOP
    ensure_nltk()
    STOP = set(stopwords.words('english'))
    U_raw = df[args.text_col].astype(str).tolist()
    U = [simple_preprocess(u) for u in U_raw]
    df["_clean"] = U

    # TF–IDF representation (term similarity stage)
    vec = TfidfVectorizer(ngram_range=(1, args.ngram_max),
                          min_df=args.min_df, max_df=args.max_df,
                          norm='l2', lowercase=False, analyzer='word',
                          token_pattern=r'(?u)\b\w+\b')
    X = vec.fit_transform(U)               # shape: N x |V|
    X = normalize(X, norm='l2', axis=1)    # ensure unit-norm rows
    vocab = np.array(vec.get_feature_names_out())

    # Elbow selection over k-means
    k_values = list(range(args.k_min, args.k_max + 1))
    inertias = []
    for K in k_values:
        km = KMeans(n_clusters=K, init='k-means++', n_init=args.k_restarts, tol=1e-4, random_state=args.seed)
        km.fit(X)
        inertias.append(km.inertia_)
    K_star, elbow_idx, dists = choose_k_by_elbow(k_values, inertias)
    print(f"[INFO] Selected K* (number of clusters / topics) = {K_star}")

    # Final k-means fit
    kmeans = KMeans(n_clusters=K_star, init='k-means++', n_init=args.k_restarts, tol=1e-4, random_state=args.seed)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_

    # Seed lexicons: top-M terms per centroid
    seeds = top_terms_from_centroids(kmeans, vocab, args.seed_terms)
    for k, words in enumerate(seeds):
        print(f"[SEED k={k}]: {', '.join(words)}")

    # Build corpus for LDA (unsupervised PLDA refinement)
    # Use the same tokenisation (space-split from cleaned text)
    token_lists = [u.split() for u in U]
    dictionary = Dictionary(token_lists)
    # Keep only words present in TF–IDF vocabulary to stay consistent
    keep_ids = [dictionary.token2id[w] for w in list(dictionary.token2id.keys()) if w in vec.vocabulary_]
    dictionary.filter_tokens(bad_ids=[i for i in dictionary.token2id.values() if i not in keep_ids])
    dictionary.compactify()
    corpus = [dictionary.doc2bow(toks) for toks in token_lists]

    # Seeded eta prior (Φ prior)
    T_topics = K_star  # T = K*
    eta = build_seeded_eta(T_topics, dictionary, seeds, base_eta=args.base_eta, boost=args.eta_boost)

    # Alpha prior (Θ prior)
    alpha = [args.alpha] * T_topics

    # Train LDA
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=T_topics,
        iterations=args.lda_iterations,
        passes=1,
        random_state=args.seed,
        alpha=alpha,
        eta=eta,
        minimum_probability=0.0
    )

    # Topic-word distributions Φ and top keywords per topic
    Phi = lda.get_topics()  # shape T x |V|
    top_rows = []
    for t in range(T_topics):
        top_ids = np.argsort(Phi[t])[::-1][:args.top_n]
        words = [dictionary[id_] for id_ in top_ids]
        top_rows.append(words)

    # Build a table (columns = Topic0..TopicT-1, rows = keywords)
    max_len = max(len(row) for row in top_rows)
    topic_table = pd.DataFrame(
        {f"Topic{t}": (top_rows[t] + [""] * (max_len - len(top_rows[t]))) for t in range(T_topics)}
    )
    os.makedirs(args.out_dir, exist_ok=True)
    topics_path = os.path.join(args.out_dir, "topics_table.csv")
    topic_table.to_csv(topics_path, index=False)
    print(f"[SAVE] Topic keyword table → {topics_path}")

    # Document-topic distributions Θ and predicted topic (argmax)
    Theta = np.vstack([dense_doc_topic(lda, bow, T_topics) for bow in corpus])
    y_pred_topics = Theta.argmax(axis=1)

    # If labels available, evaluate with Hungarian mapping
    eval_result = None
    if args.label_col:
        y_true = df[args.label_col].astype(str).fillna("NA").tolist()
        eval_result = evaluate_with_hungarian(y_true, y_pred_topics, T_topics)
        eval_path = os.path.join(args.out_dir, "evaluation.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2)
        print(f"[SAVE] Evaluation metrics → {eval_path}")
        print("\n=== Classification Report (after Hungarian alignment) ===")
        print(eval_result["classification_report"])

    # Save per-utterance assignments (cluster, topic, and Theta distribution)
    assign_path = os.path.join(args.out_dir, "utterance_assignments.csv")
    out_df = pd.DataFrame({
        "utterance": df[args.text_col],
        "cluster": cluster_labels,
        "topic_argmax": y_pred_topics
    })
    if args.label_col:
        out_df["gold_label"] = df[args.label_col]
    # Append Θ as columns
    for t in range(T_topics):
        out_df[f"theta_{t}"] = Theta[:, t]
    out_df.to_csv(assign_path, index=False)
    print(f"[SAVE] Utterance-level assignments (Θ) → {assign_path}")

    # Also store seeds used and inertias for reproducibility
    meta = {
        "K_candidates": k_values,
        "inertias": inertias,
        "selected_K": K_star,
        "elbow_index": int(elbow_idx),
        "seed_terms_per_cluster": {int(k): seeds[k] for k in range(len(seeds))},
        "top_keywords_per_topic": {f"Topic{t}": top_rows[t] for t in range(T_topics)},
        "alpha": args.alpha,
        "base_eta": args.base_eta,
        "eta_boost": args.eta_boost,
        "lda_iterations": args.lda_iterations,
        "vectorizer": {
            "min_df": args.min_df, "max_df": args.max_df, "ngram_max": args.ngram_max
        }
    }
    meta_path = os.path.join(args.out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVE] Metadata (inertias, seeds, settings) → {meta_path}")

    print("\n[DONE] Unsupervised topic detection complete.")
    if not args.label_col:
        print("No gold labels were provided; evaluation metrics were skipped.")

# -----------------------
# CLI
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised Topic Detection (TF–IDF → k-means (elbow) → seeded LDA)")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--text-col", type=str, default="utterance", help="Text column name")
    parser.add_argument("--label-col", type=str, default="label", help="Optional label column for evaluation")

    parser.add_argument("--min-df", type=int, default=5, help="min_df for TF–IDF")
    parser.add_argument("--max-df", type=float, default=0.95, help="max_df for TF–IDF")
    parser.add_argument("--ngram-max", type=int, default=2, help="Use unigrams..ngram-max in TF–IDF")

    parser.add_argument("--k-min", type=int, default=5, help="Minimum K for elbow search")
    parser.add_argument("--k-max", type=int, default=60, help="Maximum K for elbow search")
    parser.add_argument("--k-restarts", type=int, default=20, help="k-means n_init restarts")
    parser.add_argument("--seed-terms", type=int, default=10, help="Top-M centroid terms used as seeds")

    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (Θ prior)")
    parser.add_argument("--base-eta", type=float, default=1.0, help="Base eta for Φ prior")
    parser.add_argument("--eta-boost", type=float, default=15.0, help="Add this value to seed words in eta")
    parser.add_argument("--lda-iterations", type=int, default=1000, help="LDA iterations")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default="outputs_unsupervised", help="Directory to save outputs")

    args = parser.parse_args()
    run_experiment(args)

