"""
STEP 2: Explore Data + Build BM25 Baseline
============================================
Before training any ML model, always:
1. Understand your data (EDA)
2. Build a simple baseline to beat

WHY BM25 AS BASELINE?
BM25 = "Best Match 25" — it's the gold standard keyword search algorithm.
Used by Elasticsearch, Solr, and older versions of Google.
Our LambdaMART model must BEAT BM25 to justify using ML.

WHAT IS NDCG@10?
NDCG = Normalized Discounted Cumulative Gain
- Measures ranking quality for top-10 results
- Range: 0.0 (worst) to 1.0 (perfect)
- "Discounted" = results ranked #1 matter more than #10
- Formula: DCG / IDCG (actual gain / ideal gain)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# NDCG IMPLEMENTATION (core metric — understand this well!)
# ============================================================

def dcg_at_k(relevances, k=10):
    """
    Discounted Cumulative Gain at K.
    
    Formula: sum of (2^rel - 1) / log2(rank + 1)
    
    The log2(rank+1) "discounts" lower-ranked results.
    Rank 1 has discount=1.0, rank 2=0.63, rank 10=0.29
    
    Args:
        relevances: list of relevance scores in ranked order
        k: only consider top-k results
    """
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    
    # gains: 2^rel - 1  (relevance 4 → gain=15, rel 0 → gain=0)
    gains = 2 ** relevances - 1
    
    # discounts: 1/log2(rank+1), ranks start at 1
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    
    return np.sum(gains / discounts)


def ndcg_at_k(relevances, k=10):
    """
    Normalized DCG at K.
    
    Divide actual DCG by the IDEAL DCG (best possible ranking).
    This normalizes to [0, 1] range across different queries.
    """
    actual_dcg = dcg_at_k(relevances, k)
    
    # Ideal = sort by relevance descending (perfect ranking)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg


def evaluate_ndcg(df, score_col, k=10):
    """
    Compute mean NDCG@K across all queries.
    
    For each query:
    1. Sort documents by score_col (descending)
    2. Get their relevance labels in that order
    3. Compute NDCG@K
    
    Return: mean NDCG across all queries
    """
    ndcg_scores = []
    
    for query_id, group in df.groupby('query_id'):
        # Sort documents by the ranking score
        ranked = group.sort_values(score_col, ascending=False)
        relevances = ranked['label'].tolist()
        
        ndcg = ndcg_at_k(relevances, k)
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 55)
print("STEP 2: Data Exploration + BM25 Baseline")
print("=" * 55)

df = pd.read_csv('data/search_ranking_data.csv')

print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Queries: {df['query_id'].nunique()}")
print(f"Docs per query: {df.groupby('query_id').size().mean():.0f}")

# ============================================================
# EDA: UNDERSTAND YOUR DATA
# ============================================================

print("\n--- Feature Statistics ---")
feature_cols = ['bm25_score','tfidf_score','pagerank','ctr',
                'title_match','url_depth','doc_length_norm',
                'freshness','domain_authority']
print(df[feature_cols].describe().round(3).to_string())

print("\n--- Label Distribution ---")
label_counts = df['label'].value_counts().sort_index()
for label, count in label_counts.items():
    bar = "█" * (count // 100)
    print(f"  Label {label}: {count:5d} docs  {bar}")

# ============================================================
# BM25 BASELINE EVALUATION
# ============================================================

print("\n--- BM25 Baseline ---")
print("Ranking documents by BM25 score only (no ML)...")

bm25_ndcg = evaluate_ndcg(df, score_col='bm25_score', k=10)
print(f"BM25 NDCG@10 = {bm25_ndcg:.4f}")

# Also compute random baseline (lower bound)
df['random_score'] = np.random.rand(len(df))
random_ndcg = evaluate_ndcg(df, score_col='random_score', k=10)
print(f"Random NDCG@10 = {random_ndcg:.4f}  (lower bound)")

print(f"\nOur ML model must beat {bm25_ndcg:.4f} to be useful!")

# ============================================================
# CORRELATION: which features are most related to relevance?
# ============================================================

print("\n--- Feature–Label Correlations ---")
correlations = df[feature_cols + ['label']].corr()['label'].drop('label').sort_values(ascending=False)
for feat, corr in correlations.items():
    bar_len = int(abs(corr) * 30)
    direction = "+" if corr > 0 else "-"
    print(f"  {feat:<20} {direction}{'█' * bar_len}  {corr:+.3f}")

# ============================================================
# VISUALIZATIONS
# ============================================================

fig = plt.figure(figsize=(14, 10))
fig.suptitle('Project 1: Search Ranking — Data Exploration', 
             fontsize=14, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1: Label distribution
ax1 = fig.add_subplot(gs[0, 0])
colors = ['#d32f2f','#f57c00','#fbc02d','#388e3c','#1976d2']
bars = ax1.bar(label_counts.index, label_counts.values, color=colors, edgecolor='white')
ax1.set_title('Relevance Label Distribution', fontweight='bold')
ax1.set_xlabel('Label (0=irrelevant → 4=perfect)')
ax1.set_ylabel('Count')
for bar, count in zip(bars, label_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             str(count), ha='center', va='bottom', fontsize=9)

# Plot 2: BM25 score distribution by label
ax2 = fig.add_subplot(gs[0, 1])
for label in range(5):
    subset = df[df['label'] == label]['bm25_score']
    ax2.hist(subset, bins=30, alpha=0.6, label=f'Label {label}', color=colors[label])
ax2.set_title('BM25 Score by Relevance Label', fontweight='bold')
ax2.set_xlabel('BM25 Score')
ax2.set_ylabel('Count')
ax2.legend(fontsize=8)

# Plot 3: Feature correlations
ax3 = fig.add_subplot(gs[0, 2])
corr_vals = correlations.values
corr_feats = [f.replace('_score','').replace('_norm','') for f in correlations.index]
colors_corr = ['#1976d2' if v > 0 else '#d32f2f' for v in corr_vals]
bars3 = ax3.barh(corr_feats, corr_vals, color=colors_corr)
ax3.set_title('Feature Correlation with Relevance', fontweight='bold')
ax3.set_xlabel('Pearson Correlation')
ax3.axvline(0, color='black', linewidth=0.8)

# Plot 4: NDCG comparison (baseline)
ax4 = fig.add_subplot(gs[1, 0])
methods = ['Random\nBaseline', 'BM25\nBaseline', 'ML Model\n(Target)']
ndcg_vals = [random_ndcg, bm25_ndcg, bm25_ndcg * 1.18]
colors4 = ['#d32f2f', '#f57c00', '#388e3c']
bars4 = ax4.bar(methods, ndcg_vals, color=colors4, width=0.5)
ax4.set_title('NDCG@10 Comparison', fontweight='bold')
ax4.set_ylabel('NDCG@10')
ax4.set_ylim(0, 1.0)
for bar, val in zip(bars4, ndcg_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
ax4.axhline(bm25_ndcg, color='orange', linestyle='--', alpha=0.7, label='BM25 target')

# Plot 5: PageRank distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(df['bm25_score'], df['pagerank'], 
            c=df['label'], cmap='RdYlGn', alpha=0.3, s=10)
ax5.set_title('BM25 vs PageRank\n(colored by relevance)', fontweight='bold')
ax5.set_xlabel('BM25 Score')
ax5.set_ylabel('PageRank')

# Plot 6: NDCG@K curve
ax6 = fig.add_subplot(gs[1, 2])
k_values = list(range(1, 21))
bm25_ndcg_k = [evaluate_ndcg(df, 'bm25_score', k) for k in k_values]
random_ndcg_k = [evaluate_ndcg(df, 'random_score', k) for k in k_values]
ax6.plot(k_values, bm25_ndcg_k, 'o-', color='#f57c00', label='BM25', linewidth=2)
ax6.plot(k_values, random_ndcg_k, 's--', color='#d32f2f', label='Random', linewidth=2)
ax6.set_title('NDCG@K Curve', fontweight='bold')
ax6.set_xlabel('K (top-K results)')
ax6.set_ylabel('NDCG@K')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.savefig('outputs/step2_eda_baseline.png', dpi=150, bbox_inches='tight')
print("\nEDA plots saved to outputs/step2_eda_baseline.png")

print("\n" + "=" * 55)
print(f"SUMMARY:")
print(f"  Random NDCG@10  = {random_ndcg:.4f}  (dumb baseline)")
print(f"  BM25   NDCG@10  = {bm25_ndcg:.4f}  (strong baseline)")
print(f"  Our ML target   > {bm25_ndcg:.4f}  (goal: +15-20%)")
print("=" * 55)
print("\nNext: Run step3_train_model.py to train LambdaMART!")
