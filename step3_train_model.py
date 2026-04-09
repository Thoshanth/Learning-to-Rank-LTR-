"""
STEP 3: Train LambdaMART Model
================================
LambdaMART = the industry standard Learning-to-Rank algorithm.
Used by: Bing, Yahoo, LinkedIn, Airbnb, etc.

HOW LAMBDAMART WORKS (simplified):
1. Start with BM25 scores as initial ranking
2. Find pairs of documents where the ranking is WRONG
   (irrelevant doc ranked above relevant doc)
3. Train a gradient boosted tree to fix those wrong pairs
4. Repeat hundreds of times, each tree fixing remaining errors

WHY NOT JUST USE REGRESSION?
- Regression optimizes absolute scores (MSE)
- Ranking cares about RELATIVE ORDER, not exact scores
- A doc with predicted score 0.9 ranked above 0.8 is fine
  even if true relevance is 3 and 2 respectively

LAMBDARANK LOSS (the math):
- For each wrong pair (i,j) where rel_i > rel_j but score_j > score_i
- Compute lambda_ij = |ΔNDCG| × sigmoid(score_j - score_i)
- |ΔNDCG| = how much NDCG improves if we swap their positions
- Trees are trained to predict these lambdas (gradients)

INSTALL FOR YOUR LOCAL MACHINE:
    pip install lightgbm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# ============================================================
# COPY NDCG functions from step 2
# ============================================================

def dcg_at_k(relevances, k=10):
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    gains = 2 ** relevances - 1
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(gains / discounts)

def ndcg_at_k(relevances, k=10):
    actual_dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

def evaluate_ndcg(df, score_col, k=10):
    ndcg_scores = []
    for query_id, group in df.groupby('query_id'):
        ranked = group.sort_values(score_col, ascending=False)
        relevances = ranked['label'].tolist()
        ndcg_scores.append(ndcg_at_k(relevances, k))
    return np.mean(ndcg_scores)

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

print("=" * 55)
print("STEP 3: Training LambdaMART (LTR Model)")
print("=" * 55)

df = pd.read_csv('data/search_ranking_data.csv')

# CRITICAL: Split by query_id, NOT by row
# If you split by row, you leak — the model sees some docs from
# query 5 in training and predicts query 5 docs in test!
query_ids = df['query_id'].unique()
np.random.seed(42)
np.random.shuffle(query_ids)

n = len(query_ids)
train_ids = query_ids[:int(0.7 * n)]   # 70% queries → train
val_ids   = query_ids[int(0.7*n):int(0.85*n)]  # 15% → val
test_ids  = query_ids[int(0.85*n):]    # 15% → test

train_df = df[df['query_id'].isin(train_ids)].copy()
val_df   = df[df['query_id'].isin(val_ids)].copy()
test_df  = df[df['query_id'].isin(test_ids)].copy()

print(f"\nSplit (by query — no leakage):")
print(f"  Train: {len(train_ids)} queries → {len(train_df):,} rows")
print(f"  Val:   {len(val_ids)} queries → {len(val_df):,} rows")
print(f"  Test:  {len(test_ids)} queries → {len(test_df):,} rows")

FEATURES = ['bm25_score','tfidf_score','pagerank','ctr',
            'title_match','url_depth','doc_length_norm',
            'freshness','exact_match','domain_authority']

X_train = train_df[FEATURES].values
y_train = train_df['label'].values
X_val   = val_df[FEATURES].values
y_val   = val_df['label'].values
X_test  = test_df[FEATURES].values
y_test  = test_df['label'].values

# ============================================================
# TRY LIGHTGBM (LambdaMART) FIRST
# ============================================================

try:
    import lightgbm as lgb
    USE_LIGHTGBM = True
    print("\nLightGBM found! Using true LambdaMART.")
except ImportError:
    USE_LIGHTGBM = False
    print("\nLightGBM not installed. Using GradientBoosting (same family).")
    print("To use true LambdaMART locally: pip install lightgbm")

# ============================================================
# TRAIN MODEL
# ============================================================

train_ndcg_history = []
val_ndcg_history = []

if USE_LIGHTGBM:
    # ---- TRUE LAMBDAMART with LightGBM ----
    
    # group = how many docs per query (needed for ranking loss)
    train_group = train_df.groupby('query_id').size().values
    val_group   = val_df.groupby('query_id').size().values
    
    train_data = lgb.Dataset(X_train, label=y_train, group=train_group)
    val_data   = lgb.Dataset(X_val,   label=y_val,   group=val_group)
    
    params = {
        'objective': 'lambdarank',   # THE KEY PARAM — LambdaMART loss
        'metric': 'ndcg',
        'ndcg_eval_at': [10],
        'num_leaves': 31,            # max leaves per tree (complexity)
        'learning_rate': 0.05,       # smaller = more trees needed but better
        'min_data_in_leaf': 20,      # prevents overfitting
        'n_estimators': 200,
        'verbose': -1,
    }
    
    callbacks = [lgb.early_stopping(20), lgb.log_evaluation(20)]
    model = lgb.train(params, train_data, 
                      valid_sets=[val_data],
                      callbacks=callbacks)
    
    # Get training history
    train_scores = model.best_score
    
    # Predict
    train_df['ml_score'] = model.predict(X_train)
    val_df['ml_score']   = model.predict(X_val)
    test_df['ml_score']  = model.predict(X_test)

else:
    # ---- GRADIENT BOOSTING (sklearn — same tree boosting idea) ----
    # This uses regression on relevance labels.
    # Not exactly LambdaMART, but demonstrates the same concept.
    # Replace with LightGBM locally for true LambdaMART.
    
    print("\nTraining Gradient Boosted Trees...")
    print("(Tracking NDCG@10 on val set every 10 trees)\n")
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Train progressively and track NDCG (simulates LightGBM's eval)
    n_estimators_list = list(range(10, 201, 10))
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        subsample=0.8,           # stochastic gradient boosting
        random_state=42,
        warm_start=True,
    )
    
    for n_est in n_estimators_list:
        model.n_estimators = n_est
        model.fit(X_train, y_train)
        
        val_df_tmp = val_df.copy()
        val_df_tmp['ml_score'] = model.predict(X_val)
        val_ndcg = evaluate_ndcg(val_df_tmp, 'ml_score', k=10)
        
        train_df_tmp = train_df.copy()
        train_df_tmp['ml_score'] = model.predict(X_train)
        train_ndcg = evaluate_ndcg(train_df_tmp, 'ml_score', k=10)
        
        train_ndcg_history.append(train_ndcg)
        val_ndcg_history.append(val_ndcg)
        
        if n_est % 50 == 0:
            print(f"  Trees: {n_est:3d} | Train NDCG@10: {train_ndcg:.4f} | Val NDCG@10: {val_ndcg:.4f}")
    
    # Final predictions
    train_df['ml_score'] = model.predict(X_train)
    val_df['ml_score']   = model.predict(X_val)
    test_df['ml_score']  = model.predict(X_test)

# ============================================================
# EVALUATE: ML vs BM25
# ============================================================

print("\n" + "=" * 55)
print("EVALUATION RESULTS")
print("=" * 55)

metrics = {}
for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    bm25_ndcg  = evaluate_ndcg(split_df, 'bm25_score', k=10)
    ml_ndcg    = evaluate_ndcg(split_df, 'ml_score', k=10)
    improvement = (ml_ndcg - bm25_ndcg) / bm25_ndcg * 100
    metrics[split_name] = {'bm25': bm25_ndcg, 'ml': ml_ndcg, 'pct': improvement}
    
    print(f"\n{split_name} Set:")
    print(f"  BM25 NDCG@10  = {bm25_ndcg:.4f}")
    print(f"  ML   NDCG@10  = {ml_ndcg:.4f}  (+{improvement:.1f}% improvement)")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if USE_LIGHTGBM:
    importances = model.feature_importance(importance_type='gain')
else:
    importances = model.feature_importances_

feat_imp = pd.DataFrame({
    'feature': FEATURES,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nFeature Importance (which signals matter most):")
for _, row in feat_imp.iterrows():
    bar = "█" * int(row['importance'] / feat_imp['importance'].max() * 25)
    print(f"  {row['feature']:<20} {bar}  {row['importance']:.1f}")

# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs('models', exist_ok=True)
with open('models/ranking_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved to models/ranking_model.pkl")

# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Step 3: Model Training Results', fontsize=14, fontweight='bold')

# Plot 1: Training curve (NDCG vs trees)
ax = axes[0]
if train_ndcg_history:
    ax.plot(n_estimators_list, train_ndcg_history, 'b-o', markersize=3, label='Train NDCG@10')
    ax.plot(n_estimators_list, val_ndcg_history, 'r-o', markersize=3, label='Val NDCG@10')
    ax.axhline(metrics['Val']['bm25'], color='orange', linestyle='--', label=f"BM25 ({metrics['Val']['bm25']:.3f})")
    ax.set_title('Learning Curve\n(NDCG@10 vs. Number of Trees)')
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('NDCG@10')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 2: BM25 vs ML comparison
ax = axes[1]
splits = ['Train', 'Val', 'Test']
x = np.arange(len(splits))
width = 0.35
bm25_vals = [metrics[s]['bm25'] for s in splits]
ml_vals   = [metrics[s]['ml']   for s in splits]
b1 = ax.bar(x - width/2, bm25_vals, width, label='BM25 Baseline', color='#f57c00')
b2 = ax.bar(x + width/2, ml_vals,   width, label='LambdaMART ML', color='#388e3c')
ax.set_title('NDCG@10: BM25 vs ML Model')
ax.set_ylabel('NDCG@10')
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.legend()
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{bar.get_height():.3f}', ha='center', fontsize=9)
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')

# Plot 3: Feature importance
ax = axes[2]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_imp)))
bars = ax.barh(feat_imp['feature'], feat_imp['importance'], color=colors)
ax.set_title('Feature Importance\n(Which signals matter most?)')
ax.set_xlabel('Importance (Gain)')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/step3_training_results.png', dpi=150, bbox_inches='tight')
print("Training plots saved to outputs/step3_training_results.png")

# Save metrics for next step
metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv('outputs/model_metrics.csv')

test_score = metrics['Test']['ml']
test_bm25  = metrics['Test']['bm25']
pct = metrics['Test']['pct']
print(f"\n{'='*55}")
print(f"FINAL TEST SET RESULT:")
print(f"  BM25 NDCG@10 = {test_bm25:.4f}")
print(f"  ML   NDCG@10 = {test_score:.4f}  (+{pct:.1f}% over BM25)")
print(f"\nYOUR RESUME BULLET (use your actual numbers):")
print(f'  "Built a learning-to-rank model using GradientBoosting')
print(f'   on 10K (query, doc) pairs"')
print(f'  "Improved NDCG@10 by {pct:.0f}% over BM25 baseline"')
print(f"{'='*55}")
print("\nNext: Run step4_evaluate_resume.py for full evaluation!")
