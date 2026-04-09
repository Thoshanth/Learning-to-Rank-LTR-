"""
STEP 4: Final Evaluation + Resume Bullet Generator
====================================================
This is the final step — polish everything for your portfolio.

This script has been refactored to use object-oriented patterns,
proper logging, type hinting, and maintainable data flows.
"""

import argparse
import logging
import pickle
import os
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RankingEvaluator:
    """Handles the computation of ranking metrics such as NDCG."""

    @staticmethod
    def dcg_at_k(relevances: List[float], k: int = 10) -> float:
        """Calculates Discounted Cumulative Gain at K."""
        relevances_arr = np.array(relevances[:k], dtype=float)
        if len(relevances_arr) == 0:
            return 0.0
        gains = 2 ** relevances_arr - 1
        discounts = np.log2(np.arange(2, len(relevances_arr) + 2))
        return float(np.sum(gains / discounts))

    @staticmethod
    def ndcg_at_k(relevances: List[float], k: int = 10) -> float:
        """Calculates Normalized Discounted Cumulative Gain at K."""
        actual_dcg = RankingEvaluator.dcg_at_k(relevances, k)
        ideal_relevances = sorted(relevances, reverse=True)
        ideal_dcg = RankingEvaluator.dcg_at_k(ideal_relevances, k)
        if ideal_dcg == 0:
            return 0.0
        return actual_dcg / ideal_dcg

    @staticmethod
    def evaluate_ndcg(df: pd.DataFrame, score_col: str, k: int = 10) -> float:
        """Evaluates mean NDCG@K across all queries in a DataFrame."""
        ndcg_scores = []
        for _, group in df.groupby('query_id'):
            ranked = group.sort_values(score_col, ascending=False)
            relevances = ranked['label'].tolist()
            ndcg_scores.append(RankingEvaluator.ndcg_at_k(relevances, k))
        return float(np.mean(ndcg_scores))

    @staticmethod
    def full_evaluation(test_df: pd.DataFrame, k_values: List[int]) -> Dict[str, Dict[str, float]]:
        """Evaluates multiple score columns across multiple K values."""
        results = {}
        evaluation_configs = [
            ('Random', 'random_score'), 
            ('BM25', 'bm25_score'),
            ('ML Model', 'ml_score')
        ]
        
        for method, col in evaluation_configs:
            results[method] = {}
            for k in k_values:
                results[method][f'NDCG@{k}'] = RankingEvaluator.evaluate_ndcg(test_df, col, k)
        return results

    @staticmethod
    def per_query_analysis(test_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """Computes query-level metrics comparing BM25 vs ML model."""
        per_query = []
        for query_id, group in test_df.groupby('query_id'):
            bm25_relevances = group.sort_values('bm25_score', ascending=False)['label'].tolist()
            ml_relevances = group.sort_values('ml_score', ascending=False)['label'].tolist()
            
            bm25_n = RankingEvaluator.ndcg_at_k(bm25_relevances, k)
            ml_n = RankingEvaluator.ndcg_at_k(ml_relevances, k)
            
            per_query.append({
                'query_id': query_id, 
                'bm25_ndcg': bm25_n, 
                'ml_ndcg': ml_n,
                'improvement': ml_n - bm25_n
            })
        
        return pd.DataFrame(per_query)


class EvaluationVisualizer:
    """Handles the creation and saving of evaluation plots."""
    
    def __init__(self, output_path: str):
        self.output_path = output_path

    def plot_results(self, results: Dict[str, Dict[str, float]], 
                     pq_df: pd.DataFrame, k_values: List[int]) -> None:
        """Generates a 2x2 subplot comparing model evaluation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Project 1: Search Ranking Model — Final Results', 
                     fontsize=14, fontweight='bold')

        self._plot_ndcg_curves(axes[0, 0], results, k_values)
        self._plot_improvement_dist(axes[0, 1], pq_df)
        self._plot_scatter_comparison(axes[1, 0], pq_df)
        self._plot_final_summary(axes[1, 1], results)

        plt.tight_layout()
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        plt.savefig(self.output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Final results saved to {self.output_path}")

    def _plot_ndcg_curves(self, ax: plt.Axes, results: Dict[str, Dict[str, float]], k_values: List[int]) -> None:
        colors_map = {'Random': '#d32f2f', 'BM25': '#f57c00', 'ML Model': '#388e3c'}
        markers = {'Random': 's', 'BM25': 'o', 'ML Model': 'D'}
        for method in ['Random', 'BM25', 'ML Model']:
            vals = [results[method][f'NDCG@{k}'] for k in k_values]
            ax.plot(k_values, vals, marker=markers[method], color=colors_map[method],
                    linewidth=2.5, markersize=7, label=method)
        ax.set_title('NDCG@K Comparison (Test Set)')
        ax.set_xlabel('K')
        ax.set_ylabel('NDCG@K')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)

    def _plot_improvement_dist(self, ax: plt.Axes, pq_df: pd.DataFrame) -> None:
        improvements = pq_df['improvement'].values
        ax.hist(improvements, bins=25, color='#1976d2', edgecolor='white', alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax.axvline(improvements.mean(), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {improvements.mean():+.3f}')
        ax.set_title('Per-Query NDCG Improvement\n(ML minus BM25)')
        ax.set_xlabel('NDCG@10 improvement per query')
        ax.set_ylabel('Number of queries')
        ax.legend()

    def _plot_scatter_comparison(self, ax: plt.Axes, pq_df: pd.DataFrame) -> None:
        pct_better = (pq_df['improvement'] > 0).mean() * 100
        pct_worse = (pq_df['improvement'] < 0).mean() * 100
        
        colors_scatter = ['#388e3c' if imp > 0 else '#d32f2f' for imp in pq_df['improvement']]
        ax.scatter(pq_df['bm25_ndcg'], pq_df['ml_ndcg'], 
                   c=colors_scatter, alpha=0.5, s=30)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Equal performance')
        ax.set_title('Per-Query: BM25 vs ML NDCG@10\n(green = ML wins, red = BM25 wins)')
        ax.set_xlabel('BM25 NDCG@10')
        ax.set_ylabel('ML NDCG@10')
        
        better_patch = mpatches.Patch(color='#388e3c', label=f'ML better ({pct_better:.0f}%)')
        worse_patch = mpatches.Patch(color='#d32f2f', label=f'BM25 better ({pct_worse:.0f}%)')
        ax.legend(handles=[better_patch, worse_patch])

    def _plot_final_summary(self, ax: plt.Axes, results: Dict[str, Dict[str, float]]) -> None:
        bm25_ndcg10 = results['BM25']['NDCG@10']
        ml_ndcg10 = results['ML Model']['NDCG@10']
        improvement = (ml_ndcg10 - bm25_ndcg10) / bm25_ndcg10 * 100

        methods_bar = ['Random\nBaseline', 'BM25\nBaseline', 'LambdaMART\nML Model']
        ndcg_vals = [results['Random']['NDCG@10'], bm25_ndcg10, ml_ndcg10]
        colors_bar = ['#d32f2f', '#f57c00', '#388e3c']
        
        bars = ax.bar(methods_bar, ndcg_vals, color=colors_bar, width=0.5, edgecolor='white')
        ax.set_title(f'NDCG@10 Summary\n(ML beats BM25 by +{improvement:.1f}%)')
        ax.set_ylabel('NDCG@10')
        ax.set_ylim(0, min(1.0, max(ndcg_vals) * 1.15))
        
        for bar, val in zip(bars, ndcg_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.annotate(f'+{improvement:.1f}%', 
                    xy=(2, ml_ndcg10), xytext=(1.5, ml_ndcg10 + 0.04),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    color='green', fontweight='bold', fontsize=12)


class ResumeReportGenerator:
    """Generates the final terminal output, summary metrics, and resume bullets."""

    @staticmethod
    def print_evaluation_summary(results: Dict[str, Dict[str, float]], pq_df: pd.DataFrame) -> None:
        """Prints a summary of the evaluations metrics."""
        results_df = pd.DataFrame(results).T
        print("=" * 60)
        print("STEP 4: Final Evaluation + Resume Bullet Generator")
        print("=" * 60)
        print("\nNDCG Scores at Multiple K Values (Test Set):")
        print(results_df.round(4).to_string())

        bm25_ndcg10 = results['BM25']['NDCG@10']
        ml_ndcg10 = results['ML Model']['NDCG@10']
        improvement = (ml_ndcg10 - bm25_ndcg10) / bm25_ndcg10 * 100

        print(f"\nKey Result:")
        print(f"  BM25 NDCG@10 = {bm25_ndcg10:.4f}")
        print(f"  ML   NDCG@10 = {ml_ndcg10:.4f}")
        print(f"  Improvement  = +{improvement:.1f}%")

        pct_better = (pq_df['improvement'] > 0).mean() * 100
        pct_worse = (pq_df['improvement'] < 0).mean() * 100

        print(f"\nPer-Query Analysis ({len(pq_df)} test queries):")
        print(f"  ML better than BM25: {pct_better:.0f}% of queries")
        print(f"  ML worse than BM25:  {pct_worse:.0f}% of queries")
        print(f"  Avg improvement: {pq_df['improvement'].mean():+.4f}")

    @staticmethod
    def print_resume_bullets(total_dataset_size: int, test_set_size: int, 
                             improvement_pct: float, query_win_pct: float) -> None:
        """Generates dynamic resume bullet points."""
        print("\n" + "=" * 60)
        print("YOUR RESUME BULLETS (based on YOUR actual results)")
        print("=" * 60)
        
        data_size_k = int(total_dataset_size / 1000)
        
        print(f"""
Project: Search Ranking Model
------------------------------
• Built a learning-to-rank model using Gradient Boosted Trees
  on {data_size_k}K (query, document) pairs with 10 ranking signals
  including BM25, PageRank, CTR, and title match features

• Improved NDCG@10 by {improvement_pct:.0f}% over BM25 keyword baseline
  on held-out test set ({test_set_size} queries), with ML ranking
  superior on {query_win_pct:.0f}% of individual queries
""")

    @staticmethod
    def print_interview_prep() -> None:
        """Provides common interview Q&A for ranking models."""
        print("=" * 60)
        print("INTERVIEW Q&A PREP")
        print("=" * 60)
        print("""
Q: Why did you use NDCG instead of Precision/Recall?
A: Because in search ranking, position matters. A highly relevant document at 
   position 1 is much more valuable than one at position 10. NDCG penalizes 
   (discounts) relevance scores linearly with logarithmic position.

Q: How did you handle Position Bias / Click Bias?
A: In this dataset, CTR was used as a feature, which inherently carries position 
   bias. Ideally, in production, we would use strict randomized data or inverse 
   propensity weighting (IPW), but this served as a solid baseline.

Q: Why a Gradient Boosted Tree (LambdaMART) over Neural Networks?
A: GBDTs are traditionally state-of-the-art for tabular/dense feature ranking due 
   to robustness against unscaled data and smaller datasets. They require less tuning
   compared to deep learning rankers like RankNet.
        """)

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate search ranking model")
    parser.add_argument('--data_file', type=str, default='data/search_ranking_data.csv',
                        help='Path to the formatted CSV data')
    parser.add_argument('--model_file', type=str, default='models/ranking_model.pkl',
                        help='Path to the trained ranking model')
    parser.add_argument('--output_img', type=str, default='outputs/step4_final_results.png',
                        help='Path to save the generated matplotlib plot')
    args = parser.parse_args()

    logger.info("Loading data and model...")
    df = pd.read_csv(args.data_file)
    
    with open(args.model_file, 'rb') as f:
        model = pickle.load(f)

    # Prepare datasets
    query_ids = df['query_id'].unique()
    np.random.seed(42)
    np.random.shuffle(query_ids)
    
    n = len(query_ids)
    test_ids = query_ids[int(0.85 * n):]
    test_df = df[df['query_id'].isin(test_ids)].copy()

    FEATURES = ['bm25_score', 'tfidf_score', 'pagerank', 'ctr',
                'title_match', 'url_depth', 'doc_length_norm',
                'freshness', 'exact_match', 'domain_authority']

    # Predict
    logger.info("Generating predictions...")
    test_df['ml_score'] = model.predict(test_df[FEATURES].values)
    test_df['random_score'] = np.random.rand(len(test_df))

    # Evaluate
    logger.info("Evaluating NDCG metrics...")
    k_values = [1, 3, 5, 10, 20]
    results = RankingEvaluator.full_evaluation(test_df, k_values)
    pq_df = RankingEvaluator.per_query_analysis(test_df, k=10)

    # Visualize
    logger.info("Plotting evaluation distributions...")
    visualizer = EvaluationVisualizer(args.output_img)
    visualizer.plot_results(results, pq_df, k_values)

    # Report
    ResumeReportGenerator.print_evaluation_summary(results, pq_df)
    
    improvement = (results['ML Model']['NDCG@10'] - results['BM25']['NDCG@10']) / results['BM25']['NDCG@10'] * 100
    win_pct = (pq_df['improvement'] > 0).mean() * 100
    
    ResumeReportGenerator.print_resume_bullets(
        total_dataset_size=len(df),
        test_set_size=len(test_ids),
        improvement_pct=improvement,
        query_win_pct=win_pct
    )
    
    ResumeReportGenerator.print_interview_prep()
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
