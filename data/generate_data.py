"""
STEP 1: Generate Search Ranking Dataset
========================================
In a real job, you'd use MSLR-WEB10K (Microsoft Learning to Rank dataset).
Here we simulate the same structure so you understand every feature.

WHAT IS A SEARCH RANKING DATASET?
- Each row = one (query, document) pair
- Features = signals about how relevant the doc is to the query
- Label = relevance score (0=irrelevant, 1=fair, 2=good, 3=excellent, 4=perfect)

REAL FEATURES GOOGLE/BING USE:
- BM25 score (keyword overlap)
- TF-IDF score
- PageRank of the document
- Click-through rate (CTR)
- Query-document title match
- Document freshness
- User dwell time, etc.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_dataset(n_queries=500, docs_per_query=20):
    """
    Generate a realistic search ranking dataset.
    
    Args:
        n_queries: number of unique search queries
        docs_per_query: number of candidate documents per query
    
    Returns:
        DataFrame with features + relevance labels
    """
    rows = []

    for query_id in range(n_queries):
        # Each query has a "true difficulty" — some queries are harder to rank
        query_difficulty = np.random.uniform(0.3, 1.0)

        for doc_id in range(docs_per_query):
            # ---- FEATURE 1: BM25 score ----
            # BM25 = classic keyword overlap score (baseline method)
            # Higher = more keyword matches between query and doc
            bm25 = np.random.exponential(scale=2.0)

            # ---- FEATURE 2: TF-IDF score ----
            # Term frequency × Inverse document frequency
            tfidf = np.random.exponential(scale=1.5)

            # ---- FEATURE 3: PageRank ----
            # How important/authoritative is this page on the web?
            pagerank = np.random.beta(a=2, b=5)  # most pages have low pagerank

            # ---- FEATURE 4: Click-Through Rate (CTR) ----
            # Historical % of users who clicked this result for similar queries
            ctr = np.random.beta(a=1, b=10)  # most pages have low CTR

            # ---- FEATURE 5: Title match score ----
            # Does the query appear in the document's title?
            title_match = np.random.uniform(0, 1)

            # ---- FEATURE 6: URL depth ----
            # Deep URLs (example.com/a/b/c/d) tend to be less relevant
            url_depth = np.random.randint(1, 8)

            # ---- FEATURE 7: Document length (normalized) ----
            doc_length = np.random.lognormal(mean=6, sigma=1)
            doc_length_norm = min(doc_length / 10000, 1.0)

            # ---- FEATURE 8: Freshness score ----
            # How recently was the document updated?
            freshness = np.random.uniform(0, 1)

            # ---- FEATURE 9: Exact query match in body ----
            exact_match = np.random.binomial(1, 0.3)

            # ---- FEATURE 10: Domain authority ----
            domain_authority = np.random.beta(a=2, b=3)

            # ---- COMPUTE RELEVANCE LABEL ----
            # In real datasets, humans label this (0-4 scale)
            # We simulate it based on a weighted combination of features
            # (this mirrors how actual relevance works)
            relevance_score = (
                0.30 * bm25 / 5 +           # keyword match matters most
                0.20 * tfidf / 4 +           # tf-idf is also important
                0.15 * pagerank +            # authority matters
                0.15 * ctr * 5 +             # user behavior signal
                0.10 * title_match +         # title match
                0.05 * (1 - url_depth/8) +  # simpler URLs preferred
                0.05 * freshness             # fresh content preferred
            ) * query_difficulty

            # Add noise (relevance isn't perfectly predictable)
            relevance_score += np.random.normal(0, 0.1)
            relevance_score = np.clip(relevance_score, 0, 1)

            # Convert continuous score → discrete label (0, 1, 2, 3, 4)
            if relevance_score < 0.2:
                label = 0   # irrelevant
            elif relevance_score < 0.4:
                label = 1   # fair
            elif relevance_score < 0.6:
                label = 2   # good
            elif relevance_score < 0.8:
                label = 3   # excellent
            else:
                label = 4   # perfect

            rows.append({
                'query_id': query_id,
                'doc_id': doc_id,
                'label': label,                    # TARGET: relevance (0-4)
                'bm25_score': round(bm25, 4),
                'tfidf_score': round(tfidf, 4),
                'pagerank': round(pagerank, 4),
                'ctr': round(ctr, 4),
                'title_match': round(title_match, 4),
                'url_depth': url_depth,
                'doc_length_norm': round(doc_length_norm, 4),
                'freshness': round(freshness, 4),
                'exact_match': exact_match,
                'domain_authority': round(domain_authority, 4),
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset(n_queries=500, docs_per_query=20)
    
    # Save to CSV
    df.to_csv('data/search_ranking_data.csv', index=False)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total (query, doc) pairs: {len(df)}")
    print(f"Unique queries: {df['query_id'].nunique()}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().sort_index())
    print(f"\nSample rows:")
    print(df.head(3).to_string())
    print("\nDataset saved to data/search_ranking_data.csv")
