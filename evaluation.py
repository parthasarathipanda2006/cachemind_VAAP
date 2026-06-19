import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
from core.cold_storage import ColdStorage
from core.hot_storage import HotStorage
from core.policy import LRUPolicy,LFUPolicy
from core.embedding_model import embedding_model

# 1. Query Pool
QUERY_POOL = [
    "What is backpropagation in neural networks?",
    "How does gradient descent work?",
    "What are convolutional neural networks?",
    "What is regularization in deep learning?",
    "How does dropout prevent overfitting?",
    "What is a recurrent neural network?",
    "What is the vanishing gradient problem?",
    "How does batch normalization work?",
    "What are generative adversarial networks?",
    "What is the difference between supervised and unsupervised learning?",
    "What is stochastic gradient descent?",
    "How does a feedforward neural network work?",
    "What is the chain rule in deep learning?",
    "What are activation functions in neural networks?",
    "What is the role of the learning rate?",
    "What is an autoencoder?",
    "How does weight initialization affect training?",
    "What is momentum in optimization?",
    "What are recurrent neural networks used for?",
    "What is the difference between RNN and LSTM?",
    "How does attention mechanism work?",
    "What is transfer learning?",
    "What is the curse of dimensionality?",
    "How does max pooling work in CNNs?",
    "What is cross entropy loss?",
]


# 2. Zipf Query Generator
def generate_zipf_queries(query_pool, n=500, alpha=1.2):
    ranks = np.arange(1, len(query_pool) + 1)
    weights = 1 / (ranks ** alpha)
    weights = weights / weights.sum()
    indices = np.random.choice(len(query_pool), size=n, p=weights, replace=True)
    return [query_pool[i] for i in indices]


# 3. LRU Experiment Runner
def run_experiment(queries, cache_size=50):
    cold_store = ColdStorage(
        host='localhost', user='root',
        password='root', database='cold_storage'
    )
    hot_store = HotStorage(
        collection_name='hot_cache_lru',
        embedding_model=embedding_model
    )
    # reset hot store for clean experiment
    try:
        hot_store.client.delete_collection('hot_cache_lru')
        from chromadb.config import Settings
        hot_store.collection = hot_store.client.get_or_create_collection(
            name='hot_cache_lru',
            metadata={"hnsw:space": "cosine"}
        )
    except Exception:
        pass

    policy = LRUPolicy(max_cache_size=cache_size)

    latencies_hit = []
    latencies_miss = []
    cache_sizes = []
    hit_sequence = []

    for query in queries:
        start_time = time.time()
        hot_results = hot_store.search(query, top_k=1)
        hot_latency = (time.time() - start_time) * 1000

        if hot_results and hot_results[0]['similarity'] > 0.5:
            # CACHE HIT
            doc_id = hot_results[0]['id']
            policy.record_access(doc_id)
            latencies_hit.append(hot_latency)
            hit_sequence.append(1)
        else:
            # CACHE MISS
            start_time = time.time()
            cold_results = cold_store.search_bm25(query, top_k=1)
            cold_latency = (time.time() - start_time) * 1000

            if cold_results:
                doc_id = str(cold_results[0].id)
                doc_text = cold_results[0].text

                if not hot_store.document_exists(doc_id):
                    if policy.is_full():
                        victim_id = policy.evict_candidate()
                        if victim_id:
                            hot_store.delete_document(victim_id)
                            policy.remove(victim_id)

                    hot_store.add_document(doc_id, doc_text)
                    policy.admit(doc_id)
                else:
                    policy.record_access(doc_id)

            latencies_miss.append(hot_latency + cold_latency)
            hit_sequence.append(0)

        cache_sizes.append(hot_store.collection.count())

    cold_store.close()

    total = len(queries)
    hit_count = sum(hit_sequence)
    hit_rate = (hit_count / total) * 100 if total > 0 else 0
    avg_hit_latency = np.mean(latencies_hit) if latencies_hit else 0
    avg_miss_latency = np.mean(latencies_miss) if latencies_miss else 0

    return {
        'hits': hit_count,
        'misses': total - hit_count,
        'latencies_hit': latencies_hit,
        'latencies_miss': latencies_miss,
        'cache_sizes': cache_sizes,
        'hit_rate': hit_rate,
        'avg_hit_latency': avg_hit_latency,
        'avg_miss_latency': avg_miss_latency,
        'hit_sequence': hit_sequence,
    }


# 4. No-Cache Baseline
def run_no_cache_baseline(queries):
    cold_store = ColdStorage(
        host='localhost', user='root',
        password='root', database='cold_storage'
    )
    latencies = []
    for query in queries:
        start = time.time()
        cold_store.search_bm25(query, top_k=1)
        latencies.append((time.time() - start) * 1000)
    cold_store.close()
    return np.mean(latencies) if latencies else 0


# 5. Graphs
def generate_graphs(results, no_cache_avg_latency):
    plt.style.use('seaborn-v0_8')

    # Graph 1 — Hit Rate Curve
    plt.figure(figsize=(10, 6))
    cumulative_hits = np.cumsum(results['hit_sequence'])
    query_numbers = np.arange(1, len(results['hit_sequence']) + 1)
    cumulative_hit_rate = (cumulative_hits / query_numbers) * 100
    plt.plot(query_numbers, cumulative_hit_rate, linewidth=2, color='blue', label='LRU')
    plt.xlabel('Query Number')
    plt.ylabel('Cumulative Hit Rate (%)')
    plt.title('Cache Hit Rate Over Time (Zipf Distribution)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('hit_rate_curve.png', dpi=300)
    plt.close()
    print("Saved hit_rate_curve.png")

    # Graph 2 — Latency Comparison
    plt.figure(figsize=(10, 6))
    categories = ['No Cache', 'LRU Miss', 'LRU Hit']
    latencies = [
        no_cache_avg_latency,
        results['avg_miss_latency'],
        results['avg_hit_latency'],
    ]
    colors = ['red', 'orange', 'green']
    bars = plt.bar(categories, latencies, color=colors, alpha=0.8, edgecolor='black')
    plt.ylabel('Average Latency (ms)')
    plt.title('Average Query Latency by Retrieval Path')
    plt.grid(True, alpha=0.3, axis='y')
    for bar, lat in zip(bars, latencies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(latencies) * 0.01,
            f'{lat:.1f}ms', ha='center', va='bottom', fontweight='bold'
        )
    plt.tight_layout()
    plt.savefig('latency_comparison.png', dpi=300)
    plt.close()
    print("Saved latency_comparison.png")


# 6. Summary
def print_summary(results, no_cache_avg_latency, cache_size):
    print("\nPolicy    | Hit Rate | Avg Hit Latency | Avg Miss Latency | Docs Cached")
    print("-" * 70)
    print(f"LRU       | {results['hit_rate']:.0f}%       | {results['avg_hit_latency']:.1f}ms          | {results['avg_miss_latency']:.1f}ms           | {cache_size}")
    print(f"No Cache  | -        | -               | {no_cache_avg_latency:.1f}ms           | 0")


# 7. Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CacheMind RAG system')
    parser.add_argument('--quick', action='store_true', help='Run 100 queries for fast testing')
    args = parser.parse_args()

    n_queries = 100 if args.quick else 200
    cache_size = 10

    print(f"Generating {n_queries} queries using Zipf distribution...")
    queries = generate_zipf_queries(QUERY_POOL, n=n_queries, alpha=1.2)

    print("Running no-cache baseline...")
    no_cache_avg_latency = run_no_cache_baseline(queries)

    print("Running LRU experiment...")
    results = run_experiment(queries, cache_size=cache_size)

    print("Generating graphs...")
    generate_graphs(results, no_cache_avg_latency)

    print_summary(results, no_cache_avg_latency, cache_size)