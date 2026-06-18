from hot_storage import HotStorage
from cold_memory import ColdStorage
from embedding_model import embedding_model
from policy import LRUPolicy

# Initialize storages
cold_storage = ColdStorage(
    host="localhost",
    user="root",
    password="root",
    database="cold_storage"
)
hot_storage = HotStorage(
    collection_name="hot_cache",
    embedding_model=embedding_model
)

# Initialize LRU policy with a maximum cache size (e.g., 100 documents)
# This size can be adjusted as needed
policy = LRUPolicy(max_cache_size=100)

while True:
    query = input("tell me your Query:\n")
    if query == "exit":
        break

    # Search Hot Storage (ChromaDB)
    hot_results = hot_storage.search(query, top_k=3)

    # Check if any hot result meets the similarity threshold
    hit_found = False
    if hot_results:
        for res in hot_results:
            if res.get('similarity', 0) > 0.3:
                hit_found = True
                break

    if hit_found:
        # CACHE HIT - never touches cold store
        print("CACHE HIT ✅")
        # Update recency for each hit document
        for res in hot_results:
            doc_id = res['id']
            policy.record_access(doc_id)
            print(res['text'])

    else:
        # CACHE MISS - go to cold store
        print("CACHE MISS ❌")
        cold_results = cold_storage.search_bm25(query, top_k=10)
        if not cold_results:
            print("No results found in cold storage.")
            continue
        for result in cold_results[:5]:
        # Take the top retrieved document (as per original logic)
            top_doc = result
            doc_id = str(top_doc.id)
            doc_text = top_doc.text
            print("\n",doc_text,"\n")
            # Check if document is already in hot cache (shouldn't be on miss, but safeguard)
            if policy.contains(doc_id):
                print(f"Document {doc_id} already in cache, updating access")
                policy.record_access(doc_id)
            else:
                # If cache is full, evict least recently used document
                if policy.is_full():
                    victim_id = policy.evict_candidate()
                    if victim_id is not None:
                        print(f"Evicting LRU document: {victim_id}")
                        hot_storage.delete_document(victim_id)
                        policy.remove(victim_id)

                # Insert new document into Hot Storage (ChromaDB)
                hot_storage.add_document(doc_id, doc_text)
                # Register the document in LRU policy as most recently used
                policy.admit(doc_id)
                print(f"Admitted new document: {doc_id}")

        # Print all cold results (as per original logic)
        for result in cold_results:
            print(result.text)

    # Final search to display results (keeping original logic for compatibility)
    result = hot_storage.search(
        query=query,
        top_k=5
    )
    k = False
    for i, res in enumerate(result, 1):
        if res["similarity"] > 0.3:
            k = True
            print(f"{i}. ID: {res['id']}")
            print(f"   Text: {res['text']}")
            print(f"   similarity: {res['similarity']}")
            print()
    if k == False:
        print("CACHE_MISS")
    else:
        print("CACHE_HIT")
    k = False
cold_storage.close()