from core.hot_storage import HotStorage
from core.cold_storage import ColdStorage
from core.embedding_model import embedding_model
from core.policy import LRUPolicy
from chatbot import chatbot

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
policy = LRUPolicy(max_cache_size=10)

while True:
    context=[]
    query = input("tell me your Query:\n")
    if query == "exit":
        break

    # Search Hot Storage (ChromaDB)
    hot_results = hot_storage.search(query, top_k=3)

    # Check if any hot result meets the similarity threshold
    hit_found = False
    if hot_results:
        for res in hot_results:
            if res.get('similarity', 0) > 0.5:
                hit_found = True

    if hit_found:
        # CACHE HIT - never touches cold store
        print("CACHE HIT ✅")
        # Update recency for each hit document
        for res in reversed(hot_results):
            if res.get('similarity', 0) > 0.3:
                #print(res['text'])
                doc_id = res['id']
                if res.get('similarity', 0) > 0.3:
                    policy.record_access(doc_id)
                    context.append(res["text"])
            else:
                doc_id = res['id']
                print(res["similarity"])
                hot_storage.delete_document(doc_id)
                policy.remove(doc_id)
        
    else:
        # CACHE MISS - go to cold store
        print("CACHE MISS ❌")
        cold_results = cold_storage.search_bm25(query, top_k=10)
        if not cold_results:
            print("No results found in cold storage.")
            continue
        for result in cold_results[:5]:
        # Take the top retrieved document (as per original logic)
            print("\n",result.score,"\n")
            top_doc = result
            doc_id = str(top_doc.id)
            doc_text = top_doc.text
            #print("\n",doc_text,"\n")
            # Check if document is already in hot cache (shouldn't be on miss, but safeguard)
            if policy.contains(doc_id):
                pass 
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

        result = hot_storage.search(
            query=query,
            top_k=5
        )
        for res in reversed(result):
            if res.get('similarity', 0) > 0.3:
                #print(res['text'])
                doc_id = res['id']
                if res.get('similarity', 0) > 0.3:
                    policy.record_access(doc_id)
                    context.append(res["text"])
            else:
                doc_id = res['id']
                hot_storage.delete_document(doc_id)
                print(res["similarity"])
                policy.remove(doc_id)

    if context==[]:
        print("No valid results found in cold storage.")
        continue
    response = chatbot.invoke({
        "context":  "\n".join(context),
        "question": query
    })
    print(response.content)

cold_storage.close()