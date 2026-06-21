import streamlit as st
from core.cold_storage import ColdStorage
from core.embedding_model import embedding_model
from chatbot import chatbot
from cache.scrl_cache import SCRLCache
import numpy as np

# Initialize resources once at startup using st.cache_resource
@st.cache_resource
def init_cold_storage():
    return ColdStorage(
        host="localhost",
        user="root",
        password="root",
        database="cold_storage"
    )

@st.cache_resource
def init_hot_storage():
    return SCRLCache(capacity=100)

# Initialize storages
cold_storage = init_cold_storage()
hot_storage = init_hot_storage()

# Sidebar with live cache statistics
with st.sidebar:
    st.title("📊 Cache Statistics")

    # Live stats
    stats = hot_storage.get_stats()

    st.metric("Cache Size", stats['size'])
    st.metric("Total Hits", stats['hits'])
    st.metric("Total Misses", stats['misses'])
    st.metric("Hit Rate", f"{stats['hit_rate']*100:.1f}%")

    st.divider()

    # Hedge weights bar chart
    st.subheader("🧠 Expert Weights (Hedge)")
    weights = stats['weights']
    import pandas as pd
    weight_df = pd.DataFrame({
        'Expert': ['SemLRU', 'LFU', 'RDGE'],
        'Weight': [weights['SemLRU'], weights['LFU'], weights['RDGE']]
    })
    st.bar_chart(weight_df.set_index('Expert'))

    st.divider()

    # Expert eviction counts
    st.subheader("🎯 Expert Evictions")
    evictions = stats['expert_evictions']
    st.table({
        'Expert': ['SemLRU', 'LFU', 'RDGE'],
        'Evictions': [evictions['SemLRU'], evictions['LFU'], evictions['RDGE']]
    })

    st.divider()

    # Weight updates count
    st.metric("Hedge Weight Updates", stats['weight_updates'])
    st.metric("History Hits", stats['history_hits'])

# Initialize session state for query history
if 'history' not in st.session_state:
    st.session_state.history = []

# App title
st.title("CacheMind RAG System")

# Text input for query
query = st.chat_input("Enter your query:")

# Submit button

if query:
    # Embed query
    query_embedding = np.array(embedding_model.embed_query(query), dtype=np.float32)

    # Search hot store
    results = hot_storage.request(query_embedding)

    # Check cache hit/miss
    if len(results) >= 1:
        cache_status = "CACHE HIT ✅"
        cache_color = "green"
    else:
        cache_status = "CACHE MISS ❌"
        cache_color = "red"
        # Cache miss: search cold storage
        cold_results_display = []  # Initialize for display
        cold_results = cold_storage.search_bm25(query, top_k=10)
        if cold_results:
            cold_results_display = cold_results  # Store for display
            cold_embeddings = []
            for result in cold_results[:10]:
                # Embed document text
                doc_embedding = np.array(
                    embedding_model.embed_documents([result.text])[0],
                    dtype=np.float32
                )
                # Insert into hot store
                hot_storage.insert(str(result.id), doc_embedding)
                cold_embeddings.append(doc_embedding)
            # Update retrieved scores in hot store
            hot_storage.update_retrieved(cold_embeddings)
            # Search hot store again
            results = hot_storage.request(query_embedding)
        else:
            results = []  # No results from cold storage

    # Build context from results
    context = []
    for doc_id, _ in results:
        doc = cold_storage.get_document(int(doc_id))
        if doc:
            context.append(doc)

    # Invoke chatbot
    if context:
        response = chatbot.invoke({
            "context": "\n".join(context),
            "question": query
        })
    else:
        response = "No relevant documents found."

    # Display cache status
    st.markdown(f"<span style='color:{cache_color}'><b>{cache_status}</b></span>", unsafe_allow_html=True)

    # Display LLM response
    st.subheader("Response:")
    st.write(response)

    # Section 1 — Documents Used for Answer
    st.subheader("📄 Documents Used for Answer")
    for doc_id, sim in results:
        with st.expander(f"Doc ID: {doc_id} | Similarity: {sim:.4f}"):
            if sim > 0.7:
                st.progress(sim, text=f"Similarity: {sim:.4f} 🟢")
            elif sim > 0.4:
                st.progress(sim, text=f"Similarity: {sim:.4f} 🟡")
            else:
                st.progress(sim, text=f"Similarity: {sim:.4f} 🔴")
            doc_text = cold_storage.get_document(int(doc_id))
            st.write(doc_text[:300] if doc_text else "Text not found")

    # Section 2 — On CACHE MISS only, show cold storage results
    if cache_status == "CACHE MISS ❌":
        st.subheader("🧊 Documents Fetched from Cold Storage (BM25)")
        for i, r in enumerate(cold_results_display):
            with st.expander(f"Rank {i+1} | Doc ID: {r.id} | BM25 Score: {r.score:.4f}"):
                st.write(r.text[:200])

    # Section 3 — On CACHE MISS only, show promoted documents
    if cache_status == "CACHE MISS ❌" and cold_results_display:
        st.subheader("🔥 Documents Promoted to Hot Store")
        max_score = cold_results_display[0].score
        cols = st.columns(5)
        for i, r in enumerate(cold_results_display[:10]):
            ce = min(1.0, r.score / max_score)
            with cols[i % 5]:
                st.metric(label=f"Doc {r.id}", value=f"{ce*100:.1f}% CE")

    # Store in history
    st.session_state.history.append({
        "query": query,
        "response": response,
        "cache_hit": len(results) >= 1 if 'results' in locals() else False
    })
else:
    st.warning("Please enter a query.")

# Display query history
if st.session_state.history:
    st.subheader("Query History")
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Query {len(st.session_state.history)-i}: {item['query'][:50]}..."):
            st.write(f"**Query:** {item['query']}")
            st.write(f"**Response:** {item['response']}")
            st.write(f"**Cache:** {'Hit' if item['cache_hit'] else 'Miss'}")