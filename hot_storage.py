import chromadb
from chromadb.config import Settings
from typing import List, Dict, Union, Optional


class HotStorage:

    def __init__(self, collection_name: str, embedding_model):

        self.embedding_model = embedding_model

        # Initialize ChromaDB client (in-memory for hot storage cache)
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )

        # Get or create collection with cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine distance for similarity
        )

    def add_document(self, doc_id: str, text: str) -> None:

        embedding = self.embedding_model.embed_query(text)

        # Store in ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text]
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:

        # Generate embedding for the query
        query_embedding = self.embedding_model.embed_query(query)

        # Query ChromaDB for similar documents
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'distances']  # We need documents and distances
        )

        # Format results
        output = []
        if results['ids'] and len(results['ids']) > 0:
            # Extract results for the first (and only) query
            ids = results['ids'][0]
            documents = results['documents'][0]
            distances = results['distances'][0]

            # Convert distances to similarity scores (cosine similarity = 1 - cosine distance)
            for i in range(len(ids)):
                similarity = 1 - distances[i]
                output.append({
                    'id': ids[i],
                    'text': documents[i],
                    'similarity': similarity
                })

        return output

    def delete_document(self, doc_id: str) -> None:

        self.collection.delete(ids=[doc_id])

    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document exists in hot storage by its ID.

        Args:
            doc_id: ID of the document to check.

        Returns:
            True if document exists, False otherwise.
        """
        results = self.collection.get(ids=[doc_id], include=[])
        return len(results['ids']) > 0