from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from cold_memory import ColdStorage

loader = DirectoryLoader(
    "./documents",      
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

storage = ColdStorage(
        host="localhost",
        user="root",
        password="root",
        database="cold_storage"
    )

for i, chunk in enumerate(chunks):
    storage.add_document(
        f"chunk_{i}",
        chunk.page_content
    )


query = input("Enter your Query")
results = storage.search_bm25(query, top_k=3)

print(f"\nSearch results for: '{query}'")
print("-" * 50)
for i, result in enumerate(results, 1):
    print(f"{i}. Doc ID: {result.doc_id}")
    print(f"   Score: {result.score:.4f}")
    print(f"   Text: {result.text}")
    print()
storage.delete_table()
storage.close()