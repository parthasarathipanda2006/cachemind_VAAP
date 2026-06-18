from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from cold_memory import ColdStorage
from hot_storage import HotStorage

cold_storage = ColdStorage(
    host="localhost",
    user="root",
    password="root",
    database="cold_storage"
)


loader = DirectoryLoader(
    "./documents",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40
)

chunks = splitter.split_documents(documents)


for chunk in chunks:
    doc_id_int = cold_storage.add_document(chunk.page_content)

cold_storage.close()