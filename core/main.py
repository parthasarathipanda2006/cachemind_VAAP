from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.cold_storage import ColdStorage
from core.hot_storage import HotStorage

cold_storage = ColdStorage(
    host="localhost",
    user="root",
    password="root",
    database="cold_storage"
)


loader = DirectoryLoader(
    "./documents",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)


for chunk in chunks:
    doc_id_int = cold_storage.add_document(chunk.page_content)

cold_storage.close()