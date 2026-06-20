from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.cold_storage import ColdStorage
import os


import re

def clean_markdown(text):
    text = re.sub(r'#{1,6}\s*', '', text)        # remove headers
    text = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', text)  # remove bold/italic
    text = re.sub(r'\|.*?\|', '', text)           # remove table cells
    text = re.sub(r'[-]{3,}', '', text)           # remove horizontal rules
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # remove links keep text
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text)  # remove code blocks
    text = re.sub(r'\s+', ' ', text)              # normalize whitespace
    return text.strip()
parser = LlamaParse(
    api_key="llx-0LUs7ABRy9FA5Y7Q7OjAMBNNAihcDsdRDVEfdl4Q4mxHCR4x",
    result_type="markdown",
    verbose=True
)

all_docs = []
for filename in os.listdir("./documents"):
    if filename.endswith(".pdf"):
        filepath = f"./documents/{filename}"
        print(f"Parsing {filename}...")
        docs = parser.load_data(filepath)
        all_docs.extend(docs)

print(f"Total docs loaded: {len(all_docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

cold_storage = ColdStorage(
    host="localhost",
    user="root",
    password="root",
    database="cold_storage"
)

chunks = splitter.create_documents([doc.text for doc in all_docs])

print(f"Total chunks: {len(chunks)}")

for chunk in chunks:
    cleaned = clean_markdown(chunk.page_content)
    if cleaned:
        cold_storage.add_document(cleaned)

print(f"Done. {len(chunks)} chunks inserted.")
cold_storage.close()