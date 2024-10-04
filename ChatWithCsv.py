from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_FAISS_PATH = "vectorstore/db_faiss"
loader = CSVLoader(file_path='data/2019.csv', encoding='utf-8', csv_args={'delimiter': ','})
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc_search = FAISS.from_documents(text_chunks, embeddings)
doc_search.save_local(DB_FAISS_PATH)

query = "What is the value of GDP per capita of Finland provided in the data?"

docs = doc_search.similarity_search(query, k=3)

print("Result", docs)