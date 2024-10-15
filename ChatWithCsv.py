import sys
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ctransformers import CTransformers
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

# query = "What is the value of GDP per capita of Finland provided in the data?"
#
# docs = doc_search.similarity_search(query, k=3)
#
# print("Result", docs)

# Replace the below model with large size model
llm = CTransformers(model="/Users/mpichikala/personal/Llama-2-7B-Chat-GGML",
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.1)

qa = ConversationalRetrievalChain.from_llm(llm, retriever=doc_search.as_retriever())

while True:
    chat_history = []
    #query = "What is the value of  GDP per capita of Finland provided in the data?"
    query = input("Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa({"question":query, "chat_history":chat_history})
    print("Response: ", result['answer'])