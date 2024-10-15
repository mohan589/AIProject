from dotenv import load_dotenv
import os

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from github import fetch_github_issues
from note import note_tool
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from transformers import LlamaTokenizer, LlamaForCausalLM

load_dotenv()

PINECONE_API_KEY="5f00d642-14fd-4fd3-acb4-60f9976000ea"

# Create a serverless index
# "dimension" needs to match the dimensions of the vectors you upsert
pc = Pinecone(api_key="5f00d642-14fd-4fd3-acb4-60f9976000ea")

# pc.create_index(name="github", dimension=1536, 
#   spec=ServerlessSpec(cloud='aws', region='us-east-1') 
# )

# Target the index
index = pc.Index("github")

# Mock vector and metadata objects (you would bring your own)
# vector = [0.010, 2.34,...] # len(vector) = 1536
# metadata = {"id": 3056, "description": "Networked neural adapter"}

# Upsert your vector(s)
# index.upsert(
#   vectors=[
#     {"id": "some_id", "values": vector, "metadata": metadata}
#   ]
# ) 

def connect_to_vstore():
  embeddings = HuggingFaceBgeEmbeddings()

  docsearch = PineconeVectorStore(index=index, pinecone_api_key=PINECONE_API_KEY, embedding=embeddings)#.from_texts([t.page_content for t in docs], embeddings, index_name=index)

  return docsearch

vstore = connect_to_vstore()
add_to_vectorstore = input("Do you want to update the issues? (y/N): ").lower() in [
  "yes",
  "y",
]

if add_to_vectorstore:
  owner = "techwithtim"
  repo = "Flask-Web-App-Tutorial"
  issues = fetch_github_issues(owner, repo)

  try:
      vstore.delete_collection()
  except:
      pass

  vstore = connect_to_vstore()
  vstore.add_documents(issues)

  # results = vstore.similarity_search("flash messages", k=3)
  # for res in results:
  #     print(f"* {res.page_content} {res.metadata}")

retriever = vstore.as_retriever(search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
  retriever,
  "github_search",
  "Search for information about github issues. For any questions about github issues, you must use this tool!",
)

# Define how to use the model to generate responses
def llama_generate(prompt):
  inputs = tokenizer(prompt, return_tensors="pt")
  outputs = model.generate(**inputs)
  return tokenizer.decode(outputs[0], skip_special_tokens=True)

model_name = "/Users/mpichikala/personal/llama-2"  # You would replace this with the actual LLaMA model path
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)


prompt = "Can you search for github issues?"
# llm = ChatOpenAI()

# Example tool functions
tools = {
  "search": "github issues"
}
result = create_tool_calling_agent(llama_generate, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

while (question := input("Ask a question about github issues (q to quit): ")) != "q":
  result = agent_executor.invoke({"input": question})
  print(result["output"])