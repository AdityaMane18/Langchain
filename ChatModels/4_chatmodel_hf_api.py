from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Explicitly load token to ensure it's available
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") or os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_token  # Parameter name is api_token, not access_token
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print(result.content)
