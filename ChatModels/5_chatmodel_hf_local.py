from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

# HuggingFace will automatically use HUGGINGFACEHUB_ACCESS_TOKEN or HF_TOKEN from environment
# Add HUGGINGFACEHUB_ACCESS_TOKEN or HF_TOKEN to your .env file for gated models

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.3, max_new_tokens=10)
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India and what is the GDP of India?")
print(result.content)