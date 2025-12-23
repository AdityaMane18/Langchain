from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()


# Explicitly load token to ensure it's available
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") or os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(repo_id = "google/gemma-2-2b-it",
 task = "text-generation",
 huggingfacehub_api_token=hf_token  # Parameter name is api_token, not access_token
 )



prompt = PromptTemplate(
    template = 'Generate 5 interesting facts about {topic}',
    input_variables = ['topic']
)
model = ChatHuggingFace(llm = llm)
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'topic' : 'cricket'})

print(result)
