from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline #Here we will use a local model
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.3, max_new_tokens = 10000)
)
model = ChatHuggingFace(llm=llm)

while True:
    user_input = input('You: ')
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    print("AI: ", result.content)