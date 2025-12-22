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


st.header('Research Tool')
user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)