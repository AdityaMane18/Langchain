from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


ChatModel = ChatOpenAI(model="gpt-4", temperature = 0.3)


result = ChatModel.invoke("What is the capital of India?")
print(result)
