from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()


ChatModel = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature = 0.3)

result = ChatModel.invoke("What is the capital of India?")
print(result.content)