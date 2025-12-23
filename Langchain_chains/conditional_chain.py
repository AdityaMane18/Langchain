from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough
import os
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()


# Explicitly load token to ensure it's available
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") or os.getenv("HF_TOKEN")

llm1 = HuggingFaceEndpoint(repo_id = "google/gemma-2-2b-it",
 task = "text-generation",
 huggingfacehub_api_token=hf_token  # Parameter name is api_token, not access_token
 )

model = ChatHuggingFace(llm = llm1)
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description = 'Give the sentiment of the feedback')

parser2  =PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template = '''Classify the sentiment of the following feedback text into positive or negative.

Feedback: {feedback}

You must respond with valid JSON matching this exact format:
{format_instruction}

Return ONLY the JSON object, nothing else. Example: {{"sentiment": "negative"}}''',
    input_variables = ['feedback'],
    partial_variables = {'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2
# print(classifier_chain.invoke({'feedback':'This is a terrible smartphone'}))
# result = classifier_chain.invoke({'feedback':'This is a terrible smartphone'}).sentiment
# print(result)






prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# Preserve original input and add sentiment classification
# This ensures branch chain has access to both sentiment and original feedback text
classifier_with_input = RunnablePassthrough.assign(
    sentiment_obj=classifier_chain
)

# Branch chain that routes based on sentiment
# Note: The branch receives a dict with 'feedback' and 'sentiment_obj' keys
branch_chain = RunnableBranch(
    (lambda x: x['sentiment_obj'].sentiment == 'positive' if isinstance(x, dict) and 'sentiment_obj' in x else False, 
     prompt2 | model | parser),
    (lambda x: x['sentiment_obj'].sentiment == 'negative' if isinstance(x, dict) and 'sentiment_obj' in x else False, 
     prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_with_input | branch_chain
print(chain.invoke({'feedback': 'This is a terrible phone'}))