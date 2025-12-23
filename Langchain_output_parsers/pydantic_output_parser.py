from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()


# Explicitly load token to ensure it's available
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") or os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(repo_id = "google/gemma-2-2b-it",
 task = "text-generation",
 huggingfacehub_api_token=hf_token  # Parameter name is api_token, not access_token
 )


model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name: str = Field(description = 'Name of the person')
    age: int = Field(gt = 18, description = 'Age of the person')
    city: str = Field(description = 'Name of the city the person belongs to')
parser = PydanticOutputParser(pydantic_object = Person)

template  = PromptTemplate(
    template = 'Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables = ['place'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'place': 'Indian'})
# result = model.invoke(prompt)


# final_result = parser.parse(result.content)
# print(final_result)





#Using chain
chain  = template | model | parser
final_result = chain.invoke({'place' : 'UAE'})
print(final_result)