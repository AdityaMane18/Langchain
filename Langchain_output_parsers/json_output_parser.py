from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()


# Explicitly load token to ensure it's available
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") or os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(repo_id = "google/gemma-2-2b-it",
 task = "text-generation",
 huggingfacehub_api_token=hf_token  # Parameter name is api_token, not access_token
 )


model = ChatHuggingFace(llm = llm)
parser = JsonOutputParser()


template = PromptTemplate(template = 'Give me the name, age and city of a fictional person \n {format_instruction}',
input_variables = [],
partial_variables = {'format_instruction': parser.get_format_instructions()})

# prompt = template.format()

# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

#Using chains
chain = template | model | parser
result = chain.invoke({})


print(result)