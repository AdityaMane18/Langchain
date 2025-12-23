from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema #Note the StructuredOutputParser and Response schema are depracated 
import os

load_dotenv()


# Explicitly load token to ensure it's available
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") or os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(repo_id = "google/gemma-2-2b-it",
 task = "text-generation",
 huggingfacehub_api_token=hf_token  # Parameter name is api_token, not access_token
 )


model = ChatHuggingFace(llm = llm)
schema = [ResponseSchema(name='fact_1', description = 'Fact 1 about the topic'),
ResponseSchema(name='fact_2', description = 'Fact 2 about the topic'),
ResponseSchema(name='fact_3', description = 'Fact 3 about the topic')]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give 3 facts about {topic} \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'topic': 'black hole'})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)


# Using chain
chain = template | model | parser
result = chain.invoke({'topic': 'black hole'})

# Result is already parsed by the parser in the chain
print(result)