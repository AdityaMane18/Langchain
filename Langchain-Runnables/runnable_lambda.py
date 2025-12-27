from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel
import os
load_dotenv()


# Explicitly load token to ensure it's available
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") or os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(repo_id = "google/gemma-2-2b-it",
 task = "text-generation",
 huggingfacehub_api_token=hf_token  # Parameter name is api_token, not access_token
 )

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)