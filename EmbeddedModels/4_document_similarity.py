from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()


embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)
document = ["MS Dhoni is a cricketer",
            "Virat Kohli is a cricketer",
            "Rohit Sharma is a cricketer",
            "Sachin Tendulkar is a cricketer",
            "Yuvraj Singh is a cricketer",
            "Suresh Raina is a cricketer"]
query = "Who is Virat Kohli?"

document_embeddings = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)[0]

print(scores)
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]
print(f"The most similar document is {document[index]} with a score of {score}")


