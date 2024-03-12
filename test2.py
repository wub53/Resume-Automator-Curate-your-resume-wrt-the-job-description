from openai import OpenAI
import os
from scipy.spatial.distance import cosine
from chunkipy import TextChunker, TokenEstimator
from transformers import RobertaTokenizer, RobertaModel


def read_file(filename):
    with open (filename, 'r') as f:
        file_content = f.read()
    return file_content

def similarity_search(embeddings, query_embedding, k=10):
    distances = [cosine(query_embedding, embedding) for embedding in embeddings]
    indices = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
    return distances, indices

scrapped_text = 'scraped_text.txt'

text = read_file(scrapped_text)

client = OpenAI()

response1 = client.embeddings.create(
    input=text,
    model="text-embedding-3-small"
)

context_embeddings = response1.data[0].embedding

response2 = client.embeddings.create(
    input = 'what are the job requirements and skills the above company is looking for in the candidate to join this company ?',
    model = 'text-embedding-3-small'
)
#print(response1.data[0].embedding)

query_embeddings = response2.data[0].embedding

# perform similarity search
distances, indices = similarity_search(context_embeddings, query_embeddings)

# print the most similar chunks
print("Most similar chunks:")
for i in indices[:5]:
    print(f"Chunk no.{i} ---------------------------",f"the distance ------{distances[i]}",chunks[i],'\n')


