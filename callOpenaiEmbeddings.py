from openai import OpenAI
import os
import torch
import numpy as np
import openai 
from dotenv import load_dotenv
from scipy.spatial.distance import cosine

# used export to set the open ai api_key as environment variable 
client = OpenAI()

# call the openai api endpoint per chunk out of all the chunks to get embeddings per chunk

def create_context_embeddings(chunks):

    context_embeddings = torch.empty(len(chunks),1536)

    for i, chunk in enumerate(chunks) :

        response = client.embeddings.create(
            input = chunk,
            model = "text-embedding-ada-002"
            )
        embedding_array = np.array(response.data[0].embedding)

        torch_tensor = torch.from_numpy(embedding_array)

        context_embeddings[i] = torch_tensor
    
    return context_embeddings

def create_query_embeddings():
    
    response = client.embeddings.create(
    input="what are the job requirements and skills the above company is looking for in the candidate to join this company ?",
    model="text-embedding-ada-002"
  )

    query_embeddings_array = np.array(response.data[0].embedding)

    query_embeddings = torch.from_numpy(query_embeddings_array)

    return query_embeddings

    #print(query_embeddings.shape)



def similarity_search(context_embeddings, query_embedding, k=5):
    #distances = [cosine(query_embedding, embedding) for embedding in context_embeddings]
    distances = []
    for i in range(0,len(context_embeddings)):
      distances.append(((cosine(query_embedding,context_embeddings[i])),i))
    sorted_distances = sorted(distances, key=lambda x: x[0])[:k]
    #distances = sorted(distances)
    print('\n',sorted_distances,'\n')

    # for i in indices:
    #   sentence = [tokenizer.decode(input_id) for input_id in context_sentences_input_ids[i]]
    #   print(f"chunk{i}  ----->",sentence,'\n')
    return sorted_distances