from chunkipy import TextChunker, TokenEstimator
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
import faiss
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel

from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

def read_file(filename):
    with open (filename, 'r') as f:
        file_content = f.read()
    return file_content

scrapped_text = 'scraped_text1.txt'

text = read_file(scrapped_text)

class RobertaTokenEstimator(TokenEstimator):
    def __init__(self):
        self.robert_tokenizer = tokenizer

    def estimate_tokens(self, text):
        return len(self.robert_tokenizer.encode(text))

robert_token_estimator = RobertaTokenEstimator()

print(f"Num of chars: {len(text)}")
# --> Num of chars: 3149
print(f"Num of tokens (using RoBertTokenEstimator): {robert_token_estimator.estimate_tokens(text)}")


# This creates an instance of the TextChunker class with a chunk size of 512, 
# using token-based segmentation and a custom tokenizer function bert_tokenizer.encode to count tokens.


text_chunker = TextChunker(128, tokens=True, token_estimator=RobertaTokenEstimator())
chunks = text_chunker.chunk(text)
#chunks_ = [chunk.split() for chunk in chunks]
print(type(chunks))
#print('\n,the chunks : __',chunks_)
# model = Word2Vec(chunks, vector_size=100, window=5, min_count=1, workers=4)
# # for i, chunk in enumerate(chunks):
# #     print(f"Chunk {i + 1}, num of tokens: {bert_token_estimator.estimate_tokens(chunk)} -> {chunk}")

# def get_chunk_vector(chunk, model):
#     #words = chunk.split()  # Split the chunk into individual words
#     embeddings = [model.wv[word] for word in chunk if word in model.wv]
#     if embeddings:
#         return sum(embeddings) / len(embeddings)  # Average the embeddings
#     else:
#         return None  # Handle the case when none of the words are in the vocabulary

# # Example: Building an index
# vectors = np.array([get_chunk_vector(chunk, model) for chunk in chunks_ if get_chunk_vector(chunk, model) is not None])
# index = faiss.IndexFlatL2(model.vector_size)
# index.add(vectors)

# # Example: Performing a similarity search with a query vector
# query_vector = get_chunk_vector("what are the company's job requirements from a candidate ?", model)
# if query_vector is not None:
#     _, similar_indices = index.search(np.array([query_vector]), k=5)
#     similar_chunks = [chunks[i] for i in similar_indices[0]]
#    print("\n the similar chunks are ------",similar_chunks)


################  craete embeddings adn perform similarity search ###############
###############  RobertaTokenizer  ###############


def create_embeddings(chunks):
    all_embeddings = []
    for chunk in chunks:
        input_ids = tokenizer.encode(chunk, add_special_tokens=True)
        with torch.no_grad():
            last_hidden_states = model(torch.tensor([input_ids]))[0]
        embeddings = last_hidden_states[0, :, :].numpy()
        all_embeddings.append(embeddings)
    return all_embeddings


def similarity_search(embeddings, query_embedding, k=10):
    distances = [cosine(query_embedding, embedding) for embedding in embeddings]
    indices = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
    return distances, indices

# assume chunks is a list of strings, each string representing a chunk of words
embeddings = create_embeddings(chunks)

# create a query and its embedding
query = "what are the job requirements and skills the above company is looking for in the candidate to join this company ?"
query_embedding = create_embeddings([query])[0]

# perform similarity search
distances, indices = similarity_search(embeddings, query_embedding)

# print the most similar chunks
print("Most similar chunks:")
for i in indices[:5]:
    print(f"Chunk no.{i} ---------------------------",f"the distance ------{distances[i]}",chunks[i],'\n')