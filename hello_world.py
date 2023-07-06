import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from loader import load

def chunk_text(text, chunk_size):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def embed_text(text):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(text)
    return embeddings

# User-provided documents
documents = [
    "test/animals.txt",
    "test/vegetables.txt"
]

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
# User query
query = input("Enter your query: ")

document_embeddings = load(documents, model)
query_embedding = model.encode(query)

# # Calculate cosine similarity and find the most similar document
similarity_scores = [cosine_similarity(query_embedding, doc_embedding)[0][0] for doc_embedding in document_embeddings]
most_similar_doc_index = np.argmax(similarity_scores)
most_similar_doc = documents[most_similar_doc_index]

print("Most similar document:")
print(most_similar_doc)
