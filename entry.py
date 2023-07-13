from sentence_transformers import SentenceTransformer
from loader import load
from analyzer import analyze

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
    "test/vegetables.txt",
    "test/test_pdf.pdf",
]

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
# User query
query = input("Enter your query: ")

document_embeddings = load(documents, model)
query_embedding = model.encode(query)

res = analyze(document_embeddings, query_embedding, 1)
print(res)
