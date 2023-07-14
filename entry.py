from sentence_transformers import SentenceTransformer
from loader import load
from analyzer import analyze
from glob import glob
from os.path import isdir, isfile, exists

MODEL = SentenceTransformer('distilbert-base-nli-mean-tokens')

def entry():
    document_directory = input("enter directory path of text corpus: \n")
    print(isdir(document_directory), isfile(document_directory), exists(document_directory))
    if not isdir(document_directory):
        print("text corpus must be directory: ", document_directory)
        return
    if not document_directory.endswith('/'):
        print("text corpus directory must end with \"/\" :", document_directory)
        return
    documents = glob(document_directory+"/**/*.txt", recursive=True) + glob(document_directory+"/**/*.pdf", recursive=True)
    document_embeddings = load(documents, MODEL)

    query = input("Enter your query: ")

    query_embedding = MODEL.encode(query)

    res = analyze(document_embeddings, query_embedding, 1)
    print(res)

if __name__ == '__main__':
    entry()
    
