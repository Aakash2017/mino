from typing import List, Dict
import os

def load(files: List, model) -> Dict:
    '''load chunks documents to user-configured chunks and
    calculates the embeddings for each chunk. Returns a list of
    embeddings per document.'''
    embedded_docs = []
    for file_path in files:
        with open(file_path, 'r') as file:
            embedded_docs.extend([model.encode(chunk) for chunk in read_chunks(file)])
                 

    return embedded_docs

def read_chunks(file, chunk_size=1024):
     """Generator to read a file piece by piece.
     Default chunk size: 1k."""
     while True:
        text = file.read(chunk_size)
        if not text:
            break
        yield text