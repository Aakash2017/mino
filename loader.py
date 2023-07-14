from typing import List, Dict
import os
import pandas as pd
import PyPDF2
import time

def load(files: List, model) -> Dict:
    '''load chunks documents to user-configured chunks and
    calculates the embeddings for each chunk. Returns a list of
    embeddings per document.'''
    start_time = time.time()
    embedded_docs = {}
    for file_path in files:
        ext = os.path.splitext(file_path)[-1].lower()
        match ext:
            case ".txt":
                with open(file_path, 'r', encoding="utf8", errors='ignore') as file:
                    for i, chunk in enumerate(read_chunks(file)):
                        embedded_docs[(file_path, str(i))] = model.encode(chunk)
            case ".pdf":
                # Currently we only chunk PDF's by page.
                file = open(file_path, 'rb')
                reader = PyPDF2.PdfReader(file)
                for i in range(len(reader.pages)):
                    embedded_docs[(file_path, str(i))] = model.encode(reader.pages[i].extract_text())


    df = pd.DataFrame(embedded_docs.items(), columns=['documentID', 'embedding'])
    print("finished indexing corpus in " + str(time.time()-start_time) + " seconds")               
    return df

def read_chunks(file, chunk_size=1024):
     """Generator to read a file piece by piece.
     Default chunk size: 1k."""
     while True:
        text = file.read(chunk_size)
        if not text:
            break
        yield text