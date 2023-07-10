# What is Mino

A personalized semantic search for local files. Currently stores documents in memory.

Mostly developing to play around with latest updates to semantic search technology.

Plans for future work:

1) Offline document parsing.
2) Sophisticated data chunking.
3) Weighted semantic search with sparse vectors (probably generated through BM25).
4) Model fine-tuning, cross-encoder usage for end state re-ranking.
5) Maybe proper vector store/ANN instead of NN once we hit document scale issues.

## To Run:

1) Upload file path of documents into the entry.py file
2) Execute entry.py and input associated query