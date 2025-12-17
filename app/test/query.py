import os
import faiss
import pickle
import numpy as np
import random

#function for fake_embedding
def fake_embedding(text: str, dim: int = 384):
    random.seed(hash(text) % 1_000_000)
    return [random.random() for _ in range(dim)]


DATA_PATH = "data/sample.txt"
VECTOR_PATH = "vectorstore/faiss.index"
CHUNK_PATH = "vectorstore/chunks.pkl"

CHUNK_SIZE = 300

index = faiss.read_index(VECTOR_PATH)

with open(CHUNK_PATH, "rb") as f:
    chunks = pickle.load(f)
    
def query(question: str, top_k: int = 3):
    
    #convert user input into a vector
    q_embedding = fake_embedding(question)
    q_vector = np.array([q_embedding]).astype("float32")

    #search in faiss
    distances, indices = index.search(q_vector, top_k)

    #print outputs
    print(f"\nQuestion: {question}\n")
    print("Top relevant chunks:\n")

    for i, idx in enumerate(indices[0]):
        print(f"[Result {i+1}]")
        print(chunks[idx])
        print("-" * 40)

if __name__ == "__main__":
    question = "How can I change the transfer limit?"
    query(question)
