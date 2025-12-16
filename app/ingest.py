# Save Chunk

import os
import faiss
import numpy as np
import pickle
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path = os.path.join(os.path.dirname(__file__),"..",".env"))

client = OpenAI()

#Path variables
DATA_PATH = "data/sample.txt"
VECTOR_PATH = "vectorstore/faiss.index"
CHUNK_PATH = "vectorstore/chunks.pkl"

CHUNK_SIZE = 300

def ingest():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    embeddings = []
    for chunk in chunks:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding
        embeddings.append(emb)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, VECTOR_PATH)

    with open(CHUNK_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("document vectorization + chunk saved completed!")

if __name__ == "__main__":
    ingest()
