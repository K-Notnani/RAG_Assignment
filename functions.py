import os
import pypdf
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cohere

def load_and_chunk_document(file_path_or_file, chunk_size=300, overlap=50):
    text = ""
    if isinstance(file_path_or_file, str):
        file_name = file_path_or_file
        if file_name.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(file_path_or_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        else:
            with open(file_path_or_file, "r", encoding="utf-8") as f:
                text = f.read()
    else:
        file_name = file_path_or_file.name
        if file_name.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(file_path_or_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        else:
            text = file_path_or_file.getvalue().decode("utf-8")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=False)
    return embeddings.tolist()

def search_chunks(query, chunks, embeddings, k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k_indices = np.argsort(similarities)[::-1][:k]
    top_chunks = [chunks[i] for i in top_k_indices]
    return top_chunks

def generate_answer(query, context, api_key):
    co = cohere.ClientV2(api_key=api_key)
    context_str = "\n---\n".join(context) if isinstance(context, list) else context
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the information provided in the context.
Context:
{context_str}
Question:
{query}
Answer:
"""
    response = co.chat(
        model="command-a-03-2025",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.message.content[0].text