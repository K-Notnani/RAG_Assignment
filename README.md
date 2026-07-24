#  RAG Document Question Answering System
Overview

This project is a Retrieval-Augmented Generation (RAG) based Document Question Answering System developed using Python, Streamlit, Sentence Transformers, and Cohere API.
The application allows users to upload PDF or TXT documents, retrieves the most relevant information using semantic search, and generates context-aware answers based on the uploaded document.
---

 Features
- Upload PDF or TXT documents
- Automatic document chunking
- Generate embeddings using Sentence Transformers
- Perform semantic search using cosine similarity
- Retrieve Top-K relevant text chunks
- Generate AI-powered answers using Cohere API
- Interactive Streamlit web interface

 Technologies Used
- Python
- Streamlit
- Sentence Transformers
- LangChain Text Splitter
- Cohere API
- NumPy
- Scikit-learn
- PyPDF

Project Structure
```
RAG_Assignment/
│── app.py
│── functions.py
│── RAG_assignment (2).ipynb
│── .gitignore
│── README.md
```
---

Installation
1. Clone the repository
```bash
git clone https://github.com/K-Notnani/RAG_Assignment.git
```
 2. Navigate to the project folder
```bash
cd RAG_Assignment
```
3. Install the required packages
```bash
pip install -r requirements.txt
```
4. Run the application
```bash
streamlit run app.py
```

 How It Works

1. Upload a PDF or TXT document.
2. The document is split into smaller text chunks.
3. Sentence Transformers generate embeddings for each chunk.
4. The user enters a question.
5. Cosine similarity retrieves the most relevant chunks.
6. Cohere API generates an answer based on the retrieved context.

Application Workflow
- Upload a document
- Enter the Cohere API key
- Ask a question
- View the retrieved chunks
- Receive an AI-generated answer

Author
**Kanak Notnani**

This project was developed for educational and learning purposes.
