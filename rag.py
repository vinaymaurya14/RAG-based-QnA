import os
import faiss
import numpy as np
import openai
from dotenv import load_dotenv

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS index and document embeddings
index = faiss.read_index("document_index.faiss")
paragraphs = np.load("paragraphs.npy", allow_pickle=True)

# Function to find top k matching paragraphs using FAISS
def retrieve_paragraphs(query_embedding, k=3):
    D, I = index.search(np.array([query_embedding]), k)
    return [paragraphs[i] for i in I[0]]

# Function to get embedding of the question
def get_query_embedding(query):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=query)
    return np.array(response['data'][0]['embedding'])

# Function to generate answers using GPT-3.5
def generate_answer(question, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

# Sample question
question = "What is the reverse charge mechanism in GST?"

# Get query embedding and retrieve relevant paragraphs
query_embedding = get_query_embedding(question)
retrieved_paragraphs = retrieve_paragraphs(query_embedding)
context = " ".join(retrieved_paragraphs)

# Generate answer
answer = generate_answer(question, context)
print(f"Question: {question}")
print(f"Answer: {answer}")
