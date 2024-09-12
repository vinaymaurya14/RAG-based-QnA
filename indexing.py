import os
import openai
import docx
import faiss
import numpy as np
from dotenv import load_dotenv

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from .docx file
def extract_text_from_docx(docx_file):
    print("Extracting text from the document...")
    doc = docx.Document(docx_file)
    paragraphs = [para.text for para in doc.paragraphs if para.text]
    print(f"Extracted {len(paragraphs)} paragraphs.")
    return paragraphs

# Function to get embeddings from OpenAI
def get_embedding(text):
    print(f"Generating embedding for text: {text[:50]}...")
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    print("Embedding generated.")
    return np.array(response['data'][0]['embedding'])

# Load document and get text
docx_file = "GST_Smart_Guide.docx"
paragraphs = extract_text_from_docx(docx_file)

# Limit the number of paragraphs for testing
paragraphs = paragraphs[:100]  

# Create FAISS index
embedding_dim = 1536  # ADA model output size
index = faiss.IndexFlatL2(embedding_dim)

# Store embeddings with progress tracking
paragraph_embeddings = []
for i, para in enumerate(paragraphs):
    print(f"Processing paragraph {i+1}/{len(paragraphs)}")
    paragraph_embeddings.append(get_embedding(para))

# Add embeddings to the FAISS index
index.add(np.array(paragraph_embeddings))

# Save index and paragraphs
faiss.write_index(index, "document_index.faiss")
np.save("paragraphs.npy", paragraphs)

print("FAISS index and document embeddings created successfully!")
