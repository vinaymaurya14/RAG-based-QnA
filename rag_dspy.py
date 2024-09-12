import os
import faiss
import numpy as np
import openai
import dspy
from dotenv import load_dotenv
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import answer_exact_match, answer_passage_match

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
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message["content"]

# Define the GenerateAnswer signature
class GenerateAnswer(dspy.Signature):
    """Answer questions based on the provided context."""
    
    context = dspy.InputField(desc="Relevant facts from the document")
    question = dspy.InputField(desc="User's query")
    answer = dspy.OutputField(desc="Short answer to the user's question")
    
# Define the RAG model using DSPy
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# Sample training set for DSPy optimization
trainset = [{"question": "What is GST?", "answer": "Goods and Services Tax"}]

# Validation logic for DSPy compilation
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = answer_exact_match(example, pred)
    answer_PM = answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Setup DSPy teleprompter and compile
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

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
