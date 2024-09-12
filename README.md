# RAG-based Question Answering Bot

This project demonstrates a RAG (Retrieval-Augmented Generation) system that answers questions based on information from a provided document (`GST_Smart_Guide.docx`). The system uses OpenAI's API for generating embeddings and answering questions, FAISS for fast retrieval, and DSPy for pipeline optimization.

## Prerequisites

1. Python 3.8 or above.
2. An OpenAI API key.
3. FAISS (for similarity search).
4. DSPy (for pipeline optimization).

## Project Structure

```bash
├── indexing.py               # Script to extract text from .docx and create FAISS index.
├── rag_dspy.py               # Script to use DSPy for the RAG system.
├── GST_Smart_Guide.docx      # Input document for indexing and QA.
├── requirements.txt          # Python dependencies.
├── .env                      # Environment variables for OpenAI API key.
└── README.md                 # This README file.
```

### Step 1: Set up the Virtual Environment

First, create and activate a virtual environment to manage the project dependencies.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### Step 2: Install Required Packages

Run the following command to install all required Python packages.

```bash
pip install -r requirements.txt
```

### Step 3: Setup OpenAI API Key

Create a `.env` file in the project root:

```bash
touch .env
```
Add your OpenAI API key to the `.env` file:

```makefile
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 4: Run FAISS Indexing
Before running the question-answer bot, you need to index the document with FAISS.

1. Place your `.docx` file (e.g., "GST_Smart_Guide.docx") in the project directory.

2. Run the indexing script to generate the document index:

```bash
python indexing.py
```
This script will:

- Extract text from the document.
- Generate embeddings using OpenAI's API.
- Create a FAISS index for fast retrieval.

### Step 5: Run the RAG Bot with DSPy

After indexing, you can run the RAG question-answering bot optimized using DSPy:

```bash
python rag_dspy.py
```
This script will:
- Load the FAISS index and the document embeddings.
- Retrieve relevant paragraphs using FAISS.
- Use OpenAI's GPT-3.5-turbo to answer the question based on the retrieved context.
- Optimize the question-answering pipeline using DSPy.

### Step 6: Example Query

The `rag_dspy.py` script contains a sample query. You can modify the question in the script to ask different queries.

```python
question = "What is the reverse charge mechanism in GST?"
```
The bot will retrieve relevant paragraphs from the document and generate an answer using GPT-3.5. You can replace this question with your own to explore other queries based on the indexed document.

### Project Structure Details

- **indexing.py**: Extracts text from the document, generates embeddings, and creates a FAISS index for efficient retrieval.
- **rag_dspy.py**: Implements the Retrieval-Augmented Generation (RAG) model using DSPy and OpenAI for answering queries based on document context.
- **.env**: Stores the OpenAI API key for authentication and accessing the OpenAI API.
- **requirements.txt**: Lists the necessary Python dependencies required to run the project, such as `openai`, `faiss`, `dspy`, and other essential libraries.

### Notes
- **Ensure** you have a valid OpenAI API key for generating embeddings and answering questions.
- **Modify** the question in `rag_dspy.py` to test with different queries for flexible experimentation.
- **Adjust** the number of paragraphs processed in `indexing.py` for better performance based on system capabilities and document size.