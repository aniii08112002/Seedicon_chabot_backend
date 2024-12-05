from flask import Flask, render_template, request, redirect
import os
from dotenv import load_dotenv
import PyPDF2
import numpy as np
import faiss
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Configure Flask app
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Initialize LangChain components
llm = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory)
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize Faiss index (1536-dimensional embeddings)
faiss_index = faiss.IndexFlatL2(1536)
document_embeddings = {}  # Store document text with corresponding embeddings

# Utility functions
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_document(file_path):
    """Extract text from the uploaded PDF file."""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def reset_faiss_and_embeddings():
    """Clear FAISS index and document embeddings."""
    global faiss_index, document_embeddings
    faiss_index = faiss.IndexFlatL2(1536)
    document_embeddings = {}

def store_embeddings_in_faiss(text, faiss_index):
    """Generate and store embeddings for the document text in Faiss."""
    embedding = embedding_model.embed_query(text)
    faiss_index.add(np.array([embedding], dtype="float32"))
    return embedding

def search_faiss(question, faiss_index):
    """Query the Faiss index for the most relevant stored embeddings."""
    question_embedding = embedding_model.embed_query(question)
    distances, indices = faiss_index.search(np.array([question_embedding], dtype="float32"), k=1)
    if distances[0][0] < 0.65:  # Set a strict threshold for relevance
        doc_text = document_embeddings.get(indices[0][0], "Relevant document text not found.")
        return doc_text
    else:
        return "No relevant information found in the document."

# Flask routes
@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads, process documents, and store embeddings."""
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    # Reset Faiss index and document embeddings
    reset_faiss_and_embeddings()

    # Save the file and process it
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(file_path)
    text = process_document(file_path)

    # Store embeddings in Faiss and save document text for retrieval
    embedding = store_embeddings_in_faiss(text, faiss_index)
    document_embeddings[faiss_index.ntotal - 1] = text

    # Generate summary of the document
    summary = conversation.predict(input=f"Summarize this document: {text}")

    return render_template('index.html', summary=summary)

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle user queries and provide chatbot responses."""
    question = request.form.get('question')
    if not question:
        return render_template('index.html', error="No question provided.")

    try:
        # Search Faiss for relevant document context
        context = search_faiss(question, faiss_index)

        # Ensure context is valid
        if context == "No relevant information found in the document.":
            return render_template('index.html', chat_response="This information is not available in the uploaded document.")

        # Generate chatbot response based on document context
        chat_response = conversation.predict(input=f"Context: {context}\n\nQuestion: {question}")

        return render_template('index.html', chat_response=chat_response)
    except Exception as e:
        print("Error during conversation:", e)
        return render_template('index.html', error="Error processing your request.")

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
