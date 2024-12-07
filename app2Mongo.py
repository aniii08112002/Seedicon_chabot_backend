from flask import Flask, render_template, request, redirect, session
import os
from dotenv import load_dotenv
import PyPDF2
import numpy as np
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from pymongo import MongoClient
from bson import ObjectId

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGO_URI")

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session handling

# Configure Flask app
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Initialize MongoDB Client
client = MongoClient("mongodb+srv://anirudhrandw:kn3Y2Nuvf9LNj5Z2@vectordb.tjc9z.mongodb.net/?retryWrites=true&w=majority&appName=vectordb")
db = client['test1']
collection = db['testing2']

# Initialize LangChain components
llm = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory)
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Suggested questions
SUGGESTED_QUESTIONS = [
    "Details of the CEO's of the company",
    "Evaluation of the company",
    "Competitors or the players",
    "Tools and technologies used",
    "Partners",
]

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_document(file_path):
    """Extract text from the uploaded PDF document."""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = "".join(page.extract_text() for page in reader.pages)
    return text

def chunk_text(text, chunk_size=1000):
    """Chunk the document into smaller parts."""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def store_embeddings_in_mongo(text):
    """Embed document text and store it in MongoDB."""
    chunks = chunk_text(text)
    embeddings = []
    
    # Add a unique pitchdeck identifier to each document
    pitchdeck_id = str(ObjectId())  # You can generate your own identifier if needed
    
    for chunk in chunks:
        embedding = embedding_model.embed_query(chunk)
        embeddings.append(embedding)
        
        # Store each chunk along with its embedding and pitchdeck_id in MongoDB
        result = collection.insert_one({"text": chunk, "embedding": embedding, "pitchdeck_id": pitchdeck_id})
        
        # Print confirmation in the CLI for each document stored
        if result.inserted_id:
            print(f"Successfully stored embedding for chunk with _id: {result.inserted_id}")
    
    return embeddings

def search_mongo(question):
    """Search MongoDB for the most relevant document context."""
    question_embedding = embedding_model.embed_query(question)
    
    # Find all documents with their embeddings
    documents = collection.find()
    
    # Calculate similarity and choose the most relevant one
    best_match = None
    best_similarity = float('-inf')
    
    for doc in documents:
        doc_embedding = np.array(doc['embedding'])
        similarity = np.dot(question_embedding, doc_embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(doc_embedding))
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = doc['text']
    
    if best_similarity > 0.65:
        return best_match
    else:
        return "No relevant information found in the document."

# Flask routes
@app.route('/')
def home():
    """Render the home page."""
    session['asked_questions'] = []  # Reset session for questions
    return render_template('index.html', suggested_questions=SUGGESTED_QUESTIONS)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and process the document."""
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.", suggested_questions=SUGGESTED_QUESTIONS)

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type.", suggested_questions=SUGGESTED_QUESTIONS)

    # Delete previous pitch deck data before processing the new one
    collection.delete_many({})  # Deletes all documents in the collection. Adjust if you need to filter by pitch deck ID or other conditions.

    # Save the file and process its content
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(file_path)
    text = process_document(file_path)

    # Store document embeddings in MongoDB
    store_embeddings_in_mongo(text)

    # Generate a summary of the document (using few-shot prompting)
    prompt = f"Summarize the following document with examples of the kind of information to extract:\n\n{text}"
    summary = conversation.predict(input=prompt)

    return render_template('index.html', summary=summary, suggested_questions=SUGGESTED_QUESTIONS)

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle user questions and provide responses."""
    question = request.form.get('question')
    if not question:
        return render_template('index.html', error="No question provided.", suggested_questions=SUGGESTED_QUESTIONS)

    try:
        # Search for relevant context in MongoDB
        context = search_mongo(question)
        if context == "No relevant information found in the document.":
            response = "This information is not available in the uploaded document."
        else:
            response = conversation.predict(input=f"Context: {context}\n\nQuestion: {question}")

        # Update asked questions and suggested questions
        asked_questions = session.get('asked_questions', [])
        asked_questions.append(question)
        session['asked_questions'] = asked_questions
        updated_suggestions = [q for q in SUGGESTED_QUESTIONS if q not in asked_questions]

        return render_template('index.html', chat_response=response, suggested_questions=updated_suggestions)
    except Exception as e:
        print("Error during conversation:", e)
        return render_template('index.html', error="Error processing your request.", suggested_questions=SUGGESTED_QUESTIONS)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
