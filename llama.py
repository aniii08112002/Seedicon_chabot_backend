import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pymongo
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from transformers import pipeline
from bson.objectid import ObjectId
import numpy as np
from scipy.spatial.distance import cosine

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = pymongo.MongoClient(mongo_uri)
db = client["pdf_database"]  # Database for storing the chunks and embeddings
collection = db["pdf_chunks"]  # Collection to store PDF chunks and embeddings

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create embeddings and store in MongoDB
def store_in_mongodb(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # You can change this model
    embeddings_list = embeddings.embed_documents(text_chunks)  # Use embed_documents to get embeddings for all chunks
    
    for chunk, embedding in zip(text_chunks, embeddings_list):
        document = {"text": chunk, "embedding": embedding}
        collection.insert_one(document)  # Insert each chunk with its embedding into MongoDB

# Function to create conversational chain using HuggingFace pipeline
def get_conversational_chain():
    # Set up HuggingFace pipeline (you can use a model like GPT or other available models)
    model = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")  # Example model
    llm = HuggingFacePipeline(pipeline=model)  # Wrap the model pipeline in a LangChain object

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    return chain

# Function to calculate cosine similarity between two vectors
def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Function to process user query and generate response
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Embedding model
    
    # Retrieve all documents from MongoDB
    documents = collection.find()
    
    # Embed the user query
    query_embedding = embeddings.embed_query(user_question)  # Embed the user's query
    
    # Perform similarity search by comparing embeddings
    docs = []
    for doc in documents:
        embedding = doc["embedding"]
        similarity = cosine_similarity(query_embedding, embedding)
        if similarity > 0.7:  # Threshold to find relevant documents
            docs.append(doc)
    
    # Generate a response using the conversational chain
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Display the response
    st.write("Reply: ", response["output_text"])

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with PDF using LLaMA 💁")

    # Input box for user query
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        
        # PDF file uploader
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        # Process button to process PDF documents
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Extract text from the uploaded PDF documents
                raw_text = get_pdf_text(pdf_docs)
                
                # Split the extracted text into chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Generate embeddings and store in MongoDB
                store_in_mongodb(text_chunks)
                
                # Notify user that processing is complete
                st.success("Done")

# Run the app
if __name__ == "__main__":
    main()
