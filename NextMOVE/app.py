import streamlit as st
#import fitz  # PyMuPDF
import pymupdf
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import tempfile
import openai

# Initialize OpenAI API (Replace with your API key)
openai.api_key = '<ENTER_YOUR_OPENAI_API_KEY_HERE>'


def extract_text_from_pdf(pdf_path):
    try:
        #doc = fitz.open(pdf_path)  # Open the PDF document
        doc = pymupdf.open(pdf_path)  # Open the PDF document
        #doc = fitz.Document(pdf_path)  # Open the PDF document
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # Load a page
            text += page.get_text()  # Extract text from the page
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


# Function to vectorize text
def vectorize_text(text):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    vector = model.encode(text)
    return vector


# Function to save vector to MongoDB
def save_vector_to_mongo(vector, text, collection):
    document = {
        'text': text,
        'vector': vector.tolist()  # Convert numpy array to list
    }
    collection.insert_one(document)


# Function to retrieve relevant documents from MongoDB
def retrieve_relevant_docs(query, collection):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_vector = model.encode(query).tolist()
    docs = list(collection.find())
    relevant_docs = sorted(docs, key=lambda doc: cosine_similarity(query_vector, doc['vector']), reverse=True)[:5]
    return relevant_docs


# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2)) / (sum(a * a for a in vec1) ** 0.5 * sum(b * b for b in vec2) ** 0.5)


# Function to generate chatbot response using OpenAI GPT
def generate_response(query, relevant_docs):
    augmented_query = query + " " + " ".join([doc['text'] for doc in relevant_docs])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": augmented_query}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()


# Function to save chat history to MongoDB
def save_chat_history(user_query, bot_response, collection):
    document = {
        'user_query': user_query,
        'bot_response': bot_response
    }
    collection.insert_one(document)


# Streamlit interface
st.image('assets/header.png')
st.title("NextMOVE:  Personal Chess training assistant for preparing against each opponent")


# MongoDB connection with SSL/TLS options
# Connecting to MongoDB cluster and server (Replace with your MongoDB cluster and server connection string)

client = MongoClient(
    "<ENTER_YOUR_MONGODB_CLUSTER_AND_SERVER_CONNECTION_STRING>",
    tls=True,
    tlsAllowInvalidCertificates=True
)


# Database and collections
db = client["vectordb"]
vectors_collection = db["vectors"]
chat_history_collection = db["chat_history"]


# Upload PDF
uploaded_file = st.file_uploader("Upload your opponents games", type="pdf")


if uploaded_file is not None:
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    # Extract text from PDF
    text = extract_text_from_pdf(tmp_pdf_path)

    # Vectorize text
    vector = vectorize_text(text)

    # Display extracted text and vector (for debugging)
    st.write("Extracted Text:")
    st.write(text)
    st.write("Vector:")
    st.write(vector)

    # Save vector to MongoDB
    save_vector_to_mongo(vector, text, vectors_collection)
    st.success("Vector saved to MongoDB successfully!")

# Chatbot interface
st.title("Personalized Chess Assistant")

user_query = st.text_input("Enter your move:")
if user_query:
    # Retrieve relevant documents
    relevant_docs = retrieve_relevant_docs(user_query, vectors_collection)

    # Generate response
    bot_response = generate_response(user_query, relevant_docs)

    # Display response
    st.write("NextMOVE response:")
    st.write(bot_response)

    # Save chat history to MongoDB
    save_chat_history(user_query, bot_response, chat_history_collection)
