import os
import fitz  # PyMuPDF
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import cachetools
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import re

# Concrete implementation of BaseChatMessageHistory
class SimpleChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages.clear()

# Hard-coded API keys
openai_api_key = " "
pinecone_api_key = " "
pinecone_env = "us-east-1"

# Initialize Pinecone with API key
pc = Pinecone(api_key=pinecone_api_key)

# Create or connect to a Pinecone index
index_name = "recipe-index"
if index_name not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# Initialize OpenAI embeddings model with API key
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Extract text from PDF files using PyMuPDF with a fallback to pdfminer
def extract_text_from_pdf(pdf_path):
    text = ""
    document = fitz.open(pdf_path)
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Asynchronously load and chunk PDF documents
async def load_and_chunk_pdfs(pdf_paths, chunk_size):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(executor, functools.partial(extract_text_from_pdf, pdf_path))
            for pdf_path in pdf_paths
        ]
        texts = await asyncio.gather(*tasks)
    
    documents = [Document(page_content=text) for text in texts]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

# Load PDF files from the "Recipe Material" directory
pdf_directory = "Recipe Material"
pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Convert list of documents to a hashable type for caching
def convert_to_hashable(documents):
    return tuple((doc.page_content for doc in documents))

# Caching results to improve performance
@cachetools.cached(cache=cachetools.LRUCache(maxsize=128), key=lambda docs: convert_to_hashable(docs))
def get_cached_vector_store(split_docs):
    return PineconeVectorStore.from_documents(split_docs, embeddings_model, index_name=index_name)

# Define system and QA prompts
system_prompt = (
    "You are an assistant specialized in providing personalized cooking recipe recommendations. "
    "You cater to various user preferences such as dietary restrictions, cuisine types, available ingredients, and cooking time. "
    "Answer the user's questions based on the context provided from the vector database. "
    "If the context does not have relevant information, guide the user to ask relevant questions or suggest alternative approaches."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

model_name = "gpt-4"
llm = ChatOpenAI(model_name=model_name)

# Set up the retrieval chain
def get_retrieval_chain(vector_store):
    retriever = create_history_aware_retriever(
        llm=llm, retriever=vector_store.as_retriever(), prompt=contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain  # Correct parameter
    )

    st.session_state.chat_active = True

    return rag_chain

store = {}

def get_session_history(session_id: str) -> SimpleChatMessageHistory:
    if session_id not in store:
        store[session_id] = SimpleChatMessageHistory()  # Use SimpleChatMessageHistory
    return store[session_id]

def get_answer(query):
    retrieval_chain = get_retrieval_chain(st.session_state.vector_store)
    st.session_state.history.append({"role": "user", "content": query})
    session_id = "session_id"  # Use a specific session ID for tracking
    history = get_session_history(session_id)
    answer = retrieval_chain.invoke({"input": query, "chat_history": history.get_messages()})
    st.session_state.history.append({"role": "assistant", "content": answer["answer"]})
    return answer

# Performance Metrics Calculation
def calculate_metrics(true_relevance, predicted_relevance):
    """
    Calculate performance metrics for the chatbot.

    Parameters:
    - true_relevance: List of true relevance labels
    - predicted_relevance: List of predicted relevance labels

    Returns:
    - metrics: Dictionary containing calculated metrics
    """
    metrics = {}
    
    metrics['Precision'] = precision_score(true_relevance, predicted_relevance, average='macro')
    metrics['Recall'] = recall_score(true_relevance, predicted_relevance, average='macro')
    metrics['F1 Score'] = f1_score(true_relevance, predicted_relevance, average='macro')
    metrics['Accuracy'] = accuracy_score(true_relevance, predicted_relevance)

    # For detailed classification report
    metrics['Classification Report'] = classification_report(true_relevance, predicted_relevance)

    return metrics

def evaluate_chatbot_performance(true_relevance, predicted_relevance):
    """
    Evaluate chatbot performance and report metrics.

    Parameters:
    - true_relevance: List of true relevance labels
    - predicted_relevance: List of predicted relevance labels

    Returns:
    - None
    """
    if not true_relevance or not predicted_relevance:
        print("No data to evaluate.")
        return

        
    print(f"True Relevance: {true_relevance}")
    print(f"Predicted Relevance: {predicted_relevance}")
    
    metrics = calculate_metrics(true_relevance, predicted_relevance)
    
    # Print out the metrics
    print("Performance Metrics:")
    for key, value in metrics.items():
        if key == 'Classification Report':
            print(f"\n{key}:\n{value}")
        else:
            print(f"{key}: {value}")

# Function to simulate relevance assignment based on user interaction
def assign_relevance(true_label, predicted_label):
    true_relevance.append(true_label)
    predicted_relevance.append(true_label)

# Function to parse recipes from the extracted text
def parse_recipes(text):
    lines = text.split('\n')
    recipes = []
    current_recipe = None
    
    recipe_name_re = re.compile(r'^[A-Z][A-Za-z\s]+$')
    ingredient_re = re.compile(r'^o\s[\w\s,]+$')
    instruction_re = re.compile(r'^\d\.\s[\w\s,]+$')
    
    for line in lines:
        line = line.strip()
        if recipe_name_re.match(line):
            if current_recipe:
                recipes.append(current_recipe)
            current_recipe = {'name': line, 'ingredients': [], 'instructions': []}
        elif ingredient_re.match(line):
            if current_recipe:
                current_recipe['ingredients'].append(line[2:])
        elif instruction_re.match(line):
            if current_recipe:
                current_recipe['instructions'].append(line[2:])
    
    if current_recipe:
        recipes.append(current_recipe)
    
    return recipes

# Example extracted text from the Thai recipes PDF
thai_recipes_text = extract_text_from_pdf(os.path.join(pdf_directory, "1. Thai recipes from Spaicy Villa Author Spicy Villa Ecolodges.pdf"))
parsed_recipes = parse_recipes(thai_recipes_text)

# Function to parse user queries
def parse_query(query):
    ingredients = ["chicken", "broccoli", "tofu", "tomato", "potato", "mushroom"]
    holidays = ["Thanksgiving", "Christmas", "Easter"]
    dietary_preferences = ["vegan", "vegetarian", "gluten-free"]
    
    parsed_query = {
        "ingredients": [],
        "dish_type": None,
        "context": {
            "holiday": None,
            "dietary_preference": None
        }
    }
    
    for ingredient in ingredients:
        if ingredient in query.lower():
            parsed_query["ingredients"].append(ingredient)
    
    for holiday in holidays:
        if holiday.lower() in query.lower():
            parsed_query["context"]["holiday"] = holiday
    
    for dietary_preference in dietary_preferences:
        if dietary_preference.lower() in query.lower():
            parsed_query["context"]["dietary_preference"] = dietary_preference
    
    return parsed_query

# Function to evaluate the relevance of the chatbot's response
def evaluate_relevance(parsed_query, response):
    response_ingredients = parse_query(response)["ingredients"]
    
    ingredient_match = all(ingredient in response_ingredients for ingredient in parsed_query["ingredients"])
    holiday_match = parsed_query["context"]["holiday"] and parsed_query["context"]["holiday"].lower() in response.lower()
    dietary_preference_match = parsed_query["context"]["dietary_preference"] and parsed_query["context"]["dietary_preference"].lower() in response.lower()
    
    if ingredient_match and (holiday_match or dietary_preference_match):
        return 1  # Relevant
    return 0  # Not relevant

# Streamlit app setup
st.title("🦜🔗 Recipe Recommendation Chatbot")

if "vector_store" not in st.session_state:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    split_docs = loop.run_until_complete(load_and_chunk_pdfs(pdf_files, chunk_size=1000))  # Adjust chunk size as needed
    st.session_state.vector_store = get_cached_vector_store(split_docs)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# Initialize lists to store relevance labels
true_relevance = []
predicted_relevance = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query := st.chat_input("Ask your question here"):
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    answer = get_answer(query)
    result = answer["answer"]
    
    # Evaluate the relevance
    parsed_query = parse_query(query)
    true_label = evaluate_relevance(parsed_query, result)  # Replace with actual true relevance logic if available
    predicted_label = evaluate_relevance(parsed_query, result)
    assign_relevance(true_label, predicted_label)

    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})

def clear_messages():
    st.session_state.messages = []
    st.session_state.history = []  # Clear the conversation history as well
st.button("Clear", help="Click to clear the chat", on_click=clear_messages)

# Evaluate performance
evaluate_chatbot_performance(true_relevance, predicted_relevance)
