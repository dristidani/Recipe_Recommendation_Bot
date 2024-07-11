Recipe Recommendation Chatbot
Overview
The Recipe Recommendation Chatbot is designed to provide personalized cooking recipe recommendations. It leverages a Retrieval-Augmented Generation (RAG) pipeline using LangChain to manage interactions and Pinecone for vector storage. The chatbot can handle various user preferences, including dietary restrictions, cuisine types, available ingredients, and cooking time.

Features
Personalized Recipe Recommendations: Offers suggestions based on user inputs.
Context-Aware Responses: Maintains context across conversations within the same session.
Robust Data Handling: Processes and retrieves information from a pre-existing dataset of recipes stored in PDFs.
Out-of-Scope Query Handling: Provides appropriate responses for queries outside the chatbot's scope.
Unavailable Data Management: Informs users when requested data is not available.
Installation
Prerequisites
Python 3.8 or higher
Streamlit
OpenAI API key
Pinecone API key
Setup
Clone the Repository

bash
Copy code
git clone https://github.com/yourusername/recipe-recommendation-chatbot.git
cd recipe-recommendation-chatbot
Create and Activate Virtual Environment

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

bash
Copy code
pip install -r requirements.txt
Set Up Secrets

Create a secrets.toml file in the .streamlit directory with the following content:

toml
Copy code
[openai]
api_key = "your_openai_api_key"

[pinecone]
api_key = "your_pinecone_api_key"
environment = "your_pinecone_environment"
Run the Application

bash
Copy code
streamlit run recipechatbot.py
Usage
Asking Questions
Start the chatbot and interact through the chat interface.
You can ask for recipe recommendations based on specific ingredients, dietary restrictions, or cuisine types.
Handling Out-of-Scope Queries
The chatbot will inform users when a question is out of scope.

Example: "Why is the sky blue?"

plaintext
Copy code
I'm here to assist with cooking recipe recommendations. For inquiries about natural phenomena such as why the sky is blue, you might need to consult a science resource or ask a science expert. However, I'm here to help with any cooking or recipe questions you have!
Handling Unavailable Data
The chatbot will inform users if the requested information is not available.

Example: "Can you give me a recipe for a dish not in the database?"

plaintext
Copy code
I'm sorry, but I don't have the information you're looking for in my database. Can I help you with something else related to cooking recipes or ingredients?
Code Structure
recipechatbot.py: Main application file.
requirements.txt: List of required Python packages.
.streamlit/secrets.toml: Configuration file for API keys.
Report
Approach Taken
The chatbot was developed using a combination of LangChain for managing interactions and Pinecone for vector storage. The primary goal was to create a RAG pipeline to provide accurate and context-aware recipe recommendations.

Challenges Faced
Integration with Pinecone: Ensuring efficient indexing and retrieval of recipe data.
Context Management: Maintaining conversation context across multiple user interactions.
Handling Out-of-Scope Queries: Implementing a mechanism to detect and respond appropriately to out-of-scope queries.
Overcoming Challenges
Efficient Data Chunking: Implemented an asynchronous method to load and chunk PDF data, improving response times.
Enhanced Prompts: Used detailed and structured prompts to guide the language model in generating accurate and context-aware responses.
Error Handling: Added checks for out-of-scope queries and unavailable data to enhance user experience.
Video Demonstration
A video demonstration of the chatbot in action is available on YouTube. Watch the Demo

Contributing
If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are welcome.

License
This project is licensed under the MIT License.
