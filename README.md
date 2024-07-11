# Recipe Recommendation Chatbot

## Overview

The Recipe Recommendation Chatbot is designed to provide personalized cooking recipe recommendations. It leverages a Retrieval-Augmented Generation (RAG) pipeline using LangChain to manage interactions and Pinecone for vector storage. The chatbot can handle various user preferences, including dietary restrictions, cuisine types, available ingredients, and cooking time.

## Features

- **Personalized Recipe Recommendations**: Offers suggestions based on user inputs.
- **Context-Aware Responses**: Maintains context across conversations within the same session.
- **Robust Data Handling**: Processes and retrieves information from a pre-existing dataset of recipes stored in PDFs.
- **Out-of-Scope Query Handling**: Provides appropriate responses for queries outside the chatbot's scope.
- **Unavailable Data Management**: Informs users when requested data is not available.

## Installation

### Prerequisites

- Python 3.8 or higher
- Streamlit
- OpenAI API key
- Pinecone API key

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/recipe-recommendation-chatbot.git
   cd recipe-recommendation-chatbot
   
2. **Create and Activate Virtual Environment**

```bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies

```bash
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
