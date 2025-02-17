# ChatBot

This repository contains a chatbot built using Flask, LangChain, and Qdrant. The chatbot is designed to answer questions based on the provided content by leveraging Qdrant as a vector database for efficient retrieval-augmented generation (RAG).

## Features
- **Flask Backend:** Lightweight and efficient API for handling user queries.
- **LangChain Integration:** Enables structured retrieval and natural language processing.
- **Qdrant Vector Database:** Stores and retrieves embeddings for semantic search.
- **Contextual Responses:** The chatbot provides answers based on the stored content.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/syeddanishsaleem/ChatBot.git
   cd ChatBot

2. Create a virtual environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

4. Install dependencies:
   ```sh
   pip install -r requirements.txt

5. Set up environment variables:

    Create a .env file in the root directory and add the following content:
    ```sh
    API_KEY=your_openai_api_key_here
    
6. Replace your_openai_api_key_here with your actual OpenAI API key.
   Run the Flask application:
   ```sh
    python app.py
   
7. Access the chatbot at:
    ```sh
    http://127.0.0.1:5000
