1. Model Development
1.1 Overview of RAG-Based Architecture
Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based methods with generative models. Instead of relying solely on a pre-trained model's parameters, it dynamically fetches relevant context from a knowledge base, improving accuracy and relevance.
1.2 Implementation Details
Data Ingestion and Preprocessing
Implemented ingest.py to process PDF documents.


Extracted text from PDFs using PDFMinerLoader.


Split the extracted text into meaningful chunks using RecursiveCharacterTextSplitter.


Generated vector embeddings for each chunk using HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2).


Stored embeddings in a vector database (Chroma).


Removed the existing database before processing new PDFs to ensure clean indexing.


Retrieval Mechanism
User queries are transformed into embeddings.


The most relevant document chunks are retrieved from Chroma.


The retrieved chunks are then passed to LaMini-T5-738M (a seq2seq model) to generate a response.


1.3 Fine-Tuning and Optimization
Applied chunking strategies to balance retrieval granularity.


Fine-tuned the number of retrieved documents for optimal performance.



2. Online Deployment 
2.1 Web Interface
Developed a Streamlit UI (app.py) to interact with the model.


Implemented a text input for user queries.


Displayed retrieved answers along with source document references.


Used Streamlit Expander to provide an app description.


2.2 Running the Application
To run this project, follow these steps:
Run python ingest.py to process and store PDF embeddings.


Run streamlit run app.py to launch the web interface.


3. Testing and Evaluation (5 Marks)
3.1 Sample Queries and Responses
Example 1:
 Query: "What is Global Warming?"
 Response: "Greenhouse gas is a gas that traps heat from the sun and contributes to global warming”.


Conclusion
This RAG-based PDF searcher successfully combines retrieval-based and generative AI techniques, allowing users to query large PDFs efficiently while maintaining high accuracy and relevance.


