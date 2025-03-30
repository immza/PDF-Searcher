import os
import chromadb
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from constants import CHROMA_SETTINGS
import shutil
if os.path.exists("db"):
    shutil.rmtree("db")


persist_directory = "db"  # Path for Chroma persistent storage

def main():
    # Ensure the Chroma client is initialized properly with the correct settings
    try:
        client = chromadb.PersistentClient(path=persist_directory, settings=CHROMA_SETTINGS)
    except Exception as e:
        print(f"Error initializing Chroma client: {e}")
        return

    collection_name = "mydb"

    # Explicitly delete the collection if it exists
    try:
        collections = client.list_collections()
        if collection_name in collections:
            print(f"Collection '{collection_name}' exists. Deleting it...")
            client.delete_collection(collection_name)
    except Exception as e:
        print(f"Error deleting collection: {e}")

    # Load and process your PDF files
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Processing file: {file}")
                loader = PDFMinerLoader(os.path.join(root, file))
                documents = loader.load()
                print(f"Loaded {len(documents)} documents")

                # Split text into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                texts = text_splitter.split_documents(documents)
                print(f"Split into {len(texts)} chunks")

                # Create embeddings
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                # Create vector store and persist to Chroma
                try:
                    db = Chroma.from_documents(
                        texts,
                        embeddings,
                        persist_directory=persist_directory,
                        collection_name=collection_name
                    )
                    db.persist()  # Ensure the data is saved to disk
                    print(f"Successfully added vector store for {file}")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    print("Finished processing documents.")



if __name__ == "__main__":
    main()

