import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS

# Load the model and tokenizer
checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.float32)
base_model.to("cpu")  # Move to CPU explicitly

def llm_pipeline():
    # Create the HuggingFace pipeline for text-to-text generation
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    # Wrap the pipeline with HuggingFacePipeline for integration with Langchain
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def qa_llm():
    # Initialize the QA pipeline with Langchain
    llm = llm_pipeline()

    # Set up embeddings using the sentence-transformers model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Initialize the Chroma vector store with embeddings and the correct collection name
    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
        collection_name="mydb"  # Ensure the collection name matches the one used in ingest.py
    )

    # Set up the retriever from the vector store
    retriever = db.as_retriever()

    # Set up the RetrievalQA chain with the LLM and retriever
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True
    )
    return qa


def process_answer(instruction):
    try:
        # Retrieve the answer using the QA pipeline
        qa = qa_llm()
        response = qa.invoke(instruction)  # Use `invoke()` instead of `run()`

        # Extract the answer and source documents from the response
        answer = response.get("result", "No answer found.")  # Default to "No answer found."
        sources = response.get("source_documents", [])  # Default to empty list if no sources found

        return answer, sources
    except Exception as e:
        # If any error occurs, show an error message
        st.error(f"Error: {str(e)}")
        return "An error occurred while processing your request.", []


def main():
    # Streamlit app interface
    st.title("Search Your PDF 🦜")

    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI Powered Question and Answering app that responds to questions about your PDF File            
            """
        )

    # Input question from user
    question = st.text_area("Enter Your Question")

    if st.button("Search"):
        # Display the user's question
        st.info("Your question: " + question)

        # Get the answer and sources by calling `process_answer()`
        answer, sources = process_answer(question)

        # Display the answer
        st.success("Answer:")
        st.write(answer)

        # If sources are found, display them as well
        if sources:
            st.info("Sources:")
            for doc in sources:
                st.write(f"- {doc.metadata['source']}")

if __name__ == '__main__':
    main()
