import os
import gradio as gr
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load the Hugging Face API token from environment variables
api_key = os.getenv("HF_TOKEN")


def answer_query(pdf_path, query):
    # Load PDF and split into document chunks
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Retrieve relevant chunks for context
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Use the model from Hugging Face Hub with API token
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.7, "max_length": 512},
        huggingfacehub_api_token=api_key
    )
    
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = llm.invoke(prompt)
    return response

def chatbot_interface(pdf, query):
    if pdf is None:
        return "Please upload a PDF file."
    pdf_path = pdf.name
    return answer_query(pdf_path, query)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 📄 PDF Chatbot")
    pdf_input = gr.File(label="Upload PDF")
    query_input = gr.Textbox(label="Enter your question")
    output = gr.Textbox(label="Answer")
    btn = gr.Button("Get Answer")
    btn.click(chatbot_interface, inputs=[pdf_input, query_input], outputs=output)

demo.launch(share=True)
