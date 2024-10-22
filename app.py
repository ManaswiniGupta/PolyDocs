import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import re

load_dotenv()

def estimate_tokens(text):
    """
    Estimate tokens for Llama models using word-based approximation
    This is a simple estimation - Llama uses BytePair Encoding (BPE) tokenization
    """
    # Split on whitespace and punctuation
    words = re.findall(r'\b\w+\b|\S', text)
    # Approximate token count - Llama typically uses slightly more tokens than words
    return int(len(words) * 1.3)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Using estimated token count for chunk size
    # Targeting roughly 2000 tokens per chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Characters, approximately 2000 tokens
        chunk_overlap=150,
        length_function=estimate_tokens,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks to generate embeddings from.")
    
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vector_store.save_local("faiss_index")
    return vector_store

def generate_response(client, system_prompt, user_prompt, max_tokens=5000):
    """Generate response with Llama-specific token limit handling"""
    try:
        # Estimate total tokens
        total_prompt_tokens = estimate_tokens(system_prompt + user_prompt)
        
        # If prompts exceed token limit, truncate the user prompt
        if total_prompt_tokens > max_tokens:
            # Leave room for system prompt and response tokens
            available_tokens = max_tokens - estimate_tokens(system_prompt) - 1000
            
            # Truncate context while maintaining question
            context, question = user_prompt.split("Question:", 1)
            while estimate_tokens(context) > available_tokens:
                # Remove sentences from context until within token limit
                context_parts = context.split(". ")
                context = ". ".join(context_parts[:-1]) + "."
            
            user_prompt = f"{context}\nQuestion:{question}"
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama3-70b-8192",
            temperature=0.3,  # Lower temperature for more focused responses
            max_tokens=2000,  # Reserve tokens for response
            top_p=0.9
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        if "rate_limit_exceeded" in str(e):
            return "The response was too large. Please try asking about a more specific aspect of the document."
        st.error(f"An error occurred: {str(e)}")
        return f"Error generating response: {str(e)}"

def user_input(user_question):
    try:
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ API key not found. Please check your environment variables.")
            return
        
        client = Groq(api_key=api_key)
        
        # Load embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Get relevant documents with limited results
        docs = vector_store.similarity_search(user_question, k=2)  # Reduced to top 2 for better token management
        
        # Combine document contents with weights favoring more relevant matches
        context_parts = []
        for i, doc in enumerate(docs):
            context_parts.append(doc.page_content)
        
        context = " ".join(context_parts)
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided documents. 
        Provide clear, concise answers while maintaining accuracy. If the context doesn't contain relevant information,
        acknowledge this limitation."""
        
        # Combine context and question
        prompt = f"Context: {context}\n\nQuestion: {user_question}\n\nAnswer:"
        
        response = generate_response(client, system_prompt, prompt)
        st.write("Reply: ", response)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again with a shorter question or upload smaller documents.")

def main():
    st.set_page_config(page_title="Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    with st.sidebar:
        st.write("---")
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files & Click on the Submit & Process Button",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
                
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

        st.write("---")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()