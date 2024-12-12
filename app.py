import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from io import BytesIO

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "Answer is not available in the context," and don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

# Function to process user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        st.error("Error loading the index. Please process your PDFs first.")
        return
    
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    
    try:
        response = chain.run(context=docs, question=user_question)
        st.write("Reply: ", response)
    except Exception as e:
        st.error(f"An error occurred while processing your question: {str(e)}")

# Main function
def main():
    st.set_page_config("Chat with Multiple PDF")
    st.header("Chat with Multiple PDFs using Gemini 😱")
    
    # JavaScript to detect mobile devices
    mobile_detect_js = """
    <script>
    function detectMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }
    if (detectMobile()) {
        document.body.classList.add('mobile-device');
    }
    </script>
    """
    st.components.v1.html(mobile_detect_js, height=0)
    
    # CSS to show/hide elements based on device type
    st.markdown("""
    <style>
    .mobile-message { display: none; }
    .mobile-device .mobile-message { display: block; }
    </style>
    """, unsafe_allow_html=True)
    
    # Mobile message (hidden by default, shown on mobile devices)
    st.markdown("""
    <div class="mobile-message">
    <div class="stAlert">
    <div class="stAlert-info">
    📱 Welcome mobile user! To upload PDFs and process them, please click the '>' icon in the top-left corner to open the sidebar.
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        if os.path.exists("faiss_index"):
            user_input(user_question)
        else:
            st.warning("Please upload and process PDFs before asking questions.")
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        # Convert all uploaded PDFs to file-like objects
                        pdf_files = [BytesIO(pdf.read()) for pdf in pdf_docs]
                        
                        # Extract text from all PDFs
                        raw_text = get_pdf_text(pdf_files)
                        
                        # Split text into chunks and create vector store
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        
                        st.success("Done processing your PDFs!")
                    except Exception as e:
                        st.error(f"An error occurred while processing your PDFs: {str(e)}")
            else:
                st.warning("Please upload at least one PDF.")

if __name__ == "__main__":
    main()

