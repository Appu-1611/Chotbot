import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "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"

#Upload PDF files
st.header("My First Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

#Extract text from PDF

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
#    st.write(text)

#Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=1000, chunk_overlap=150, length_function=len)
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

 # generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create a vector store from the chunks
    vector_store = FAISS.from_texts(chunks, embeddings)

#Get question from user
    user_question = st.text_input("Type your question here")

#Do simlarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
#        st.write(match)
        
#define the llm
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )

 #output results
        #chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)