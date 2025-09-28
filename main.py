import streamlit as st
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from operator import itemgetter
import os

#helper function for loading data
from src.loader import load_data
from src.splitter import splitter_fun
from src.vector import vectorstore_fun

from src.vector import get_fun

       
google_api_key = get_fun("GOOGLE_API_KEY")
loading=st.empty()

st.title("📚 Document ChatBot (Qdrant + Gemini)")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "chain" not in st.session_state:
    st.session_state["chain"] = None
display=st.empty()

# ---------------- FILE UPLOAD ----------------
st.sidebar.title("📂 Upload your document")
uploaded_file = st.sidebar.file_uploader("Upload your file", type=["pdf"])
start = st.sidebar.button("Start Processing")

if uploaded_file and start:
    with st.spinner("📂 Processing document..."):
        data=load_data(uploaded_file)#calling function for loading data
        chunks = splitter_fun(data)#calling function for splitting data
        vectorstore = vectorstore_fun(chunks)#calling function for creating vectorstore  
        st.session_state["vectorstore"] = vectorstore
        display.write("✅ Document processed successfully!")
    # ---------------- CHAIN SETUP ----------------
    # def format_history(history):
    #     return "\n".join([f"User: {h['question']}\nAssistant: {h['answer']}" for h in history])

    prompt_str = """
    You are an AI assistant. Answer the question using the provided context.
    Context: {context}
    Question: {question}
    Answer:
    """

    query_fetcher = itemgetter("question")
    #history_fetcher = itemgetter("history")Conversation history: {history}
    output_parser = StrOutputParser()
    _prompt = ChatPromptTemplate.from_template(prompt_str)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 15})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=google_api_key,
    )

    setup = {
        "question": query_fetcher,
       # "history": history_fetcher,
        "context": query_fetcher | retriever | (lambda x: "\n\n".join([doc.page_content for doc in x])),
    }
    chain = setup | _prompt | llm | output_parser

    st.session_state["chain"] = chain

# ---------------- CHAT UI ----------------
# Show previous chat
for chat in st.session_state["chat_history"]:
    with st.chat_message("User"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])

# Input box like ChatGPT
query = st.chat_input("Ask a question about your document...")

if query and st.session_state["chain"] is not None:
    with st.chat_message("user"):
        st.write(query)
    with st.spinner("Generating Answer..."): 
        try:
            response = st.session_state["chain"].invoke({
                "question": query,
                # "history": "\n".join(
                #     [f"User: {h['question']}\nAssistant: {h['answer']}" for h in st.session_state["chat_history"]]
                # ),
            })
            with st.chat_message("assistant"):
                st.write(response)

            # Save to history
                st.session_state["chat_history"].append({"question": query, "answer": response})

        except Exception as e:
            st.error(f" Error during chain invocation: {e}")

elif query and st.session_state["chain"] is None:
    st.warning("⚠️ Please upload and process a document first.")
