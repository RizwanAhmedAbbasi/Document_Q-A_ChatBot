from langchain.text_splitter import RecursiveCharacterTextSplitter

def splitter_fun(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents(document)
    return chunks