from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter         
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

model = ChatOpenAI(
    model='gpt-3.5-turbo-1106',
    temperature=0.4
)

def get_documents_from_web_and_split(url):
    # load
    loader = WebBaseLoader(url)
    docs = loader.load()
    #transform

    #breaking the document/s into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
    splitDocs = splitter.split_documents(docs)

    return splitDocs

def create_vector_store(docs):
    #embed
    embedding = OpenAIEmbeddings()
    #store
    vectorStore = FAISS.from_documents(docs, embedding)
    return vectorStore


#scrapes and gets all information from this web page as a document/s
docs = get_documents_from_web_and_split('https://python.langchain.com/docs/expression_language/')
vectorStore = create_vector_store(docs)

#context variable mandatory for retrieval chain
prompt = ChatPromptTemplate.from_template("""
Answer the user's question:
Context: {context}
Question: {input}
""")

# chain = prompt | model

# pretty identical to the above statement to create a chain
# but also allows us to pass documents
document_chain = create_stuff_documents_chain(model,prompt)

# retrieve
retriever = vectorStore.as_retriever()

retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

response = retrieval_chain.invoke({
    "input":"What is LCEL?"
})

print(response["answer"])