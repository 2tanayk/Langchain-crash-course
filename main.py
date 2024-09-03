from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

# gives access to the llm, gpt model in our case
llm = ChatOpenAI()

# executing a query
# response = llm.invoke("Hello, how are you?")
# print(response)

# prompt template
prompt = ChatPromptTemplate.from_template("Tell me a joke about a {subject}")

# creating a LLM chain (output from object 1 is passed to 2)
chain = prompt | llm


# invoking the chain
# response = chain.invoke({"subject":"dog"})
# print(response)


# system =  how the AI model should behave, human = asked by enduser
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI chef. Create a unique recipe based on the following main ingredient"),
    ("human", "{input}")
    ]
)


chain = prompt | llm

#result = chain.invoke({"input":"tomatoes"})
# print(result)


#output parsers

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tell me a joke about the following subject"),
    ("human", "{input}")
    ]
)

#parser
parser = StrOutputParser()

chain = prompt | llm | parser

# we get a string
result = chain.invoke({"input":"dog"})















# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain.chains import create_history_aware_retriever
# from langchain_core.prompts import MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever
# from langchain_core.prompts import MessagesPlaceholder
# from langchain_core.messages import HumanMessage, AIMessage

# # our LLM (GPT)
# llm = ChatOpenAI(openai_api_key="")

# # asking it questions without any training data/context
# #print(llm.invoke("how can langsmith help with testing?"))

# #print('\n\n')

# # asking it questions with some training data/context using prompt template
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are world class technical documentation writer."),
#     ("user", "{input}")
# ])

# chain = prompt | llm 

# #print(chain.invoke({"input": "how can langsmith help with testing?"}))

# #print('\n\n')

# # doing same to get answer in string format
# output_parser = StrOutputParser()

# chain = prompt | llm | output_parser

# #print(chain.invoke({"input": "how can langsmith help with testing?"}))

# loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

# docs = loader.load()

# embeddings = OpenAIEmbeddings()

# text_splitter = RecursiveCharacterTextSplitter()
# documents = text_splitter.split_documents(docs)
# vector = FAISS.from_documents(documents, embeddings)

# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")

# document_chain = create_stuff_documents_chain(llm, prompt)

# retriever = vector.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
# print(response["answer"])
# print('\n\n\n')

# prompt = ChatPromptTemplate.from_messages([
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
# ])
# retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
# response = retriever_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })

# print(response)
# print('\n\n\n')

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Answer the user's questions based on the below context:\n\n{context}"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
# ])
# document_chain = create_stuff_documents_chain(llm, prompt)

# retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
# response = retrieval_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })

# print(response["answer"])