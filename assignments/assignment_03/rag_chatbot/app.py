import chainlit as cl
import pandas as pd
import os
import validators
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor
from phoenix.otel import register

import bs4
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import (
    create_history_aware_retriever, 
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema.runnable.config import RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler


def process_url_content(url):
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=(url,),
    )
    loader.requests_kwargs = {'verify': False}
    docs = loader.load()
    # Extract the page content
    page_content = docs[0].page_content

    # Parse the content with BeautifulSoup
    soup = bs4.BeautifulSoup(page_content, 'html.parser')

    # Extract text from the parsed HTML
    text = soup.get_text()

    doc = Document(
        page_content=text,
        metadata=docs[0].metadata,
        id=1,
    )

    return [doc]


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    store = cl.user_session.get("store", {})
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


async def validate_url(url):
    if not validators.url(url):
        await cl.Message(content="Invalid URL. Please try again.").send()
        return False
    else:
        return True
    
    
async def set_up_llm_chain():
    


@cl.on_chat_start
async def on_chat_start():

    # print(f"Hugging Face Token Read: {os.environ.get("HUGGINGFACEHUB_API_TOKEN")}")

    px_session = px.launch_app()

    tracer_provider = register(
        project_name="llm-classroom-rag",  # Default is 'default'
    )

    cl.user_session.set("phoenix_session", px_session)

    LangChainInstrumentor().instrument()

    # Create a Chroma vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        # "sentence-transformers/all-mpnet-base-v2"
    )

    vector_store = Chroma(
        collection_name=f"session_{cl.user_session.get("id")}",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
    )

    # Create the LLM
    llm = HuggingFaceEndpoint(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        temperature=0.01,
        do_sample=False,
        repetition_penalty=1.03,
    )

    chat_llm = ChatHuggingFace(llm=llm, verbose=True)

    urls = None
    tracer_provider = register(
        project_name="rag-chatbot",  # Default is 'default'
    )

    cl.user_session.set("tracer_provider", tracer_provider)

    # Wait for the user to upload a file
    while urls is None:
        resp = await cl.AskUserMessage(
            content="Please send me a url of the blog you want to chat with.",
            timeout=180,
        ).send()

        print(resp)

        if not await validate_url(resp["output"]):
            continue
        else:
            urls = [resp["output"]]

    print(urls)

    url = urls[0]

    msg = cl.Message(
        content=f"Processing `{url}`..."
    )
    await msg.send()

    # load the file
    texts = process_url_content(url)

    # print(texts)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    splits = text_splitter.split_documents(texts)

    print("Initializing Vectorstore")
    await cl.make_async(vector_store.add_documents)(
        documents=splits,
    )
    print("Initialized Vectorstore")

    # Contextualize question #
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate one standalone question which can be understood "
        "without the chat history. Ensure not to loose the semantic & syntactic meaning of the question."
        "Do NOT answer the question or formulate multiple questions."
        "just reformulate it if needed. If it can't be done return the users question as is."
        "Follow the below JSON format of output:\n\n"
        """{{input: <put the reformatted question here>}}"""

    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever = vector_store.as_retriever()
    print("Initialized Retriever")

    history_aware_retriever = create_history_aware_retriever(
        chat_llm, retriever, contextualize_q_prompt
    )
    cl.user_session.set("retriever", history_aware_retriever)
    print("Created History aware Retriever")

    # Answer question #
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    print("Created RAG Chain")

    # Statefully manage chat history #
    store = {}
    cl.user_session.set("store", store)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    print("Created History Aware Conversational RAG Chain")

    # Let the user know that the system is ready
    msg.content = f"Processing `{url}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", conversational_rag_chain)


def process_chunks(chunk: dict):
    
    for k, v in chunk.items():
        if isinstance(v, Document):
            processed = {}
            processed["metadata"] = v.metadata
            processed["page_content"] = v.page_content
            chunk[k] = processed
        elif isinstance(v, list) or isinstance(v, dict) or isinstance(v, str):
            pass
        else:
            chunk[k] = "Unexpected output."

    return chunk
            

@cl.on_message
async def main(message: cl.Message):

    runnable = cl.user_session.get("chain")
    # sync_cb = cl.LangchainCallbackHandler()
    cb = cl.AsyncLangchainCallbackHandler()
    # sync_cb.on_llm_new_token(lambda token: process_chunks(token))

    # res = await chain.acall(message.content, callbacks=[cb])
    # answer = res["answer"]

    print(f"User Session: {cl.user_session.get("id")}")

    result = await runnable.ainvoke(
        {"input": message.content},
        config=RunnableConfig(
            callbacks=[cb],
            configurable={"session_id": cl.user_session.get("id")}
        )
    ) 

    msg = cl.Message(content=result["answer"])
    # context_msg = cl.Message(
    #     content=f"Context: {result["context"]}",
    # )

    await msg.send()
    # await context_msg.send()
    
@cl.on_chat_end
async def on_chat_end():
    # px_session = cl.user_session.get("phoenix_session")
    df = px.Client().get_spans_dataframe()
    persist_path = (
        os.environ.get("TRACER_LOG_PATH")
        + "/" + cl.user_session.get("id")
        + ".csv"
    )
    directory = os.path.dirname(persist_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directories: {directory}")
    else:
        print(f"Directories already exist: {directory}")

    df.to_csv(persist_path, sep=",", header=True, mode="a")
    # px_session.stop()

    # cl.user_session.clear()


