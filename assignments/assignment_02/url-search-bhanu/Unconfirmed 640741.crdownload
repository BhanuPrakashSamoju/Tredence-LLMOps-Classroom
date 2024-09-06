import os
import openai
import bs4
from promptflow import tool
from promptflow.connections import CustomConnection
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore



def load_url(urls: list):
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=(urls)
    )
    docs = loader.load()
    soups = [bs4.BeautifulSoup(html_doc.page_content, 'html.parser') for html_doc in docs]
    for page in soups:
        for script in page(['script', 'style']):
            script.decompose()

    text = "\n\n".join([soup.get_text() for soup in soups])
    print(text)
        
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    return splits

# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(urls: list, query:str, openai_conn: CustomConnection) -> str:
    os.environ["OPENAI_API_TYPE"] = "azure"
    print(openai_conn)
    os.environ["AZURE_OPENAI_API_VERSION"] = openai_conn["OpenAIEmbeddingModelVersion"]
    os.environ["AZURE_OPENAI_ENDPOINT"] = openai_conn["OpenAIEmbeddingModelURL"]
    os.environ["AZURE_OPENAI_API_KEY"] = openai_conn["OpenAIEmbeddingModelKey"]
    
    embeddings = AzureOpenAIEmbeddings(
        deployment = openai_conn["OpenAIDeploymentName"],
        model = "text-embedding-ada-002"
    )

    doc_splits = load_url(urls)
    # docs = [s.page_content for s in doc_splits]

    vectorstore = InMemoryVectorStore.from_texts(
        doc_splits,
        embedding=embeddings,
    )

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()

    # Retrieve the most similar text
    retrieved_documents = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in retrieved_documents])

    return context