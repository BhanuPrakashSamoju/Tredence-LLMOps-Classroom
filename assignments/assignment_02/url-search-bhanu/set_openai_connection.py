from promptflow import tool
from promptflow.connections import CustomConnection
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002"
)


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(input1: str, conn:CustomConnection) -> str:
    return 'hello ' + input1