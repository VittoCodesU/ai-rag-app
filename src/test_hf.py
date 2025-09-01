import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id=os.getenv("HF_LLM_REPO"),
        task="conversational",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
)

print(llm.invoke("Di hola en una frase corta"))
