"""Configuración compartida: variables de entorno, embeddings y conexión a Pinecone."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
INDEX_NAME = os.getenv("INDEX_NAME", "rag-python-docs")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "python-docs")  # separa estos documentos de cualquier otro dato del índice

EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536  # la dimensión de text-embedding-3-small: el índice tiene que tener la misma
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"


def chequear_variables(*nombres: str) -> None:
    faltan = [n for n in nombres if not os.getenv(n)]
    if faltan:
        raise SystemExit(f"Faltan variables de entorno: {', '.join(faltan)}. Copiá .env.example a .env y completalas.")


def crear_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def crear_llm() -> ChatOpenAI:
    # temperatura 0: para responder sobre documentación quiero la respuesta más predecible
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)


def crear_cliente_pinecone():
    from pinecone import Pinecone

    return Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def crear_vectorstore(embeddings=None):
    """El PineconeVectorStore de LangChain: guarda el texto del chunk en la metadata (clave "text")."""
    from langchain_pinecone import PineconeVectorStore

    index = crear_cliente_pinecone().Index(INDEX_NAME)
    return PineconeVectorStore(
        index=index,
        embedding=embeddings or crear_embeddings(),
        namespace=NAMESPACE,
        text_key="text",
    )
