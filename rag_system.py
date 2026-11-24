"""RAG completo: recuperador híbrido (BM25 + Pinecone) y generación de la respuesta con un LLM.

    python rag_system.py "¿Qué pasa con lru_cache si maxsize es None?"
"""

import re
import sys
import warnings

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# BM25Retriever sólo está en langchain-community, que avisa que lo van a discontinuar; funciona igual
warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)
from langchain_community.retrievers import BM25Retriever  # noqa: E402

from config import NAMESPACE, chequear_variables, crear_llm, crear_vectorstore
from ingest import cargar_chunks, vectores_en_namespace

TOP_K = 5
NO_LO_SE = "No lo sé"

SYSTEM_PROMPT = """Sos un asistente que responde preguntas sobre la documentación de la biblioteca estándar de Python.
Respondé ÚNICAMENTE con la información del CONTEXTO (fragmentos de la documentación, en inglés).

Reglas:
- Si la respuesta no está en el CONTEXTO, respondé exactamente "{no_lo_se}" y dejá las fuentes vacías.
- No uses conocimiento propio aunque sepas la respuesta: sólo lo que dice el CONTEXTO.
- En las fuentes poné el id (lo que está entre corchetes) de cada fragmento que usaste.
- Respondé en español, breve y concreto.

{format_instructions}"""

HUMAN_PROMPT = """CONTEXTO:
{contexto}

PREGUNTA: {pregunta}"""


class RespuestaRAG(BaseModel):
    """La respuesta generada y de qué chunks salió."""

    respuesta: str = Field(description=f'La respuesta, usando sólo el contexto. Si no está: "{NO_LO_SE}".')
    fuentes: list[str] = Field(
        default_factory=list,
        description='Ids de los fragmentos usados, tal cual aparecen entre corchetes (ej. "functools-p3-c1").',
    )

    @property
    def encontrada(self) -> bool:
        return not self.respuesta.strip().startswith(NO_LO_SE)


def tokenizar(texto: str) -> list[str]:
    # el BM25 por defecto separa sólo por espacios: "lru_cache(maxsize=128," quedaría como una palabra.
    # Así "lru_cache" y "maxsize" matchean aunque estén pegados a paréntesis o comas.
    return re.findall(r"\w+", texto.lower())


def formatear_contexto(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(f"[{d.metadata['chunk_id']}]\n{d.page_content}" for d in docs)


class RAGSystem:
    """Recupera los top-5 chunks combinando BM25 y búsqueda vectorial, y genera la respuesta con un LLM."""

    def __init__(
        self,
        vectorstore=None,
        chunks: list[Document] | None = None,
        llm=None,
        k: int = TOP_K,
        # BM25 / vectorial. Con 50/50 el ruido de BM25 bajaba la precisión en las preguntas conceptuales;
        # 30/70 dio lo mismo en las directas y mejor en las conceptuales (ver Resultados en el README)
        pesos=(0.3, 0.7),
    ):
        self.k = k
        # BM25 necesita el corpus en memoria: uso los mismos chunks que se subieron a Pinecone
        self.bm25 = BM25Retriever.from_documents(chunks or cargar_chunks(), k=k, preprocess_func=tokenizar)
        self.vectorstore = vectorstore or crear_vectorstore()
        self.vectorial = self.vectorstore.as_retriever(search_kwargs={"k": k})
        # combina los dos rankings con Reciprocal Rank Fusion; los docs que aparecen en ambos suben
        self.retriever = EnsembleRetriever(retrievers=[self.bm25, self.vectorial], weights=list(pesos))

        parser = PydanticOutputParser(pydantic_object=RespuestaRAG)
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]).partial(
            no_lo_se=NO_LO_SE, format_instructions=parser.get_format_instructions()
        )
        # si el modelo no devuelve el JSON bien, se reintenta
        self.generador = (prompt | (llm or crear_llm()) | parser).with_retry(
            retry_if_exception_type=(OutputParserException,), stop_after_attempt=3
        )

    # --- Recuperación --------------------------------------------------------

    def buscar(self, consulta: str) -> list[Document]:
        # el ensemble puede devolver hasta 2k (la unión de los dos), me quedo con el top-k
        return self.retriever.invoke(consulta)[: self.k]

    async def abuscar(self, consulta: str) -> list[Document]:
        """Igual que buscar() pero async, para no bloquear el event loop."""
        return (await self.retriever.ainvoke(consulta))[: self.k]

    # --- Generación ----------------------------------------------------------

    def responder(self, pregunta: str) -> RespuestaRAG:
        docs = self.buscar(pregunta)
        respuesta = self.generador.invoke({"contexto": formatear_contexto(docs), "pregunta": pregunta})
        return self._validar_fuentes(respuesta, docs)

    async def aresponder(self, pregunta: str) -> RespuestaRAG:
        docs = await self.abuscar(pregunta)
        respuesta = await self.generador.ainvoke({"contexto": formatear_contexto(docs), "pregunta": pregunta})
        return self._validar_fuentes(respuesta, docs)

    @staticmethod
    def _validar_fuentes(respuesta: RespuestaRAG, docs: list[Document]) -> RespuestaRAG:
        # me quedo sólo con fuentes de chunks que realmente se recuperaron (descarto las inventadas)
        recuperados = {d.metadata["chunk_id"] for d in docs}
        fuentes = [f for f in respuesta.fuentes if f in recuperados] if respuesta.encontrada else []
        return respuesta.model_copy(update={"fuentes": fuentes})

    def chequear_indice_cargado(self) -> None:
        if vectores_en_namespace(self.vectorstore.index) == 0:
            raise SystemExit(f"El namespace '{NAMESPACE}' está vacío: corré primero python ingest.py")


def main(pregunta: str) -> None:
    chequear_variables("PINECONE_API_KEY", "OPENAI_API_KEY")
    rag = RAGSystem()
    rag.chequear_indice_cargado()

    print(f"Pregunta: {pregunta}\n\nTop-{rag.k} recuperado (BM25 + Pinecone):")
    for i, doc in enumerate(rag.buscar(pregunta), start=1):
        m = doc.metadata
        # int(): Pinecone devuelve los números de la metadata como float (4.0)
        print(f"  {i}. {m['chunk_id']} ({m['fuente']}, pág. {int(m['pagina'])}, {m['categoria']})")

    r = rag.responder(pregunta)
    print(f"\nRespuesta: {r.respuesta}")
    print(f"Fuentes: {', '.join(r.fuentes) or '-'}")


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or "¿Qué pasa con lru_cache si maxsize es None?")
