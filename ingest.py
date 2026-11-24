"""Ingesta: lee los PDFs de data/, los parte en chunks y los sube a Pinecone con su metadata.

    python ingest.py               # sube sólo si el namespace está vacío
    python ingest.py --reindexar   # borra el namespace y vuelve a subir todo
"""

import argparse
import re
import time

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import DATA_DIR, INDEX_NAME, NAMESPACE, chequear_variables, crear_cliente_pinecone, crear_vectorstore
from setup_index import crear_indice_si_no_existe

CHUNK_SIZE = 800  # tokens: en el rango de 500-800 que recomienda la consigna
CHUNK_OVERLAP = 100

# etiquetas de cada documento: la categoría sirve para filtrar búsquedas
CATEGORIAS = {
    "json.pdf": "serializacion",
    "csv.pdf": "serializacion",
    "itertools.pdf": "programacion-funcional",
    "functools.pdf": "programacion-funcional",
    "re.pdf": "texto",
    "datetime.pdf": "fechas",
}

encoding = tiktoken.get_encoding("cl100k_base")  # el tokenizer de text-embedding-3-small


def contar_tokens(texto: str) -> int:
    return len(encoding.encode(texto))


def limpiar(texto: str) -> str:
    texto = re.sub(r"[ \t]+\n", "\n", texto)  # espacios al final de las líneas
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def cargar_paginas(data_dir=DATA_DIR) -> list[Document]:
    """Un Document por página de cada PDF, con fuente, página y categoría en la metadata."""
    paginas = []
    for pdf in sorted(data_dir.glob("*.pdf")):
        for numero, pagina in enumerate(PdfReader(pdf).pages, start=1):
            texto = limpiar(pagina.extract_text() or "")
            if texto:
                paginas.append(
                    Document(
                        page_content=texto,
                        metadata={
                            "fuente": pdf.name,
                            "modulo": pdf.stem,
                            "pagina": numero,
                            "categoria": CATEGORIAS.get(pdf.name, "otros"),
                            "etiquetas": ["python-3.12", "stdlib", pdf.stem],
                        },
                    )
                )
    return paginas


def cargar_chunks(data_dir=DATA_DIR) -> list[Document]:
    """Parte las páginas en chunks. La usan la ingesta y el BM25, así los dos ven exactamente los mismos chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=contar_tokens,
    )
    chunks = splitter.split_documents(cargar_paginas(data_dir))

    contador: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        clave = (chunk.metadata["fuente"], chunk.metadata["pagina"])
        contador[clave] = contador.get(clave, 0) + 1
        # id estable (no uuid): si vuelvo a ingestar, el upsert pisa los mismos vectores en vez de duplicarlos
        chunk.id = f"{chunk.metadata['modulo']}-p{chunk.metadata['pagina']}-c{contador[clave]}"
        chunk.metadata["chunk_id"] = chunk.id
        chunk.metadata["tokens"] = contar_tokens(chunk.page_content)
    return chunks


def vectores_en_namespace(index) -> int:
    resumen = index.describe_index_stats().namespaces.get(NAMESPACE)
    return resumen.vector_count if resumen else 0


def ingestar(vectorstore, index, reindexar: bool = False) -> int:
    """Sube los chunks a Pinecone y devuelve cuántos subió (0 si ya estaban)."""
    existentes = vectores_en_namespace(index)
    if existentes and not reindexar:
        print(f"El namespace '{NAMESPACE}' ya tiene {existentes} vectores, no vuelvo a subir (usá --reindexar)")
        return 0
    if existentes:
        index.delete(delete_all=True, namespace=NAMESPACE)

    chunks = cargar_chunks()
    for fuente in sorted({c.metadata["fuente"] for c in chunks}):
        cantidad = sum(c.metadata["fuente"] == fuente for c in chunks)
        print(f"{fuente}: {cantidad} chunks")

    # add_documents genera los embeddings y hace el upsert en lotes
    vectorstore.add_documents(chunks, ids=[c.id for c in chunks], batch_size=50)
    print(f"Subidos {len(chunks)} chunks al índice '{INDEX_NAME}', namespace '{NAMESPACE}'")
    return len(chunks)


def esperar_indexado(index, cantidad: int, timeout: float = 60) -> None:
    """Pinecone tarda unos segundos en dejar buscables los vectores nuevos: espero antes de terminar,
    así si después corro evaluate.py no busca sobre un índice a medio cargar."""
    inicio = time.monotonic()
    while vectores_en_namespace(index) < cantidad:
        if time.monotonic() - inicio > timeout:
            print("Pinecone todavía está indexando; esperá unos segundos antes de buscar")
            return
        time.sleep(2)
    print("Vectores disponibles para buscar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sube los documentos de data/ a Pinecone.")
    parser.add_argument("--reindexar", action="store_true", help="Borra el namespace y vuelve a subir todo.")
    args = parser.parse_args()

    chequear_variables("PINECONE_API_KEY", "OPENAI_API_KEY")
    crear_indice_si_no_existe(crear_cliente_pinecone())  # por si no se corrió setup_index.py antes
    vectorstore = crear_vectorstore()
    subidos = ingestar(vectorstore, vectorstore.index, reindexar=args.reindexar)
    if subidos:
        esperar_indexado(vectorstore.index, subidos)
