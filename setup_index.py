"""Crea el índice serverless de Pinecone si no existe.

    python setup_index.py
"""

from config import CLOUD, DIMENSION, INDEX_NAME, METRIC, REGION, chequear_variables, crear_cliente_pinecone


def crear_indice_si_no_existe(pc, index_name: str = INDEX_NAME, dimension: int = DIMENSION) -> bool:
    """Devuelve True si lo creó y False si ya existía."""
    from pinecone import ServerlessSpec

    if index_name in pc.list_indexes().names():
        dimension_actual = pc.describe_index(index_name).dimension
        if dimension_actual != dimension:
            # el error #1: subir vectores de una dimensión a un índice de otra
            raise SystemExit(
                f"El índice '{index_name}' tiene dimensión {dimension_actual} y el modelo de embeddings "
                f"genera {dimension}. Usá otro INDEX_NAME o borrá el índice."
            )
        print(f"El índice '{index_name}' ya existe ({dimension_actual} dimensiones), no hago nada")
        return False

    print(f"Creando el índice '{index_name}' ({dimension} dimensiones, {METRIC}, {CLOUD}/{REGION})...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )  # espera a que el índice esté listo
    print("Índice listo")
    return True


if __name__ == "__main__":
    chequear_variables("PINECONE_API_KEY")
    crear_indice_si_no_existe(crear_cliente_pinecone())
