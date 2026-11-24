"""Tests sin red: en lugar de Pinecone y OpenAI uso un InMemoryVectorStore de LangChain con embeddings de prueba
y un chat model falso de LangChain."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.vectorstores import InMemoryVectorStore

from config import DATA_DIR, DIMENSION, NAMESPACE
from evaluate import GOLDEN_SET, evaluar, precision_at_k, recall_at_k

GOLDEN_SET_CONCEPTUAL = GOLDEN_SET.parent / "golden_set_conceptual.json"
from ingest import CATEGORIAS, CHUNK_SIZE, cargar_chunks, cargar_paginas, ingestar
from rag_system import NO_LO_SE, RAGSystem, tokenizar
from setup_index import crear_indice_si_no_existe


class EmbeddingsDePrueba(Embeddings):
    """Vectores de trigramas de caracteres, de 1536 dimensiones como text-embedding-3-small."""

    def _vector(self, texto: str) -> list[float]:
        v = np.zeros(DIMENSION)
        t = f"  {texto.lower()} "
        for i in range(len(t) - 2):
            v[int(hashlib.md5(t[i : i + 3].encode()).hexdigest(), 16) % DIMENSION] += 1
        return (v / (np.linalg.norm(v) or 1)).tolist()

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture(scope="module")
def chunks():
    return cargar_chunks()


@pytest.fixture(scope="module")
def vectorstore(chunks):
    vs = InMemoryVectorStore(EmbeddingsDePrueba())
    vs.add_documents(chunks, ids=[c.id for c in chunks])
    return vs


def llm_falso(*respuestas) -> FakeListChatModel:
    return FakeListChatModel(responses=[r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in respuestas])


def crear_rag(vectorstore, chunks, llm) -> RAGSystem:
    # los embeddings de prueba no entienden significado, así que acá le doy más peso a BM25
    # (con los embeddings reales de OpenAI se usa el default, 30/70)
    return RAGSystem(vectorstore=vectorstore, chunks=chunks, llm=llm, pesos=(0.7, 0.3))


@pytest.fixture(scope="module")
def rag(vectorstore, chunks):
    return crear_rag(vectorstore, chunks, llm_falso({"respuesta": NO_LO_SE}))


def stats(vectores: int):
    namespaces = {NAMESPACE: SimpleNamespace(vector_count=vectores)} if vectores else {}
    return SimpleNamespace(namespaces=namespaces)


# --- Setup del índice --------------------------------------------------------


def test_crea_el_indice_serverless_si_no_existe():
    pc = MagicMock()
    pc.list_indexes.return_value.names.return_value = []

    assert crear_indice_si_no_existe(pc, "mi-indice") is True
    kwargs = pc.create_index.call_args.kwargs
    assert (kwargs["name"], kwargs["dimension"], kwargs["metric"]) == ("mi-indice", 1536, "cosine")
    assert (kwargs["spec"].cloud, kwargs["spec"].region) == ("aws", "us-east-1")


def test_no_crea_el_indice_si_ya_existe():
    pc = MagicMock()
    pc.list_indexes.return_value.names.return_value = ["mi-indice"]
    pc.describe_index.return_value.dimension = 1536

    assert crear_indice_si_no_existe(pc, "mi-indice") is False
    pc.create_index.assert_not_called()


def test_avisa_si_la_dimension_no_coincide():
    pc = MagicMock()
    pc.list_indexes.return_value.names.return_value = ["mi-indice"]
    pc.describe_index.return_value.dimension = 768

    with pytest.raises(SystemExit, match="768"):
        crear_indice_si_no_existe(pc, "mi-indice")


# --- Ingesta -----------------------------------------------------------------


def test_paginas_con_fuente_pagina_y_categoria():
    paginas = cargar_paginas()
    assert {p.metadata["fuente"] for p in paginas} == {p.name for p in DATA_DIR.glob("*.pdf")}
    for p in paginas:
        assert p.metadata["pagina"] >= 1
        assert p.metadata["categoria"] == CATEGORIAS[p.metadata["fuente"]]
        assert "stdlib" in p.metadata["etiquetas"]


def test_chunks_con_tamanio_razonable_e_ids_estables(chunks):
    tokens = [c.metadata["tokens"] for c in chunks]
    assert max(tokens) <= CHUNK_SIZE
    assert 500 <= sum(tokens) / len(tokens) <= 800
    assert len({c.id for c in chunks}) == len(chunks)
    assert [c.id for c in cargar_chunks()] == [c.id for c in chunks]  # mismos ids en cada corrida


def test_ingesta_sube_los_chunks_con_texto_y_metadata(chunks):
    vectorstore = InMemoryVectorStore(EmbeddingsDePrueba())
    index = MagicMock()
    index.describe_index_stats.return_value = stats(0)

    assert ingestar(vectorstore, index) == len(chunks)
    guardado = vectorstore.get_by_ids(["functools-p3-c1"])[0]
    assert "lru_cache" in guardado.page_content
    assert guardado.metadata["fuente"] == "functools.pdf" and guardado.metadata["pagina"] == 3


def test_no_reingesta_si_el_namespace_ya_tiene_datos():
    vectorstore = MagicMock()
    index = MagicMock()
    index.describe_index_stats.return_value = stats(54)

    assert ingestar(vectorstore, index) == 0
    vectorstore.add_documents.assert_not_called()


def test_reindexar_borra_el_namespace_y_vuelve_a_subir(chunks):
    vectorstore = MagicMock()
    index = MagicMock()
    index.describe_index_stats.return_value = stats(54)

    assert ingestar(vectorstore, index, reindexar=True) == len(chunks)
    index.delete.assert_called_once_with(delete_all=True, namespace=NAMESPACE)


# --- Recuperador híbrido -----------------------------------------------------


def test_tokenizar_separa_nombres_pegados_a_simbolos():
    assert tokenizar("lru_cache(maxsize=128, typed=False)") == ["lru_cache", "maxsize", "128", "typed", "false"]


def test_buscar_devuelve_top_5_con_metadata(rag):
    consulta = "¿Qué pasa con lru_cache si maxsize es None?"
    docs = rag.buscar(consulta)
    assert len(docs) == 5
    assert all({"fuente", "pagina", "categoria"} <= d.metadata.keys() for d in docs)
    # BM25 encuentra el nombre exacto "lru_cache" aunque la pregunta esté en español
    assert rag.bm25.invoke(consulta)[0].id == "functools-p3-c1"
    assert "functools-p3-c1" in [d.id for d in docs]


async def test_el_ensemble_combina_bm25_y_vectorial(rag):
    consulta = "fullmatch pattern string"
    ids_bm25 = {d.id for d in await rag.bm25.ainvoke(consulta)}
    ids_vector = {d.id for d in await rag.vectorial.ainvoke(consulta)}
    ids_hibrido = {d.id for d in await rag.abuscar(consulta)}
    assert len(ids_hibrido) == 5
    assert ids_hibrido <= ids_bm25 | ids_vector
    assert ids_hibrido == {d.id for d in rag.buscar(consulta)}  # la versión sync y la async devuelven lo mismo


def test_avisa_si_el_indice_esta_vacio(rag, monkeypatch):
    monkeypatch.setattr(rag.vectorstore, "index", MagicMock(), raising=False)  # el de memoria no tiene .index
    monkeypatch.setattr("rag_system.vectores_en_namespace", lambda index: 0)
    with pytest.raises(SystemExit, match="ingest.py"):
        rag.chequear_indice_cargado()


# --- Generación --------------------------------------------------------------


def test_responder_usa_los_chunks_recuperados_y_descarta_fuentes_inventadas(vectorstore, chunks):
    llm = llm_falso({"respuesta": "Se desactiva el LRU y el caché crece sin límite.", "fuentes": ["functools-p3-c1", "inventado-p9-c9"]})
    rag = crear_rag(vectorstore, chunks, llm)

    r = rag.responder("¿Qué pasa con lru_cache si maxsize es None?")
    assert r.encontrada
    assert r.fuentes == ["functools-p3-c1"]


async def test_aresponder_no_lo_se_deja_las_fuentes_vacias(vectorstore, chunks):
    llm = llm_falso({"respuesta": NO_LO_SE, "fuentes": ["functools-p3-c1"]})
    rag = crear_rag(vectorstore, chunks, llm)

    r = await rag.aresponder("¿Cuál es la capital de Francia?")
    assert not r.encontrada
    assert r.fuentes == []


def test_responder_reintenta_si_el_modelo_no_devuelve_json(vectorstore, chunks):
    llm = llm_falso("Respuesta sin JSON", {"respuesta": "Devuelve None.", "fuentes": []})
    rag = crear_rag(vectorstore, chunks, llm)

    assert rag.responder("¿Qué devuelve fullmatch si no coincide?").respuesta == "Devuelve None."


# --- Evaluación --------------------------------------------------------------


def test_metricas():
    fuentes = ["re.pdf", "re.pdf", "json.pdf", "re.pdf", "csv.pdf"]
    assert precision_at_k(fuentes, "re.pdf") == 0.6
    assert recall_at_k(fuentes, "re.pdf") == 1.0
    assert precision_at_k(fuentes, "datetime.pdf") == 0.0
    assert recall_at_k(fuentes, "datetime.pdf") == 0.0


@pytest.mark.parametrize("archivo", [GOLDEN_SET, GOLDEN_SET_CONCEPTUAL])
def test_golden_sets_tienen_5_preguntas_con_documentos_que_existen(archivo):
    golden = json.loads(archivo.read_text(encoding="utf-8"))
    assert len(golden) == 5
    for g in golden:
        assert g["pregunta"] and g["respuesta_esperada"]
        assert (DATA_DIR / g["documento_id_esperado"]).exists()


def test_las_preguntas_conceptuales_no_nombran_la_funcion():
    golden = json.loads(GOLDEN_SET_CONCEPTUAL.read_text(encoding="utf-8"))
    funciones = ["cache", "combinations", "timedelta", "findall", "writer", "lru_cache"]
    assert not any(f in g["pregunta"].lower() for g in golden for f in funciones)


async def test_evaluar_con_el_golden_set(rag):
    golden = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    filas = await evaluar(rag.abuscar, golden)
    assert len(filas) == 5
    assert all(0 <= f["precision"] <= 1 and f["recall"] in (0, 1) for f in filas)
