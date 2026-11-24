"""Evaluación del recuperador con el golden set: Precision@5 y Recall@5.

    python evaluate.py                                          # golden_set.json (5 preguntas)
    python evaluate.py --golden-set golden_set_conceptual.json  # preguntas sin el nombre de la función

Compara tres configuraciones: sólo BM25, sólo vectorial (Pinecone) e híbrido (los dos).
"""

import argparse
import asyncio
import json
from pathlib import Path

from config import chequear_variables
from rag_system import TOP_K, RAGSystem

GOLDEN_SET = Path(__file__).parent / "golden_set.json"


def precision_at_k(fuentes: list[str], esperado: str, k: int = TOP_K) -> float:
    """Qué parte de los k recuperados viene del documento correcto."""
    return sum(f == esperado for f in fuentes[:k]) / k


def recall_at_k(fuentes: list[str], esperado: str, k: int = TOP_K) -> float:
    """1 si el documento correcto aparece entre los k recuperados, 0 si no (hay un solo documento relevante)."""
    return 1.0 if esperado in fuentes[:k] else 0.0


async def evaluar(buscar, golden_set: list[dict]) -> list[dict]:
    """Corre las preguntas y calcula las métricas de cada una."""
    filas = []
    for item in golden_set:
        # una por una y no con gather: el PineconeVectorStore comparte su cliente async
        # entre consultas y lo cierra al terminar cada una ("Session is closed")
        docs = await buscar(item["pregunta"])
        fuentes = [d.metadata["fuente"] for d in docs]
        filas.append(
            {
                "pregunta": item["pregunta"],
                "respuesta_esperada": item["respuesta_esperada"],
                "esperado": item["documento_id_esperado"],
                "fuentes": fuentes,
                "precision": precision_at_k(fuentes, item["documento_id_esperado"]),
                "recall": recall_at_k(fuentes, item["documento_id_esperado"]),
            }
        )
    return filas


def promedio(filas: list[dict], metrica: str) -> float:
    return sum(f[metrica] for f in filas) / len(filas)


def imprimir_detalle(filas: list[dict]) -> None:
    for i, f in enumerate(filas, start=1):
        ok = "✓" if f["recall"] else "✗"
        print(f"\n{i}. {f['pregunta']}")
        print(f"   respuesta esperada: {f['respuesta_esperada']}")
        print(f"   documento esperado: {f['esperado']} | P@5 = {f['precision']:.0%} | R@5 = {f['recall']:.0f} {ok}")
        print(f"   top-5 recuperado: {', '.join(f['fuentes'])}")


async def main(golden_set_path: Path) -> None:
    chequear_variables("PINECONE_API_KEY", "OPENAI_API_KEY")
    golden_set = json.loads(golden_set_path.read_text(encoding="utf-8"))
    rag = RAGSystem()
    rag.chequear_indice_cargado()

    configuraciones = {
        "BM25 (léxico)": rag.bm25.ainvoke,
        "Vectorial (Pinecone)": rag.vectorial.ainvoke,
        "Híbrido (Ensemble)": rag.abuscar,
    }
    resumen = {nombre: await evaluar(buscar, golden_set) for nombre, buscar in configuraciones.items()}

    print("=" * 70)
    print(f"Golden set: {golden_set_path.name}")
    print(f"Detalle del recuperador híbrido ({len(golden_set)} preguntas, k={TOP_K})")
    print("=" * 70)
    imprimir_detalle(resumen["Híbrido (Ensemble)"])

    print("\n" + "=" * 70)
    print(f"Resumen (promedio de las {len(golden_set)} preguntas)")
    print(f"{'Configuración':<25}{'Precision@5':>15}{'Recall@5':>15}")
    print("-" * 70)
    for nombre, filas in resumen.items():
        print(f"{nombre:<25}{promedio(filas, 'precision'):>15.0%}{promedio(filas, 'recall'):>15.0%}")
    print("=" * 70)
    print("Precision@5: qué porcentaje de los 5 chunks recuperados viene del documento correcto.")
    print("Recall@5: en qué porcentaje de las preguntas el documento correcto aparece en el top-5.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa el recuperador con Precision@5 y Recall@5.")
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET, help="JSON con las preguntas de prueba.")
    asyncio.run(main(parser.parse_args().golden_set))
