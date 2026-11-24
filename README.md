# RAG híbrido en la nube con Pinecone

Sistema RAG completo sobre documentación técnica: sube los documentos a un índice
**Pinecone Serverless**, los recupera combinando **búsqueda vectorial** (embeddings de
OpenAI) con **búsqueda léxica (BM25)** y genera la respuesta con un LLM usando sólo lo
recuperado (si no está, responde "No lo sé"). Incluye un script de evaluación que mide
**Precision@5** y **Recall@5** con golden sets de preguntas.

```python
from rag_system import RAGSystem

rag = RAGSystem()

docs = rag.buscar("¿Qué pasa con lru_cache si maxsize es None?")         # top-5 chunks (sync)
docs = await rag.abuscar("¿Qué pasa con lru_cache si maxsize es None?")  # top-5 chunks (async)

r = rag.responder("¿Qué pasa con lru_cache si maxsize es None?")         # respuesta generada
print(r.respuesta)   # Si maxsize es None, las características LRU están deshabilitadas...
print(r.fuentes)     # ['functools-p3-c1']
```

## Dataset

En `data/` hay 6 PDFs con la documentación de módulos de la biblioteca estándar de
**Python 3.12** (generada con `pydoc` a partir de los docstrings oficiales):

| Archivo | Páginas | Categoría |
|---|---|---|
| `json.pdf` | 8 | serializacion |
| `csv.pdf` | 7 | serializacion |
| `itertools.pdf` | 9 | programacion-funcional |
| `functools.pdf` | 4 | programacion-funcional |
| `re.pdf` | 11 | texto |
| `datetime.pdf` | 10 | fechas |

Usé PDFs para que cada chunk tenga una página real en la metadata.

## Estructura

```
├── data/                       # los 6 PDFs
├── config.py                   # variables de entorno, embeddings, LLM y conexión a Pinecone
├── setup_index.py              # crea el índice serverless si no existe
├── ingest.py                   # PDFs -> chunks -> embeddings -> Pinecone (con metadata)
├── rag_system.py               # RAGSystem: EnsembleRetriever (BM25 + Pinecone) + generación
├── evaluate.py                 # Precision@5 y Recall@5
├── golden_set.json             # 5 preguntas con su respuesta y documento esperado
├── golden_set_conceptual.json  # 5 preguntas más, sin el nombre de la función
├── tests/                      # tests sin red (no usan Pinecone ni OpenAI)
├── .env.example
└── requirements.txt
```

## Cómo replicar el índice

**1. Instalar** (Python 3.12+). Todas las dependencias, incluidas las de los tests, están
en `requirements.txt`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configurar las variables.** Necesitás una cuenta de [Pinecone](https://app.pinecone.io)
(el plan gratuito alcanza) y una API key de OpenAI.

```bash
cp .env.example .env   # y completá las keys en .env (no en .env.example, que se sube al repo)
```

| Variable | Obligatoria | Default | Descripción |
|---|---|---|---|
| `PINECONE_API_KEY` | sí | – | API key de Pinecone |
| `OPENAI_API_KEY` | sí | – | Para los embeddings y para generar las respuestas |
| `INDEX_NAME` | no | `rag-python-docs` | Nombre del índice |
| `PINECONE_NAMESPACE` | no | `python-docs` | Namespace donde se guardan los vectores |
| `OPENAI_MODEL` | no | `gpt-4.1-mini` | Modelo que genera las respuestas |

**3. Crear el índice** (si ya existe no hace nada, y si existe con otra dimensión avisa):

```bash
python setup_index.py
```

```
Creando el índice 'rag-python-docs' (1536 dimensiones, cosine, aws/us-east-1)...
Índice listo
```

**4. Subir los documentos** (si el índice no existe, lo crea antes):

```bash
python ingest.py               # si el namespace ya tiene datos no vuelve a subir nada
python ingest.py --reindexar   # borra el namespace y sube todo de nuevo
```

```
csv.pdf: 7 chunks
datetime.pdf: 10 chunks
functools.pdf: 4 chunks
itertools.pdf: 9 chunks
json.pdf: 12 chunks
re.pdf: 12 chunks
Subidos 54 chunks al índice 'rag-python-docs', namespace 'python-docs'
Vectores disponibles para buscar
```

Pinecone tarda unos segundos en dejar buscables los vectores recién subidos, así que el
script espera a que estén todos antes de terminar.

**5. Preguntar** (si el namespace está vacío, avisa que falta correr la ingesta):

```bash
python rag_system.py "¿Qué pasa con lru_cache si maxsize es None?"
```

```
Pregunta: ¿Qué pasa con lru_cache si maxsize es None?

Top-5 recuperado (BM25 + Pinecone):
  1. functools-p3-c1 (functools.pdf, pág. 3, programacion-funcional)
  2. datetime-p8-c1 (datetime.pdf, pág. 8, fechas)
  3. json-p5-c2 (json.pdf, pág. 5, serializacion)
  4. re-p9-c1 (re.pdf, pág. 9, texto)
  5. datetime-p9-c1 (datetime.pdf, pág. 9, fechas)

Respuesta: Si maxsize es None, las características LRU están deshabilitadas y la caché puede crecer sin límite.
Fuentes: functools-p3-c1
```

Con una pregunta que no está en los documentos no inventa, aunque el modelo conozca la
respuesta:

```
Pregunta: ¿Cómo hago una petición HTTP con la librería requests?
...
Respuesta: No lo sé
Fuentes: -
```

**6. Evaluar:**

```bash
python evaluate.py                                          # golden_set.json
python evaluate.py --golden-set golden_set_conceptual.json  # preguntas conceptuales
```

**Tests** (no necesitan keys ni conexión):

```bash
pytest
```

## Cómo funciona

### Índice (`setup_index.py`)

Índice **serverless** en `aws` / `us-east-1`, con **dimensión 1536** (la que genera
`text-embedding-3-small`) y métrica **coseno** (para la que están pensados los embeddings
de OpenAI). Antes de crearlo verifica si existe con `list_indexes().names()`, y si existe
compara la dimensión: subir vectores de 1536 a un índice de 768 falla, así que prefiero
que avise antes.

### Ingesta (`ingest.py`)

1. Lee cada PDF **página por página** con `pypdf`.
2. Lo parte con `RecursiveCharacterTextSplitter` en chunks de **hasta 800 tokens con 100
   de overlap**, medidos con `tiktoken`. Como parto página por página, un chunk nunca
   mezcla dos páginas y la página de la metadata es exacta. En promedio quedan de ~520
   tokens (la mayoría de las páginas entra en un solo chunk; los más chicos son las
   últimas páginas de cada PDF).
3. Cada chunk se sube con un **id estable** (`functools-p3-c1`), así reindexar pisa los
   mismos vectores en vez de duplicarlos, y con esta metadata:

```json
{
  "text": "|  Supports wrapping existing descriptors and handles non-descriptor\n     |  callables as instance methods...",
  "fuente": "functools.pdf",
  "modulo": "functools",
  "pagina": 3,
  "categoria": "programacion-funcional",
  "etiquetas": ["python-3.12", "stdlib", "functools"],
  "chunk_id": "functools-p3-c1",
  "tokens": 623
}
```

   El **texto original va en la metadata** (`text`): Pinecone devuelve el fragmento junto
   con el resultado y no hace falta otra base de datos para buscarlo.
4. Todo va a un **namespace** (`python-docs`), así estos documentos no se mezclan con
   otros datos del mismo índice.
5. Si el namespace ya tiene vectores no vuelve a subir nada (no gasta embeddings de
   gusto). Para forzarlo está `--reindexar`.

La subida la hace el `PineconeVectorStore` de LangChain (`add_documents`), que genera
los embeddings y hace el upsert en lotes.

### Recuperador híbrido (`rag_system.py`)

`RAGSystem` encapsula un `EnsembleRetriever` con dos recuperadores:

- **Vectorial (Pinecone):** busca por significado. Encuentra la respuesta aunque la
  pregunta esté en español y la documentación en inglés.
- **BM25 (léxico):** busca por palabras exactas. Es bueno con nombres técnicos como
  `lru_cache`, `DictReader` o `fullmatch`. Le paso un tokenizador propio porque el de por
  defecto separa sólo por espacios, y `lru_cache(maxsize=128,` quedaría como una sola
  palabra.

El ensemble combina los dos rankings con *Reciprocal Rank Fusion*, con pesos **30% BM25
/ 70% vectorial** (abajo, en Resultados, está por qué). Cada recuperador trae 5 y me
quedo con el **top-5** combinado. Tiene `buscar()` (sync) y `abuscar()` (async). BM25
necesita los chunks en memoria, así que `RAGSystem` los arma con la misma función que usa
la ingesta: los dos recuperadores ven exactamente los mismos chunks.

`BM25Retriever` está en `langchain-community` y `EnsembleRetriever` en
`langchain-classic`: en LangChain 1.x salieron del paquete `langchain`.

### Generación (`rag_system.py`)

`responder()` (y `aresponder()`, async) completa el flujo RAG:

1. Recupera el top-5 con el recuperador híbrido.
2. Arma el prompt con los 5 fragmentos, cada uno con su id (`[functools-p3-c1]`).
3. Llama a `gpt-4.1-mini` (temperatura 0). El prompt le pide responder **sólo con el
   contexto** y decir exactamente "No lo sé" si la respuesta no está.
4. Parsea la salida con `PydanticOutputParser` a un `RespuestaRAG` (`respuesta` +
   `fuentes`). Si el modelo no devuelve un JSON válido, se reintenta.
5. Descarta las fuentes que el modelo haya inventado: sólo quedan ids de chunks que
   realmente se recuperaron.

### Evaluación (`evaluate.py`)

Cada golden set tiene 5 preguntas con su respuesta (sacada de la documentación) y el
documento donde está:

```json
{
  "pregunta": "¿Qué pasa con lru_cache si maxsize es None?",
  "respuesta_esperada": "Se desactiva la parte LRU y el caché puede crecer sin límite.",
  "documento_id_esperado": "functools.pdf"
}
```

- `golden_set.json`: preguntas **directas**, que nombran la función (`lru_cache`,
  `loads`, `groupby`, `DictReader`, `fullmatch`).
- `golden_set_conceptual.json`: preguntas **conceptuales**, que describen lo que se
  quiere hacer sin nombrar la función (por ejemplo, "¿Cómo represento la diferencia de
  tiempo entre dos fechas?" para `timedelta`).

Para cada pregunta se recuperan 5 chunks y se calcula:

- **Recall@5:** ¿aparece el documento correcto entre los 5? (sí o no, y después se
  promedia: % de preguntas donde apareció)
- **Precision@5:** ¿qué porcentaje de los 5 viene del documento correcto?

El script compara tres configuraciones (sólo BM25, sólo vectorial e híbrido) e imprime el
detalle de cada pregunta y un resumen:

```
1. ¿Qué pasa con lru_cache si maxsize es None?
   respuesta esperada: Se desactiva la parte LRU y el caché puede crecer sin límite.
   documento esperado: functools.pdf | P@5 = 20% | R@5 = 1 ✓
   top-5 recuperado: functools.pdf, datetime.pdf, json.pdf, re.pdf, datetime.pdf
...
======================================================================
Resumen (promedio de las 5 preguntas)
Configuración                Precision@5       Recall@5
----------------------------------------------------------------------
BM25 (léxico)                        84%           100%
Vectorial (Pinecone)                 84%           100%
Híbrido (Ensemble)                   84%           100%
======================================================================
```

## Resultados

Corrida real: índice serverless en Pinecone (aws/us-east-1), embeddings
`text-embedding-3-small`, 54 chunks, k = 5.

| Configuración | Directas: P@5 | Directas: R@5 | Conceptuales: P@5 | Conceptuales: R@5 |
|---|---|---|---|---|
| BM25 (léxico) | 84% | 100% | 36% | 60% |
| Vectorial (Pinecone) | 84% | 100% | 92% | 100% |
| **Híbrido (30/70)** | **84%** | **100%** | **92%** | **100%** |

Detalle de las preguntas conceptuales (documento de cada chunk del top-5):

| Pregunta (documento esperado) | BM25 | Vectorial | Híbrido |
|---|---|---|---|
| Recordar resultados de llamadas (functools) | itertools, functools, re, re, csv | functools, itertools, itertools, functools, functools | functools, itertools, itertools, functools, functools |
| Elegir r elementos sin orden (itertools) | itertools ×3, csv ×2 | itertools ×5 | itertools ×5 |
| Diferencia entre dos fechas (datetime) | re ×5 ✗ | datetime ×5 | datetime ×5 |
| Coincidencias de un patrón (re) | re ×5 | re ×5 | re ×5 |
| Escribir filas separadas por comas (csv) | functools, re ×4 ✗ | csv ×5 | csv ×5 |

**Qué muestran los resultados:**

- **Con preguntas directas, los tres métodos empatan** (84% / 100%) y el documento
  correcto sale primero en las 5 preguntas: el nombre de la función está en la pregunta,
  así que BM25 lo encuentra por las palabras y los embeddings por el significado. La
  precisión no llega a 100% por la pregunta de `lru_cache` (20%): el chunk exacto con la
  respuesta sale primero, pero `functools.pdf` es corto (4 chunks) y sólo uno habla de
  `lru_cache`, así que el resto del top-5 se completa con otros documentos.
- **Con preguntas conceptuales, BM25 se cae** (36% / 60%): en 2 de 5 preguntas ni
  siquiera encuentra el documento, porque la pregunta está en español y no comparte
  palabras con la documentación en inglés. La búsqueda vectorial funciona igual de bien
  porque compara significados.
- **El híbrido iguala al mejor método en los dos casos.** Es la ventaja de combinarlos:
  no hay que saber de antemano qué tipo de pregunta va a llegar.

**Por qué 30/70 y no 50/50.** Con los pesos iguales, el híbrido daba 84% / 100% en las
directas pero **56% / 100%** en las conceptuales: corregía las fallas de BM25 (el recall
subía a 100%), pero el ruido de BM25 bajaba la precisión respecto del vectorial solo.
Probé 30/70 y dio 84% / 100% en las directas (igual) y 92% / 100% en las conceptuales,
así que lo dejé así:

| Pesos BM25 / vectorial | Directas: P@5 / R@5 | Conceptuales: P@5 / R@5 |
|---|---|---|
| 50 / 50 | 84% / 100% | 56% / 100% |
| **30 / 70** | **84% / 100%** | **92% / 100%** |

Con 10 preguntas es una muestra chica: para elegir los pesos con más seguridad habría que
evaluar sobre un golden set más grande.
