# Advanced Hybrid RAG System

Un sistema Retrieval-Augmented Generation (RAG) di livello production-ready, implementato con architettura a microservizi su **Kubernetes (Minikube)**. Il sistema è progettato per interrogare un corpus di documenti (PDF) sfruttando tecniche avanzate di retrieval ibrido, re-ranking e memoria conversazionale multi-strategia.

Tutto lo stack è completamente **self-hosted** e non si affida ad API esterne a pagamento.

---

## 🏗 Architettura e Tecnologie

L'infrastruttura è orchestrata tramite Kubernetes e comprende i seguenti componenti:

- **FastAPI**: Backend principale che espone le API REST e coordina la pipeline RAG.
- **Milvus**: Vector Database ad alte prestazioni per la ricerca semantica densa.
- **Redis**: In-memory data store utilizzato per la ricerca lessicale (BM25) e per la persistenza della memoria conversazionale.
- **Ollama**: Motore locale per l'esecuzione del Large Language Model (LLM) **Qwen 2.5 (7B)**.
- **Sentence-Transformers**: Modelli di embedding locale (`all-MiniLM-L6-v2`) e Re-Ranking Cross-Encoder (`mmarco-mMiniLMv2-L12-H384-v1`).

---

## 🧠 Scelte Architetturali e Tecniche

Il sistema non si limita a un semplice approccio "retrieve-and-generate", ma implementa una pipeline sofisticata per massimizzare la precisione e il contesto:

### 1. Retrieval Ibrido (Semantico + Lessicale BM25)
La ricerca puramente vettoriale fatica su keyword esatte o codici specifici. Questo RAG implementa un **Doppio Retrieval**:
- **Ricerca Semantica (Dense):** I chunk di testo vengono embeddati e salvati in **Milvus**. Eccelle nel cogliere il significato generale della query.
- **Ricerca Lessicale (Sparse):** Implementazione di **BM25** appoggiata su **Redis**. Eccelle nel trovare corrispondenze esatte di parole chiave (es. nomi propri, acronimi, ID).

### 2. Reciprocal Rank Fusion (RRF) e Cross-Encoder Re-Ranking
Come vengono combinati i risultati dei due retrieval?
1. **RRF (Reciprocal Rank Fusion):** I risultati di Milvus e BM25 vengono normalizzati e uniti. Un documento trovato da entrambi i motori riceve un punteggio molto alto, mitigando le debolezze di ciascun approccio.
2. **Re-Ranking (Cross-Encoder):** L'RRF è un'euristica. Per un ranking accurato finale, i top-N documenti passano attraverso un **Cross-Encoder** basato su RoBERTa. Questo modello valuta la pertinenza effettiva *coppia per coppia* (Query $\leftrightarrow$ Documento) restituendo i chunk definitivi per la generazione.

### 3. Memoria Conversazionale Multi-Strategia
Il sistema non "dimentica" il contesto e non accumula semplicemente l'intera chat (causando il superamento della context window). La memoria è tripartita:
- **Finestra Scorrevole (KV):** Mantiene gli ultimi $W$ turni per il contesto immediato.
- **Riassunto Cumulativo (Summary):** Ogni $N$ turni, l'LLM sintetizza la conversazione finora, persistendola in Redis. Utile per il contesto a lungo termine.
- **Memoria Vettoriale (VM):** I turni passati vengono embeddati e archiviati in una collection Milvus dedicata. Permette di recuperare fatti discussi 100 turni prima tramite similarità semantica.

### 4. Query Augmentation / Rewriting
Se l'utente chiede *"Quali sono i requisiti per questa posizione?"*, una ricerca diretta fallirebbe. Il sistema esegue un **Query Rewriting**: l'LLM analizza la memoria e riscrive la query in modo standalone (es. *"Quali sono i requisiti per la posizione di Senior Backend Engineer presso l'azienda X?"*) *prima* di interrogarlo nel database.

### 5. Metadata-Aware Chunking
Durante l'ingestion, il nome del file (che contiene metadati cruciali come ruolo, azienda, location e modalità remote/hybrid) viene prefissato al testo di ogni chunk (`[nome_file.pdf]\nTesto...`). Questo permette al VectorDB e al BM25 di ritrovare i documenti basandosi non solo sul contenuto, ma anche sui metadati strutturali del file.

---

## 🚀 Guida all'Avvio

Il progetto include uno script di bootstrap completamente automatizzato.

### Requisiti
- `minikube` e `kubectl`
- Python 3.12+ (per l'ambiente virtuale)

### Avvio del Cluster
Lancia lo script principale:
```bash
./scripts/start.sh
```
Questo script si occupa di:
1. Avviare Minikube.
2. Applicare i manifest Kubernetes (`k8s/`).
3. Attendere la salute dei Pod e instaurare i `port-forward` in background.
4. Fare il pull automatico del modello LLM in Ollama.
5. Lanciare il server FastAPI e ricaricare automaticamente i PDF presenti nella cartella `docs/`.

### Arresto
Per fermare i port-forward e FastAPI in modo pulito:
```bash
./scripts/stop.sh
```
Per distruggere il cluster: `minikube delete`.

---

## 🔌 API Endpoints

- **Generazione Risposta (RAG)**
  ```bash
  curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "Quali aziende offrono posizioni remote?"}'
  ```

- **Caricamento di un nuovo documento**
  ```bash
  curl -X POST http://localhost:8000/upload-pdf \
    -F "file=@docs/Careers/mio_documento.pdf"
  ```
