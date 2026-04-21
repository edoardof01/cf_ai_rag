# Usa l'immagine ufficiale di Python
FROM python:3.12-slim

# Imposta la directory di lavoro nel container
WORKDIR /app

# Installa dipendenze di sistema utili (es. curl per healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia solo il file dei requisiti prima, per sfruttare la cache di Docker
# copia il file requirements.txt in /app/requirements.txt
COPY requirements.txt .

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Scarica il modello di embedding durante la build dell'immagine.
# In questo modo il modello da ~80MB viene "congelato" nell'immagine Docker
# e il container non dovrà riscaricarlo da internet ogni volta che si avvia.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copia il resto del codice sorgente nel container
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY main.py .

# Variabili d'ambiente di default (verranno sovrascritte da Kubernetes/Docker)
ENV EMBEDDING_MODEL="all-MiniLM-L6-v2"
ENV MILVUS_HOST="localhost"
ENV MILVUS_PORT="19530"
ENV MILVUS_COLLECTION="rag_documents"
ENV OLLAMA_BASE_URL="http://localhost:11434"
ENV OLLAMA_MODEL="llama3"

# Esponi la porta dell'API
EXPOSE 8000

# Il comando di default quando il container viene avviato
# uvicorn è un server ASGI, necessario per eseguire applicazioni Python che usano asyncio.
# src.api:app indica che l'applicazione si trova nel modulo src.api e la variabile app.
# --host "0.0.0.0" indica: ascolta su tutte le interfacce di rete del container (necessario per Kubernetes).
# In Kubernetes, le request arriveranno dal Service K8s all'indirizzo interno del Pod.
# --port 8000 indica che l'app deve essere accessibile sulla porta 8000.
# --reload ricarica l'app automaticamente quando il codice cambia.
# Per produzione si usa gunicorn.
# --no-access-log disabilita i log di accesso, che non sono necessari in produzione.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
