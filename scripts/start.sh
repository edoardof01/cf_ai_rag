#!/usr/bin/env bash
# =============================================================================
# start.sh — Script di avvio completo del sistema RAG
#
# Avvia l'infrastruttura su Minikube (Milvus, Redis, Ollama),
# aspetta che tutto sia pronto, avvia l'API FastAPI in locale,
# e carica automaticamente tutti i PDF dalla cartella docs/.
#
# Uso: ./scripts/start.sh
# =============================================================================

set -e  # Esce al primo errore

# Colori per l'output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DOCS_DIR="$PROJECT_DIR/docs"
API_URL="http://localhost:8000"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════╗"
echo "║           🚀 RAG System — Avvio Completo        ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ─────────────────────────────────────────────────────
# FASE 1: Minikube
# ─────────────────────────────────────────────────────
echo -e "${YELLOW}[1/6] Verifico Minikube...${NC}"
if minikube status | grep -q "Running"; then
    echo -e "${GREEN}  ✅ Minikube già in esecuzione${NC}"
else
    echo "  Avvio Minikube..."
    minikube start --memory=6144 --cpus=4
    echo -e "${GREEN}  ✅ Minikube avviato${NC}"
fi

# Abilita l'Ingress controller (se non già attivo)
minikube addons enable ingress 2>/dev/null || true

# ─────────────────────────────────────────────────────
# FASE 2: Deploy infrastruttura su K8s
# ─────────────────────────────────────────────────────
echo -e "${YELLOW}[2/6] Deploy infrastruttura Kubernetes...${NC}"

echo "  Applicando Redis..."
kubectl apply -f "$PROJECT_DIR/k8s/redis.yaml"

echo "  Applicando Milvus (etcd + MinIO + standalone)..."
kubectl apply -f "$PROJECT_DIR/k8s/milvus.yaml"

echo "  Applicando Ollama..."
kubectl apply -f "$PROJECT_DIR/k8s/ollama.yaml"

echo -e "${GREEN}  ✅ Manifest applicati${NC}"

# ─────────────────────────────────────────────────────
# FASE 3: Aspetta che i pod siano Ready
# ─────────────────────────────────────────────────────
echo -e "${YELLOW}[3/6] Attendo che i pod siano pronti (può richiedere 2-3 minuti)...${NC}"

echo "  Aspettando Redis..."
kubectl wait --for=condition=available deployment/redis --timeout=120s

echo "  Aspettando Milvus etcd..."
kubectl wait --for=condition=available deployment/milvus-etcd --timeout=120s

echo "  Aspettando Milvus MinIO..."
kubectl wait --for=condition=available deployment/milvus-minio --timeout=120s

echo "  Aspettando Milvus standalone..."
kubectl wait --for=condition=available deployment/milvus-standalone --timeout=180s

echo "  Aspettando Ollama..."
kubectl wait --for=condition=available deployment/ollama --timeout=180s

echo -e "${GREEN}  ✅ Tutti i pod sono pronti${NC}"

# ─────────────────────────────────────────────────────
# FASE 4: Port-forwarding (background)
# ─────────────────────────────────────────────────────
echo -e "${YELLOW}[4/6] Attivo port-forwarding...${NC}"

# Uccidi eventuali port-forward precedenti
pkill -f "kubectl port-forward.*milvus" 2>/dev/null || true
pkill -f "kubectl port-forward.*redis" 2>/dev/null || true
pkill -f "kubectl port-forward.*ollama" 2>/dev/null || true
sleep 1

# Avvia i port-forward in background
kubectl port-forward svc/milvus-standalone-service 19530:19530 &>/dev/null &
PF_MILVUS=$!

kubectl port-forward svc/redis-service 6379:6379 &>/dev/null &
PF_REDIS=$!

kubectl port-forward svc/ollama-service 11434:11434 &>/dev/null &
PF_OLLAMA=$!

sleep 2  # Breve attesa per il setup dei port-forward

# Verifica connettività Milvus (TCP probe — leggero e veloce)
echo "  Verifico connettività Milvus..."
MILVUS_READY=false
for i in $(seq 1 40); do
    if (echo > /dev/tcp/localhost/19530) 2>/dev/null; then
        MILVUS_READY=true
        break
    fi
    sleep 3
done
if [ "$MILVUS_READY" = false ]; then
    echo -e "${RED}  ❌ Milvus non raggiungibile dopo 2 minuti.${NC}"
    exit 1
fi
echo "  ✓ Milvus raggiungibile"

# Verifica connettività Redis
echo "  Verifico connettività Redis..."
for i in $(seq 1 20); do
    if (echo > /dev/tcp/localhost/6379) 2>/dev/null; then
        break
    fi
    sleep 2
done
echo "  ✓ Redis raggiungibile"

echo -e "${GREEN}  ✅ Port-forward attivi e backends raggiungibili:"
echo "     Milvus  → localhost:19530"
echo "     Redis   → localhost:6379"
echo -e "     Ollama  → localhost:11434${NC}"

# Attesa extra: Milvus accetta connessioni TCP prima di essere pronto a servire gRPC.
# Diamo 10 secondi di margine per l'inizializzazione interna.
echo "  Attendo inizializzazione interna Milvus..."
sleep 10

# ─────────────────────────────────────────────────────
# FASE 5: Pull del modello Qwen (se non presente)
# ─────────────────────────────────────────────────────
echo -e "${YELLOW}[5/6] Verifico modello LLM in Ollama...${NC}"

# Esporta OLLAMA_HOST per tutti i comandi successivi
export OLLAMA_HOST="http://localhost:11434"

# Aspetta che Ollama sia raggiungibile via port-forward
echo "  Attendo che Ollama sia raggiungibile..."
OLLAMA_READY=false
for i in $(seq 1 60); do
    if curl --max-time 3 -sf "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
        OLLAMA_READY=true
        break
    fi
    sleep 3
done

if [ "$OLLAMA_READY" = false ]; then
    echo -e "${RED}  ❌ Ollama non raggiungibile dopo 3 minuti.${NC}"
    exit 1
fi

# Verifica se il modello è già presente (via API REST, non CLI)
if curl --max-time 5 -sf "$OLLAMA_HOST/api/tags" 2>/dev/null | grep -q "qwen2.5:7b"; then
    echo -e "${GREEN}  ✅ qwen2.5:7b già disponibile${NC}"
else
    echo "  Download di qwen2.5:7b in corso (4.7 GB, potrebbe richiedere qualche minuto)..."
    curl -sf "$OLLAMA_HOST/api/pull" -d '{"name": "qwen2.5:7b"}' --no-buffer | while read -r line; do
        STATUS=$(echo "$line" | grep -o '"status":"[^"]*"' | head -1)
        if echo "$line" | grep -q "pulling"; then
            echo -ne "\r  $STATUS"
        fi
    done
    echo ""
    echo -e "${GREEN}  ✅ qwen2.5:7b scaricato${NC}"
fi

# ─────────────────────────────────────────────────────
# FASE 6: Avvio FastAPI + Caricamento PDF
# ─────────────────────────────────────────────────────
echo -e "${YELLOW}[6/6] Avvio server FastAPI...${NC}"

# Avvia il server in background
cd "$PROJECT_DIR"
$VENV_PYTHON -m uvicorn src.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Attendi che il server sia pronto (può richiedere 30-60s per caricare embedder + cross-encoder)
echo "  Attendo che FastAPI sia pronto (caricamento modelli in corso)..."
SERVER_READY=false
for i in $(seq 1 30); do
    if curl --max-time 3 -sf "$API_URL/health" >/dev/null 2>&1; then
        SERVER_READY=true
        break
    fi
    sleep 3
done

if [ "$SERVER_READY" = false ]; then
    echo -e "${RED}  ❌ Il server FastAPI non risponde dopo 90 secondi. Controlla i log sopra.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ FastAPI in ascolto su $API_URL${NC}"

# ─────────────────────────────────────────────────────
# Caricamento automatico PDF
# ─────────────────────────────────────────────────────
PDF_COUNT=$(find "$DOCS_DIR" -type f -name "*.pdf" 2>/dev/null | wc -l)

if [ "$PDF_COUNT" -gt 0 ]; then
    echo ""
    echo -e "${CYAN}📚 Trovati $PDF_COUNT PDF in docs/. Inizio caricamento...${NC}"
    echo ""
    
    LOADED=0
    FAILED=0

    find "$DOCS_DIR" -type f -name "*.pdf" | while read -r pdf; do
        BASENAME=$(basename "$pdf")
        RESPONSE=$(curl -sf -X POST "$API_URL/upload-pdf" \
            -H "accept: application/json" \
            -F "file=@$pdf" 2>&1) || true

        if echo "$RESPONSE" | grep -q '"status":"ok"'; then
            CHUNKS=$(echo "$RESPONSE" | grep -o '"chunks_indexed":[0-9]*' | grep -o '[0-9]*')
            echo -e "  ${GREEN}✅${NC} $BASENAME → $CHUNKS chunks"
        else
            echo -e "  ${RED}❌${NC} $BASENAME — Errore: $RESPONSE"
        fi
    done

    echo ""
    echo -e "${GREEN}📚 Caricamento completato!${NC}"
else
    echo -e "${YELLOW}  ⚠️  Nessun PDF trovato in $DOCS_DIR${NC}"
fi

# ─────────────────────────────────────────────────────
# Riepilogo finale
# ─────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗"
echo "║           🎉 Sistema RAG PRONTO!                ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                  ║"
echo "║  API:     http://localhost:8000                  ║"
echo "║  Health:  http://localhost:8000/health            ║"
echo "║  Docs:    http://localhost:8000/docs (Swagger)   ║"
echo "║                                                  ║"
echo "║  Esempio:                                        ║"
echo "║  curl -X POST http://localhost:8000/ask \\        ║"
echo "║    -H 'Content-Type: application/json' \\        ║"
echo "║    -d '{\"question\": \"Ciao!\"}' │ jq             ║"
echo "║                                                  ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Per fermare tutto: ./scripts/stop.sh            ║"
echo -e "╚══════════════════════════════════════════════════╝${NC}"

# Tieni il server in primo piano (CTRL+C per uscire)
wait $API_PID
