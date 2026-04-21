#!/usr/bin/env bash
# =============================================================================
# stop.sh — Ferma tutto il sistema RAG
#
# Uccide il server FastAPI, i port-forward e (opzionalmente) Minikube.
#
# Uso: ./scripts/stop.sh
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Arresto sistema RAG...${NC}"

# Ferma il server FastAPI
echo "  Fermo FastAPI..."
pkill -f "uvicorn src.api:app" 2>/dev/null || true

# Ferma tutti i port-forward
echo "  Fermo port-forward..."
pkill -f "kubectl port-forward" 2>/dev/null || true

echo -e "${GREEN}✅ Sistema RAG fermato.${NC}"
echo ""
echo "I pod su Minikube sono ancora attivi (i dati sono persistenti)."
echo "Per spegnere anche Minikube: minikube stop"
