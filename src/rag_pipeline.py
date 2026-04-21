"""
rag_pipeline.py — Orchestratore della pipeline RAG.

Questo modulo collega tutti i componenti:
  1. Riceve la domanda dell'utente
  2. La converte in un vettore (embedding)
  3. Applica la funzione φ(M_{t-1}) per arricchire la query (se memoria presente)
  4. Doppio retrieval: cerca nella Knowledge Base (Milvus) e nella Vector Memory
  5. Token Budget: verifica che il contesto totale rientri nella context window
  6. Costruisce un prompt con il contesto unificato + cronologia KV
  7. Invia il prompt al LLM per generare la risposta (generation)
  8. Aggiorna la memoria conversazionale

Formalismo:
    q'_t = q_t ⊕ φ(M_{t-1})
    C_t  = Retrieve_KB(q'_t) ∪ Retrieve_VM(q_t)
    |C_t| + |KV_{t-1}| + |q'_t| ≤ L_max
    R_t  = LLM(C_t, q_t, KV_{t-1})
    M_t  = Update(M_{t-1}, q_t, R_t)
"""

from src.embedder import Embedder
from src.vector_store import MilvusVectorStore
from src.bm25_store import BM25Store
from src.reranker import CrossEncoderReranker
from src.llm_client import OllamaClient
from src.memory import ConversationMemory, estimate_tokens


# Template del prompt RAG CON memoria conversazionale.
# Include una sezione per la cronologia KV (se disponibile).
RAG_PROMPT_TEMPLATE = """Sei un assistente esperto di analisi del mercato del lavoro tech. Rispondi alle domande basandoti sui documenti forniti nel contesto.

ISTRUZIONI:
1. Ogni chunk di contesto è preceduto dal nome del file tra parentesi quadre [nome_file.pdf]. Il nome del file contiene informazioni preziose: azienda, ruolo, città e modalità (Remote/Hybrid).
2. Usa TUTTE le informazioni disponibili: sia il testo del documento sia i metadati nel nome del file.
3. Quando la domanda riguarda più documenti, sintetizza le informazioni in una risposta completa e strutturata.
4. Cita sempre le fonti indicando il nome del file e il numero di pagina.
5. Se davvero non trovi alcuna informazione rilevante nei documenti, dillo chiaramente.
6. Rispondi in italiano.

{history_section}
=== CONTESTO ===
{context}
=== FINE CONTESTO ===

DOMANDA: {question}

RISPOSTA:"""

# Token usati dal template fisso (istruzioni di sistema, tag === CONTESTO === ecc.)
_TEMPLATE_OVERHEAD_TOKENS = estimate_tokens(RAG_PROMPT_TEMPLATE.replace("{history_section}", "")
                                            .replace("{context}", "")
                                            .replace("{question}", ""))


class RAGPipeline:
    """Pipeline completa: query → retrieval → generation (con memoria e token budget)."""

    def __init__(self, embedder: Embedder, vector_store: MilvusVectorStore,
                 bm25_store: BM25Store, reranker: CrossEncoderReranker,
                 llm_client: OllamaClient, top_k: int = 5, 
                 max_context_tokens: int = 6000):
        """
        Args:
            embedder: Modello di embedding.
            vector_store: Client Milvus per la Knowledge Base (Semantic).
            bm25_store: Indice BM25 per la Knowledge Base (Lexical).
            reranker: Modello Cross-Encoder per il re-ranking finale.
            llm_client: Client Ollama per la generazione.
            top_k: Numero massimo di chunk da recuperare dalla KB.
            max_context_tokens: Budget massimo di token per il contesto
                                (esclude i ~2000 token riservati alla risposta LLM).
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.reranker = reranker
        self.llm_client = llm_client
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens

    def query(self, question: str, memory: ConversationMemory | None = None) -> str:
        """
        Esegue una query RAG completa, con supporto opzionale alla memoria
        e gestione del token budget.

        Args:
            question: La domanda dell'utente (q_t).
            memory: Istanza di ConversationMemory (None = nessuna memoria).

        Returns:
            La risposta generata dal LLM.
        """
        # Step 1: Embedding della domanda originale (q_t)
        query_embedding = self.embedder.embed_text(question)

        # Step 2: Query extension con φ(M_{t-1})
        if memory and not memory.is_empty():
            phi_text = memory.compute_phi(question, query_embedding)
            augmented_query = question + phi_text
            augmented_embedding = self.embedder.embed_text(augmented_query)
        else:
            augmented_query = question
            augmented_embedding = query_embedding

        # Step 3: Doppio Retrieval (Hybrid KB + VM)
        # 3a. Retrieve_KB(q'_t) — ricerca IBRIDA (Vector + BM25) sulla Knowledge Base
        # Recuperiamo un pool più ampio (top_k * 4) da entrambi
        pool_size = self.top_k * 4
        vec_results = self.vector_store.search(augmented_embedding, top_k=pool_size)
        bm25_results = self.bm25_store.search(augmented_query, top_k=pool_size)
        
        # Fusione RRF (Reciprocal Rank Fusion) per ottenere il super-pool dei candidati
        candidates = self._reciprocal_rank_fusion(vec_results, bm25_results, top_k=pool_size)

        # 3b. Re-Ranking (Cross-Encoder)
        # Il Reranker valuta le coppie (query, doc) tra i candidati RRF e sceglie i migliori top_k
        kb_results = self.reranker.rerank(question, candidates, top_k=self.top_k)

        # 3c. Retrieve_VM(q_t) — cerca nella Vector Memory con la query originale
        mem_results = []
        if memory and not memory.is_empty():
            mem_results = memory.search_similar_turns(query_embedding, top_k=2)

        if not kb_results and not mem_results:
            return "⚠️ Nessun documento rilevante trovato nel database."

        # Step 4: Recupera la cronologia KV
        kv_text = ""
        if memory and not memory.is_empty():
            kv_text = memory.get_kv_history()

        # Step 5: Token Budget Management
        # Applica il trimming per rispettare |C_t| + |KV| + |q'_t| ≤ L_max
        kb_results, mem_results, kv_text = self._apply_token_budget(
            question, kb_results, mem_results, kv_text
        )

        # Step 6: Costruzione del contesto unificato C_t
        context = self._format_dual_context(kb_results, mem_results)

        # Step 7: Costruzione della sezione cronologia KV
        history_section = ""
        if kv_text:
            history_section = (
                f"=== CRONOLOGIA CONVERSAZIONE ===\n"
                f"{kv_text}\n"
                f"=== FINE CRONOLOGIA ===\n"
            )

        # Step 8: Costruzione del prompt completo
        prompt = RAG_PROMPT_TEMPLATE.format(
            history_section=history_section,
            context=context,
            question=question,
        )

        # Step 9: Generazione della risposta
        total_tokens = estimate_tokens(prompt)
        print(f"\n📚 Trovati {len(kb_results)} chunk KB (Ibridi RRF) + {len(mem_results)} turni memoria "
              f"(~{total_tokens} token stimati). Generazione risposta...\n")
        response = self.llm_client.generate(prompt)

        # Step 10: Aggiornamento memoria M_t = Update(M_{t-1}, q_t, R_t)
        if memory:
            memory.add_turn(question, response)

        return response

    # ------------------------------------------------------------------ #
    #                     RECIPROCAL RANK FUSION (RRF)                    #
    # ------------------------------------------------------------------ #

    def _reciprocal_rank_fusion(self, vec_results: list[dict], bm25_results: list[dict], top_k: int, k: int = 60) -> list[dict]:
        """
        Fonde i risultati vettoriali e lessicali usando Reciprocal Rank Fusion.

        RRF score = 1 / (k + rank)
        Un documento trovato da entrambi i retriever otterrà un punteggio molto alto.

        Args:
            vec_results: Risultati dal vector store (ordinati per score decrescente).
            bm25_results: Risultati dal BM25 store (ordinati per score decrescente).
            top_k: Numero di risultati finali da restituire.
            k: Costante di smoothing standard per RRF.

        Returns:
            Lista fusa dei top_k documenti.
        """
        # Mappa per accumulare gli RRF score per ogni testo univoco.
        # Usiamo il testo come chiave univoca del chunk.
        rrf_scores = {}
        doc_map = {}

        # 1. Assegna RRF score dai risultati vettoriali
        for rank, res in enumerate(vec_results, start=1):
            text = res["text"]
            if text not in rrf_scores:
                rrf_scores[text] = 0.0
                doc_map[text] = res
            rrf_scores[text] += 1.0 / (k + rank)

        # 2. Assegna RRF score dai risultati BM25
        for rank, res in enumerate(bm25_results, start=1):
            text = res["text"]
            if text not in rrf_scores:
                rrf_scores[text] = 0.0
                doc_map[text] = res
            rrf_scores[text] += 1.0 / (k + rank)

        # 3. Ordina per RRF score descrescente
        sorted_texts = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)

        # 4. Estrai i top_k
        final_results = []
        for text in sorted_texts[:top_k]:
            doc = doc_map[text].copy()
            doc["score"] = rrf_scores[text]  # Sostituisci lo score originale con l'RRF
            final_results.append(doc)

        return final_results

    # ------------------------------------------------------------------ #
    #                      TOKEN BUDGET MANAGEMENT                        #
    # ------------------------------------------------------------------ #

    def _apply_token_budget(
        self,
        question: str,
        kb_results: list[dict],
        mem_results: list[str],
        kv_text: str,
    ) -> tuple[list[dict], list[str], str]:
        """
        Taglia dinamicamente il contesto per rispettare il budget di token.

        Priorità di taglio (dal meno al più importante):
          1. Chunk KB con score più basso (si rimuovono per ultimi i più rilevanti)
          2. Turni KV più vecchi (si rimuovono i primi della lista)
          3. Summary/KV troncati come ultima risorsa

        Args:
            question: La domanda dell'utente.
            kb_results: Risultati dalla Knowledge Base.
            mem_results: Risultati dalla Vector Memory.
            kv_text: Cronologia KV formattata.

        Returns:
            Tupla (kb_results_trimmed, mem_results_trimmed, kv_text_trimmed).
        """
        budget = self.max_context_tokens

        # Sottrai i token fissi (template + domanda)
        budget -= _TEMPLATE_OVERHEAD_TOKENS
        budget -= estimate_tokens(question)

        # Calcola i token di ogni componente
        kv_tokens = estimate_tokens(kv_text)
        mem_tokens = sum(estimate_tokens(t) for t in mem_results)
        kb_tokens = [estimate_tokens(r["text"]) for r in kb_results]
        total = kv_tokens + mem_tokens + sum(kb_tokens)

        if total <= budget:
            # Tutto rientra nel budget, nessun trimming necessario
            return kb_results, mem_results, kv_text

        print(f"⚠️ [Token Budget] Contesto troppo grande (~{total} token, budget={budget}). Trimming...")

        # Fase 1: Rimuovi chunk KB dal meno al più rilevante (score crescente)
        while kb_results and total > budget:
            removed = kb_results.pop()  # L'ultimo è il meno rilevante (score più basso)
            total -= kb_tokens.pop()
            print(f"   ✂️ Rimosso chunk KB (ne restano {len(kb_results)})")

        # Fase 2: Riduci la KV history (rimuovi i turni più vecchi)
        if total > budget and kv_text:
            kv_lines = kv_text.split("\n\n")
            while kv_lines and total > budget:
                removed_turn = kv_lines.pop(0)  # Rimuovi il più vecchio
                total -= estimate_tokens(removed_turn)
                print(f"   ✂️ Rimosso turno KV vecchio (ne restano {len(kv_lines)})")
            kv_text = "\n\n".join(kv_lines)

        # Fase 3: Tronca il testo KV residuo se ancora troppo grande
        if total > budget and kv_text:
            max_kv_chars = budget * 4  # Conversione inversa token → chars
            kv_text = kv_text[:max_kv_chars]
            print(f"   ✂️ KV troncata a ~{budget} token")

        return kb_results, mem_results, kv_text

    # ------------------------------------------------------------------ #
    #                    FORMATTAZIONE CONTESTO                           #
    # ------------------------------------------------------------------ #

    def _format_dual_context(self, kb_results: list[dict], mem_results: list[str]) -> str:
        """
        Formatta il contesto unificato C_t = Retrieve_KB ∪ Retrieve_VM.

        I risultati dalla KB hanno metadati (file, pagina).
        I risultati dalla memoria sono testo libero dei turni passati.
        """
        context_parts = []

        # Contesto dalla Knowledge Base (documenti PDF)
        for i, r in enumerate(kb_results, 1):
            context_parts.append(
                f"[Fonte {i}: {r['source_file']}, pag. {r['page_number']}]\n"
                f"{r['text']}\n"
            )

        # Contesto dalla Vector Memory (turni passati rilevanti)
        if mem_results:
            context_parts.append("--- DALLA MEMORIA CONVERSAZIONALE ---")
            for i, turn_text in enumerate(mem_results, 1):
                context_parts.append(
                    f"[Memoria, turno correlato {i}]\n"
                    f"{turn_text}\n"
                )

        return "\n---\n".join(context_parts)
