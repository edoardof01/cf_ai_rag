"""
llm_client.py — Client per l'API REST di Ollama.

Ollama espone un'API HTTP locale per interagire con modelli LLM.
Questo modulo fa una semplice chiamata POST a /api/generate
per ottenere una risposta dal modello.
"""

import requests
import json


class OllamaClient:
    """Client HTTP per comunicare con il server Ollama."""

    def __init__(self, base_url="http://localhost:11434", model="qwen2.5:7b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        print(f"🤖 LLM Client configurato: {model} @ {base_url}")

    def generate(self, prompt: str, stream: bool = True, temperature: float = 0.0) -> str:
        """
        Genera una risposta dal modello LLM.

        Args:
            prompt: Il prompt completo (incluso il contesto RAG).
            stream: Se True, stampa la risposta token per token.
            temperature: Controlla la creatività. 0.0 è ideale per RAG (massima aderenza ai fatti).

        Returns:
            La risposta completa del modello come stringa.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature
            }
        }

        try:
            response = requests.post(url, json=payload, stream=stream)
            response.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(
                f"❌ Impossibile connettersi a Ollama su {self.base_url}.\n"
                f"   Assicurati che Ollama sia in esecuzione: ollama serve"
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"❌ Errore da Ollama: {e}")

        if stream:
            return self._handle_stream(response)
        else:
            return response.json().get("response", "")

    def _handle_stream(self, response) -> str:
        """Processa la risposta in streaming, stampando token per token."""
        full_response = []
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                print(token, end="", flush=True)
                full_response.append(token)
                if data.get("done", False):
                    break
        print()  # Newline finale
        return "".join(full_response)

    def is_available(self) -> bool:
        """Verifica se il server Ollama è raggiungibile."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        """Restituisce la lista dei modelli disponibili in Ollama."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = r.json().get("models", [])
            return [m["name"] for m in models]
        except Exception:
            return []
