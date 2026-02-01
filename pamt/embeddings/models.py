from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Protocol
from urllib import request

from ..config import EmbeddingConfig


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> List[float]:
        ...


@dataclass
class OllamaEmbeddings:
    config: EmbeddingConfig

    def embed(self, text: str) -> List[float]:
        payload = {"model": self.config.model_name, "prompt": text}
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.config.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.request_timeout) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return parsed.get("embedding", [])


@dataclass
class DeepSeekEmbeddings:
    config: EmbeddingConfig

    def embed(self, text: str) -> List[float]:
        payload = {"model": self.config.model_name, "input": text}
        data = json.dumps(payload).encode("utf-8")
        url = self.config.api_base_url.rstrip("/") + "/embeddings"
        req = request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.request_timeout) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return parsed["data"][0]["embedding"]


@dataclass
class HFLocalEmbeddings:
    config: EmbeddingConfig
    _tokenizer: object | None = None
    _model: object | None = None
    _device: object | None = None

    def embed(self, text: str) -> List[float]:
        self._ensure_model()
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.hf_max_length,
            padding=True,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        if self.config.hf_normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.squeeze(0).tolist()

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install transformers to use hf embeddings.") from exc
        import torch

        device = self.config.hf_device
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModel.from_pretrained(self.config.model_name)
        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def _mean_pool(hidden_states, attention_mask):
        import torch

        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts
