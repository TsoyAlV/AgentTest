import httpx
import numpy as np
import time

from .errors import ServiceError


class Embeddings:
    def __init__(self, settings, client: httpx.Client):
        self.settings = settings
        self.client = client

    def encode(self, query: str, deadline=None) -> np.ndarray:
        return self.encode_many([query],deadline=deadline)[0]

    def encode_many(self, texts: list[str], deadline=None) -> np.ndarray:
        if not texts or not all(isinstance(text,str) and text.strip() for text in texts):
            raise ValueError('Expected nonempty input texts')
        cfg = self.settings
        if not cfg.embedding_base_url:
            raise ServiceError("Set EMBEDDING_BASE_URL after deploying BAAI/bge-m3")
        key = cfg.embedding_api_key.get_secret_value()
        options={}
        if deadline is not None:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise ServiceError('Request time budget exhausted',504)
            options={'timeout':min(cfg.http_timeout_seconds,remaining)}
        try:
            response = self.client.post(
                cfg.embedding_base_url.rstrip("/") + "/embeddings",
                headers={"Authorization": f"Bearer {key}"} if key else {},
                json={"model": cfg.embedding_model, "input": texts},
                **options,
            )
            response.raise_for_status()
            data=response.json()['data']
            if len(data)!=len(texts):raise ValueError('Wrong embedding count')
            if len(texts)>1 or 'index' in data[0]:
                data=sorted(data,key=lambda item:item['index'])
                if [item['index'] for item in data]!=list(range(len(texts))):raise ValueError('Invalid embedding indexes')
            vectors = np.asarray([item['embedding'] for item in data], dtype=np.float32)
        except httpx.TimeoutException:
            raise ServiceError("Embedding request timed out; check model loading and HTTP_TIMEOUT_SECONDS", 504) from None
        except httpx.HTTPStatusError as exc:
            raise ServiceError(f"Embedding server returned HTTP {exc.response.status_code}; check model ID and API key", 502) from None
        except httpx.RequestError:
            raise ServiceError("Cannot reach embedding server; check EMBEDDING_BASE_URL and network access", 502) from None
        except (KeyError, IndexError, ValueError, TypeError):
            raise ServiceError("Embedding service failed or returned an invalid response", 502) from None
        if (vectors.shape != (len(texts),cfg.embedding_dimensions)
                or not np.isfinite(vectors).all() or (np.linalg.norm(vectors,axis=1)==0).any()):
            raise ServiceError("Embedding has invalid dimensions or values", 502)
        return vectors
