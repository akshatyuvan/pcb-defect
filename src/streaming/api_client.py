"""
The only place that knows how to talk to the FastAPI scorer.

Locked decision 10: consumers never import torch and never load the model, they
call the service over HTTP. That makes this client load-bearing, so it does two
unusual things on purpose:

1. stdlib urllib, not requests. Zero third-party deps means the consumer image
   is python:3.12-slim + confluent-kafka and nothing else. We implement our own
   retry anyway, so requests would only be buying us syntax.

2. It resolves the request field name for /predict/board from the served
   OpenAPI schema at startup instead of hardcoding it. A wrong field name
   returns 422, and 422 is our "poison message" signal, so hardcoding it wrong
   would dead-letter every board while looking like correct behaviour.
   PCB_API_IMAGE_FIELD overrides the resolution if you ever need to pin it.
   (Day 7 probe confirmed the field is `image_b64` on ImageRequest; the
   resolution is belt-and-braces against a future schema change.)
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class ApiClientRejection(Exception):
    """4xx. The MESSAGE is bad -- undecodable base64, wrong dimensions.
    Retrying cannot help. This is a poison message: DLQ it and commit."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:400]}")
        self.status = status
        self.body = body


class ApiServerError(Exception):
    """5xx, timeout or connection refused. WE are broken, not the message.
    Retry. Never commit -- the board must survive our outage."""


class PcbApiClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._image_field: Optional[str] = os.environ.get("PCB_API_IMAGE_FIELD") or None

    # ---------------------------------------------------------------- plumbing
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if 400 <= e.code < 500:
                raise ApiClientRejection(e.code, body) from None
            raise ApiServerError(f"HTTP {e.code}: {body[:400]}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ApiServerError(str(e)) from None

    def _get(self, path: str) -> Dict[str, Any]:
        try:
            with urllib.request.urlopen(self.base_url + path, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise ApiServerError(f"HTTP {e.code} on GET {path}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ApiServerError(str(e)) from None

    # ------------------------------------------------------------------- setup
    def wait_for_health(self, timeout_s: float = 120.0, interval: float = 1.0) -> Dict[str, Any]:
        """compose depends_on only waits for the container, not for the model
        load. The model takes ~0.03s but uvicorn startup does not, so poll."""
        deadline = time.monotonic() + timeout_s
        last = "never attempted"
        while time.monotonic() < deadline:
            try:
                return self._get("/health")
            except ApiServerError as e:
                last = str(e)
                time.sleep(interval)
        raise ApiServerError(f"API not healthy after {timeout_s}s: {last}")

    def model_info(self) -> Dict[str, Any]:
        return self._get("/model")

    def image_field(self) -> str:
        if self._image_field:
            return self._image_field
        spec = self._get("/openapi.json")
        try:
            body = spec["paths"]["/predict/board"]["post"]["requestBody"]
            ref = body["content"]["application/json"]["schema"]["$ref"]
            model = spec["components"]["schemas"][ref.rsplit("/", 1)[-1]]
            props = model.get("properties", {})
        except Exception as e:
            raise ApiServerError(f"cannot read /predict/board schema: {e}") from None

        names = list(props)
        for n in names:
            if any(t in n.lower() for t in ("image", "b64", "base64", "data")):
                self._image_field = n
                return n
        strings = [n for n in names if props[n].get("type") == "string"]
        if len(strings) == 1:
            self._image_field = strings[0]
            return strings[0]
        raise ApiServerError(
            f"cannot identify the image field on /predict/board. fields={names}. "
            f"Set PCB_API_IMAGE_FIELD to pin it."
        )

    # ------------------------------------------------------------------- calls
    def predict_board(self, image_b64: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {self.image_field(): image_b64}
        if extra:
            payload.update(extra)
        return self._post("/predict/board", payload)