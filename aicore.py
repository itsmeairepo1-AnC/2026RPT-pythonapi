"""
SAP AI Core OAuth client for inference endpoints.
"""
import time
import requests

TOKEN_TIMEOUT = 30
HTTP_TIMEOUT = 60
TOKEN_REFRESH_BUFFER = 60


class AICoreClient:
    """OAuth-managed HTTP client for SAP AI Core inference endpoints."""

    def __init__(self, auth_url: str, client_id: str, client_secret: str,
                 base_url: str, resource_group: str):
        self._auth_url = auth_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._base_url = base_url
        self._resource_group = resource_group
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0

    def _get_token(self) -> str:
        now = time.time()
        if self._access_token and now < (self._token_expires_at - TOKEN_REFRESH_BUFFER):
            return self._access_token

        response = requests.post(
            f"{self._auth_url}/oauth/token",
            data={"grant_type": "client_credentials"},
            auth=(self._client_id, self._client_secret),
            timeout=TOKEN_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        self._access_token = payload["access_token"]
        self._token_expires_at = now + int(payload.get("expires_in", 600))
        return self._access_token

    def predict(self, deployment_url: str, payload: dict) -> dict:
        token = self._get_token()
        response = requests.post(
            deployment_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "AI-Resource-Group": self._resource_group,
                "Content-Type": "application/json",
            },
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def chat_complete(self, deployment_url: str, system_prompt: str,
                      history: list[dict], user_message: str,
                      max_tokens: int = 500) -> str:
        """
        Call GPT-4o deployed on SAP AI Core via /v1/chat/completions (OpenAI-compatible).
        """
        token = self._get_token()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        url = deployment_url.rstrip("/")
        if not url.endswith("/v1/chat/completions"):
            url = f"{url}/v1/chat/completions"

        response = requests.post(
            url,
            json={"messages": messages, "max_tokens": max_tokens, "temperature": 0.3},
            headers={
                "Authorization": f"Bearer {token}",
                "AI-Resource-Group": self._resource_group,
                "Content-Type": "application/json",
            },
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

