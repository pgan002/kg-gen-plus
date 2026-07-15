import os
import time

import litellm
from oauthlib.oauth2 import LegacyApplicationClient
from requests_oauthlib import OAuth2Session

# --- Your Configuration ---
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
TOKEN_URL = os.getenv("TOKEN_URL")

# Initialize the OAuth2 Session
oauth = OAuth2Session(
    client_id=CLIENT_ID,
    client=LegacyApplicationClient(client_id=CLIENT_ID),
    auto_refresh_url=TOKEN_URL,
)

_cached_token = None
_token_expires_at = 0


def get_keycloak_token():
    global _cached_token, _token_expires_at

    # Refresh token if we don't have one, or if it expires in less than 10 seconds
    if not _cached_token or time.time() >= _token_expires_at:
        token_data = oauth.fetch_token(
            token_url=TOKEN_URL,
            username=USERNAME,
            password=PASSWORD,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            # Request the audience required by the oauth2-proxy
            audience="oauth2-proxy-api",
        )
        _cached_token = token_data["access_token"]
        # Keycloak usually returns 'expires_in' (seconds). Default to 300s if missing.
        _token_expires_at = time.time() + token_data.get("expires_in", 300) - 10

    return _cached_token


# --- 2. Wrap LiteLLM Core Functions ---
original_completion = litellm.completion
original_acompletion = litellm.acompletion


def completion_with_auth(*args, **kwargs):
    """Inject the token as the standard api_key."""
    # The OpenAI client in LiteLLM automatically converts api_key into "Authorization: Bearer <token>"
    kwargs["api_key"] = get_keycloak_token()
    return original_completion(*args, **kwargs)


async def acompletion_with_auth(*args, **kwargs):
    kwargs["api_key"] = get_keycloak_token()
    return await original_acompletion(*args, **kwargs)


litellm.completion = completion_with_auth
litellm.acompletion = acompletion_with_auth
