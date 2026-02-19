# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JWT Bearer token validation middleware for incoming MCP client requests.

Validates ``Authorization: Bearer <token>`` on every incoming HTTP request
using the JWKS endpoint of the configured identity provider (e.g. Keycloak).
Enables Option B security: MS Foundry → MCP server leg is protected so only
holders of a valid JWT issued by the trusted IdP can call the MCP server.

**Required env vars**

* ``MCP_AUTH_ENABLED=true``
* ``MCP_AUTH_JWKS_URL`` – JWKS endpoint, e.g.
  ``https://host/auth/realms/r/.well-known/jwks.json``
  (auto-discovered from ``AUTH_OPENID_CONFIG_URL`` if not set)
* ``MCP_AUTH_ISSUER`` – Expected ``iss`` claim value
  (auto-discovered from ``AUTH_OPENID_CONFIG_URL`` if not set)

**Optional env vars**

* ``MCP_AUTH_AUDIENCE`` – Expected ``aud`` claim; skipped if empty
"""

import json
import threading
import time
from awslabs.openapi_mcp_server import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from typing import Dict, Optional

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm


def discover_oidc_fields(config_url: str) -> Dict[str, str]:
    """Fetch the OIDC discovery document and return ``jwks_uri`` and ``issuer``.

    Args:
        config_url: ``.well-known/openid-configuration`` URL.

    Returns:
        Dict with keys ``'jwks_uri'`` and ``'issuer'`` (may be empty strings).

    """
    try:
        resp = httpx.get(config_url, timeout=10.0, follow_redirects=True)
        resp.raise_for_status()
        doc = resp.json()
        return {
            'jwks_uri': doc.get('jwks_uri', ''),
            'issuer': doc.get('issuer', ''),
        }
    except Exception as exc:
        logger.warning(f'MCP auth: could not discover OIDC fields from {config_url}: {exc}')
        return {'jwks_uri': '', 'issuer': ''}


class JWTBearerMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that validates incoming Bearer JWTs.

    Fetches and caches JWKS keys from the identity provider; validates
    signature, expiry, issuer, and (optionally) audience on every request.
    Returns ``401 Unauthorized`` JSON for missing, expired, or invalid tokens.
    """

    # Cache JWKS keys for 1 hour
    _JWKS_TTL = 3600

    def __init__(
        self,
        app,
        jwks_url: str,
        issuer: str = '',
        audience: str = '',
    ) -> None:
        super().__init__(app)
        self._jwks_url = jwks_url
        self._issuer: Optional[str] = issuer or None
        self._audience: Optional[str] = audience or None
        self._jwks_cache: Dict[str, object] = {}  # kid → public key object
        self._jwks_fetched_at: float = 0.0
        self._jwks_lock = threading.Lock()

    # ------------------------------------------------------------------
    # JWKS key management
    # ------------------------------------------------------------------

    def _refresh_jwks(self) -> None:
        """Fetch JWKS and rebuild the kid → public-key cache."""
        logger.debug(f'MCP auth: refreshing JWKS from {self._jwks_url}')
        resp = httpx.get(self._jwks_url, timeout=10.0, follow_redirects=True)
        resp.raise_for_status()
        keys = resp.json().get('keys', [])
        self._jwks_cache = {
            k['kid']: RSAAlgorithm.from_jwk(json.dumps(k))
            for k in keys
            if 'kid' in k
        }
        self._jwks_fetched_at = time.time()
        logger.info(f'MCP auth: loaded {len(self._jwks_cache)} JWKS key(s)')

    def _get_public_key(self, kid: str) -> object:
        """Return the public key for *kid*, refreshing JWKS cache if stale."""
        with self._jwks_lock:
            if time.time() - self._jwks_fetched_at > self._JWKS_TTL:
                self._refresh_jwks()
            key = self._jwks_cache.get(kid)
            if key is None:
                # Key not in cache — try a forced refresh (key rotation)
                self._refresh_jwks()
                key = self._jwks_cache.get(kid)
        if key is None:
            raise jwt.InvalidKeyError(f'Unknown key ID: {kid!r}')
        return key

    # ------------------------------------------------------------------
    # Middleware dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, request: Request, call_next):
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return JSONResponse(
                {'error': 'Unauthorized', 'detail': 'Missing or malformed Bearer token'},
                status_code=401,
            )

        token = auth.removeprefix('Bearer ').strip()
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get('kid', '')
            public_key = self._get_public_key(kid)

            decode_options: Dict = {}
            if not self._audience:
                decode_options['verify_aud'] = False

            jwt.decode(
                token,
                public_key,
                algorithms=['RS256', 'RS384', 'RS512'],
                issuer=self._issuer,
                audience=self._audience,
                options=decode_options,
            )
        except jwt.ExpiredSignatureError:
            logger.warning('MCP auth: rejected expired JWT')
            return JSONResponse(
                {'error': 'Unauthorized', 'detail': 'Token has expired'},
                status_code=401,
            )
        except jwt.InvalidIssuerError:
            logger.warning('MCP auth: rejected JWT with wrong issuer')
            return JSONResponse(
                {'error': 'Unauthorized', 'detail': 'Invalid token issuer'},
                status_code=401,
            )
        except (jwt.InvalidTokenError, jwt.InvalidKeyError, Exception) as exc:
            logger.warning(f'MCP auth: rejected invalid JWT: {exc}')
            return JSONResponse(
                {'error': 'Unauthorized', 'detail': 'Invalid token'},
                status_code=401,
            )

        return await call_next(request)
