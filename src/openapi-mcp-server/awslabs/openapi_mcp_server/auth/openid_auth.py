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
"""OpenID Connect authentication provider.

Supports Resource Owner Password Credentials (ROPC) flow when username/password
are supplied, and Client Credentials flow otherwise.  The token endpoint is
discovered automatically from the OpenID Connect configuration document
(``AUTH_OPENID_CONFIG_URL``).
"""

import httpx
import threading
import time
from awslabs.openapi_mcp_server import logger
from awslabs.openapi_mcp_server.api.config import Config
from awslabs.openapi_mcp_server.auth.auth_errors import (
    ConfigurationError,
    InvalidCredentialsError,
    MissingCredentialsError,
    NetworkError,
)
from awslabs.openapi_mcp_server.auth.bearer_auth import BearerAuthProvider
from typing import Dict, Optional


class OpenIDAuthProvider(BearerAuthProvider):
    """OpenID Connect authentication provider.

    Obtains an access token from an OIDC-compliant token endpoint and
    delegates to ``BearerAuthProvider`` for attaching the ``Authorization``
    header to every HTTP request.

    **Grant-type selection**

    * If ``auth_username`` *and* ``auth_password`` are set → Resource Owner
      Password Credentials (ROPC) flow (``grant_type=password``).
    * Otherwise → Client Credentials flow
      (``grant_type=client_credentials``).

    **Required env vars**

    * ``AUTH_TYPE=openid``
    * ``AUTH_OPENID_CONFIG_URL`` – OIDC discovery document URL, e.g.
      ``https://host/auth/realms/myrealm/.well-known/openid-configuration``
    * ``AUTH_OPENID_CLIENT_ID`` (or ``AUTH_OPENID_CLIENTID``) – client ID
    * ``AUTH_OPENID_CLIENT_SECRET`` (or ``AUTH_OPENID_CLIENTSECRET``) –
      client secret

    **Optional env vars**

    * ``AUTH_USERNAME`` / ``AUTH_PASSWORD`` – enable ROPC flow
    * ``AUTH_OPENID_SCOPES`` – space-separated scopes (default: ``openid``)
    """

    def __init__(self, config: Config) -> None:
        """Initialise provider, discover OIDC metadata, and obtain a token.

        Args:
            config: Application configuration object.

        Raises:
            MissingCredentialsError: If required OIDC configuration is absent.
            NetworkError: If the token endpoint cannot be reached.
            InvalidCredentialsError: If the token request is rejected.

        """
        # Store OIDC-specific configuration before calling super().__init__
        self._openid_config_url: str = config.auth_openid_config_url
        self._client_id: str = config.auth_openid_client_id
        self._client_secret: str = config.auth_openid_client_secret
        self._username: str = config.auth_username
        self._password: str = config.auth_password
        self._scopes: str = getattr(config, 'auth_openid_scopes', 'openid') or 'openid'

        # Cached token-endpoint URL (populated on first _fetch_token_endpoint call)
        self._token_endpoint: Optional[str] = None

        # Token lifetime tracking
        self._token_expires_at: float = 0.0
        self._token_lock = threading.RLock()

        # Determine grant type
        self._grant_type = (
            'password' if (self._username and self._password) else 'client_credentials'
        )
        logger.info(
            f'OpenID auth using grant type: {self._grant_type} '
            f'(client_id={self._client_id})'
        )

        # Obtain initial token – failures are fatal during startup
        token = self._get_token()
        if not token:
            raise ConfigurationError(
                'OpenID Connect authentication: failed to obtain an access token',
                {'help': 'Check AUTH_OPENID_CONFIG_URL, credentials, and network connectivity'},
            )
        config.auth_token = token

        # Delegate to BearerAuthProvider which sets up the Authorization header
        super().__init__(config)

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> bool:
        """Validate required OIDC configuration fields.

        Raises:
            MissingCredentialsError: When required fields are absent.

        """
        if not self._openid_config_url:
            raise MissingCredentialsError(
                'OpenID Connect authentication requires AUTH_OPENID_CONFIG_URL',
                {
                    'help': (
                        'Set the AUTH_OPENID_CONFIG_URL environment variable to the '
                        '.well-known/openid-configuration URL of your identity provider'
                    )
                },
            )
        if not self._client_id:
            raise MissingCredentialsError(
                'OpenID Connect authentication requires AUTH_OPENID_CLIENT_ID',
                {
                    'help': (
                        'Set the AUTH_OPENID_CLIENT_ID (or AUTH_OPENID_CLIENTID) '
                        'environment variable'
                    )
                },
            )
        if not self._client_secret:
            raise MissingCredentialsError(
                'OpenID Connect authentication requires AUTH_OPENID_CLIENT_SECRET',
                {
                    'help': (
                        'Set the AUTH_OPENID_CLIENT_SECRET (or AUTH_OPENID_CLIENTSECRET) '
                        'environment variable'
                    )
                },
            )
        # Let BearerAuthProvider validate that self._token is set
        return super()._validate_config()

    def _log_validation_error(self) -> None:
        """Log validation error details."""
        logger.error(
            'OpenID Connect authentication is misconfigured. '
            'Ensure AUTH_OPENID_CONFIG_URL, AUTH_OPENID_CLIENT_ID, and '
            'AUTH_OPENID_CLIENT_SECRET are set correctly.'
        )

    # ------------------------------------------------------------------
    # Token acquisition
    # ------------------------------------------------------------------

    def _fetch_token_endpoint(self) -> str:
        """Fetch the OIDC discovery document and return the token endpoint URL.

        Returns:
            str: Token endpoint URL.

        Raises:
            NetworkError: If the discovery document cannot be fetched.
            ConfigurationError: If the document does not contain a token_endpoint.

        """
        logger.debug(f'Fetching OpenID configuration from: {self._openid_config_url}')
        try:
            response = httpx.get(self._openid_config_url, timeout=10.0, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise NetworkError(
                f'Failed to fetch OpenID configuration: {exc}',
                {'url': self._openid_config_url, 'error': str(exc)},
            ) from exc

        oidc_config: Dict = response.json()
        token_endpoint = oidc_config.get('token_endpoint')
        if not token_endpoint:
            raise ConfigurationError(
                'OpenID configuration document does not contain a token_endpoint',
                {'url': self._openid_config_url, 'keys': list(oidc_config.keys())},
            )
        # Log at INFO so the URL is always visible in logs for troubleshooting
        logger.info(f'OpenID token endpoint discovered: {token_endpoint}')
        return token_endpoint

    def _get_token(self) -> Optional[str]:
        """Request a new access token from the OIDC token endpoint.

        Returns:
            str: Access token, or ``None`` if the server returned no token.

        Raises:
            NetworkError: On connection / HTTP errors.
            InvalidCredentialsError: When the token request is rejected (4xx).

        """
        # Lazily resolve token endpoint
        if not self._token_endpoint:
            self._token_endpoint = self._fetch_token_endpoint()

        data: Dict[str, str] = {
            'client_id': self._client_id,
            'client_secret': self._client_secret,
        }

        if self._grant_type == 'password':
            data.update(
                {
                    'grant_type': 'password',
                    'username': self._username,
                    'password': self._password,
                    'scope': self._scopes,
                }
            )
            logger.debug(f'Requesting ROPC token for user: {self._username}')
        else:
            data.update(
                {
                    'grant_type': 'client_credentials',
                    'scope': self._scopes,
                }
            )
            logger.debug('Requesting client-credentials token')

        try:
            # Do NOT follow redirects automatically: an HTTP 3xx redirect on a
            # POST will be re-issued as GET by most clients, causing 405 on the
            # token endpoint.  We handle redirects explicitly below.
            response = httpx.post(
                self._token_endpoint,
                data=data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=10.0,
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise NetworkError(
                f'Network error while requesting OpenID token: {exc}',
                {'endpoint': self._token_endpoint, 'error': str(exc)},
            ) from exc

        # If the server redirects the token request, re-POST to the new location
        # (standard follow_redirects would silently convert POST → GET).
        if response.status_code in (301, 302, 307, 308):
            redirect_url = response.headers.get('location', '')
            logger.info(
                f'Token endpoint redirected ({response.status_code}) to: {redirect_url}. '
                'Re-issuing POST to redirect target.'
            )
            try:
                response = httpx.post(
                    redirect_url,
                    data=data,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                    timeout=10.0,
                    follow_redirects=False,
                )
            except httpx.HTTPError as exc:
                raise NetworkError(
                    f'Network error while requesting OpenID token (after redirect): {exc}',
                    {'endpoint': redirect_url, 'error': str(exc)},
                ) from exc

        if response.status_code != 200:
            logger.error(
                f'Token request failed: HTTP {response.status_code} – {response.text}'
            )
            raise InvalidCredentialsError(
                f'OpenID token request rejected (HTTP {response.status_code})',
                {
                    'endpoint': self._token_endpoint,
                    'response': response.text,
                    'help': (
                        'Verify client_id, client_secret, and user credentials. '
                        'For Keycloak: ensure "Direct Access Grants" is enabled '
                        'for the client in the Keycloak admin console.'
                    ),
                },
            )

        token_data: Dict = response.json()
        access_token: Optional[str] = token_data.get('access_token')
        expires_in: int = int(token_data.get('expires_in', 3600))

        if access_token:
            self._token_expires_at = time.time() + expires_in
            logger.info(f'OpenID access token obtained (expires in {expires_in}s)')
        else:
            logger.error('Token endpoint response did not contain an access_token')

        return access_token

    # ------------------------------------------------------------------
    # Token refresh
    # ------------------------------------------------------------------

    def _is_token_expiring_soon(self) -> bool:
        """Return True if the token has expired or expires within 5 minutes."""
        return time.time() + 300 >= self._token_expires_at

    def _refresh_token(self) -> None:
        """Silently refresh the access token and update auth headers."""
        logger.debug('Refreshing OpenID access token')
        try:
            new_token = self._get_token()
            if new_token and new_token != self._token:
                self._token = new_token
                self._initialize_auth()
                logger.info('OpenID access token refreshed')
        except Exception as exc:
            logger.error(f'Failed to refresh OpenID token: {exc}')

    def get_auth_headers(self) -> Dict[str, str]:
        """Return Authorization headers, refreshing the token if necessary."""
        with self._token_lock:
            if self._is_token_expiring_soon():
                self._refresh_token()
        return super().get_auth_headers()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return 'openid'
