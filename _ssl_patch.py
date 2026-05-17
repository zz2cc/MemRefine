"""SSL certificate verification patch for restricted network environments.

Apply this at the very beginning of the program (before any HF/httpx imports)
to bypass SSL verification issues commonly encountered behind corporate proxies
on Windows systems without proper CA certificate configuration.
"""

import os
import warnings


def apply():
    """Apply SSL workarounds for restricted environments."""
    # Clear CA bundle paths so no bad cert files are used
    for env_var in ["CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "WEBSOCKET_CLIENT_CA_BUNDLE"]:
        os.environ.pop(env_var, None)

    # Allow HF hub to work without SSL verification
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # Monkey-patch httpx to skip SSL verification
    try:
        import httpx
        _original_client_init = httpx.Client.__init__

        def _patched_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            _original_client_init(self, *args, **kwargs)

        httpx.Client.__init__ = _patched_init
    except ImportError:
        pass

    # Suppress urllib3 insecure request warnings
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except (ImportError, AttributeError):
        pass

    # Suppress other SSL warnings
    warnings.filterwarnings("ignore", message=".*unverified HTTPS.*")
    warnings.filterwarnings("ignore", message=".*SSL.*")
