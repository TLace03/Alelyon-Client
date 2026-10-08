# Native sign-in contract (v1, 2026-10-07)

The identity service's routes for a native (desktop) client, as `src/client.rs` and `src/loopback.rs` call them.

## Conventions

All routes are under the identity service's base URL (`https://id.api.alelyon.com` by default;
`ALELYON_IDENTITY_URL` names another). JSON in and out, UTF-8. Native routes take no cookie and no CSRF header: they
are for a desktop client identified by `client_id`, a registered native client. The first is `alelyon-desktop`, a
public client (no secret) whose only redirect is `http://127.0.0.1:<port>/callback`.

Errors: `{"error": "<code>", "message": "<words for a person>"}` with the HTTP status below. 429 carries
`Retry-After`. 503 during maintenance carries `{"error": "maintenance", "notice": <notice>}`.

A **session** reply, everywhere one is returned:

```
{"refresh_token": "<opaque>", "expires_at": "<RFC 3339 UTC>", "token_type": "opaque-refresh",
 "account": {"id": "<stable id>", "email": "<email>", "display_name": "<name>", "username": "<or null>",
             "providers": ["password", "github", ...], "created_at": "<RFC 3339>"}}
```

`stay_signed_in: false` gives a short-lived refresh token (the session ends with the app); `true` a long one. Every
refresh rotates the token: the old one stops working.

## Routes

| route | body | replies |
|---|---|---|
| `GET /identity/status` | (public) | 200 `{"service": "up"\|"maintenance"\|"degraded", "notices": [{"id", "kind": "maintenance"\|"incident"\|"info", "title", "body", "starts_at", "ends_at", "services": [..]}], "providers": [{"id": "github", "name": "GitHub", "enabled": true}, ...], "sign_up_url", "reset_url", "account_url", "security_url"}` |
| `POST /identity/native/sign-in` | `{client_id, login, password, stay_signed_in, link_ticket?}` (login = email or username) | 200 session; 401 `invalid-credentials`; 403 `unverified-email`; 429; 503 |
| `POST /identity/native/refresh` | `{client_id, refresh_token}` | 200 session (rotated); 401 `invalid-token` |
| `POST /identity/native/sign-out` | `{client_id, refresh_token}` | 204 (the token and its family revoked) |
| `POST /identity/native/account` | `{client_id, refresh_token}` | 200 `{"account": ...}` (no rotation); 401 |
| `GET /identity/native/provider/{id}/start?client_id&redirect_uri&state&code_challenge&code_challenge_method=S256&stay_signed_in` | (the system browser opens it) | 302 to the provider; after the provider, 302 to `redirect_uri?code=..&state=..` (for a native client `redirect_uri` must be `http://127.0.0.1:<port>/callback`), or `?error=..&state=..` |
| `POST /identity/native/exchange` | `{client_id, redirect_uri, code, verifier}` | 200 session; 400 `invalid-grant` (one use, 120 s, PKCE S256 checked) |
| `POST /identity/native/pair/start` | `{client_id, stay_signed_in}` | 200 `{"pair_id", "user_code" (8 characters, shown beside the QR code), "approve_url" (what the QR code encodes: the web page where a signed-in person approves), "expires_at", "interval": 2}` |
| `POST /identity/native/pair/poll` | `{client_id, pair_id}` | 202 `{"status": "pending"}`; 200 session (once; then the pairing is spent); 410 `expired`; 403 `denied` |
| `GET /identity/pair/{handle}` (web, signed-in browser session) | | the approve page: shows the user code and the desktop's description; Approve / Deny |
| `POST /identity/pair/{handle}/approve` and `/deny` (web, session cookie + CSRF) | `{user_code}` | 204 |

`provider/{id}` is one of `github`, `google`, `microsoft`, `apple`, `orcid`, `huggingface`, `linkedin`. A provider
the deployment has not configured is `enabled: false` in status, and its start route answers 404
`provider-disabled`.

## Provider sign-in, in order

1. The client makes a fresh PKCE verifier (32 random bytes, base64url), its S256 challenge
   (base64url(SHA-256(verifier)), RFC 7636), and a random state.
2. It binds a listener on a free loopback port and opens the start route in the system browser.
3. The browser signs in at the provider. The client never sees the provider's page or tokens.
4. The service redirects the browser to the loopback with `code` and `state`, or with `error` and `state`.
5. The client accepts the answer only if `state` matches, answers the browser with a short page, and trades the code
   with the verifier at `exchange`.

Callback errors are one of `invalid-request`, `access-denied`, `provider-disabled`, `provider-failed`,
`account-exists`, `account-unavailable`, `maintenance`.

**Linking.** A provider whose verified email already belongs to an account answers the callback with
`error=account-exists&link_ticket=..`. Nothing is linked then. The provider is linked only when the next password
sign-in to that account from the same client, within 10 minutes, carries `link_ticket` (one use).

**A new account needs consent.** The first sign-in with a provider that matches no account creates nothing at the
callback: the browser shows a "Create your Alelyon account" page with the Terms of Service and Privacy Notice and an
unticked box. Only Create account with the box ticked creates it, and the loopback then gets `code` and `state` as
usual. Cancel sends `error=access-denied`. The client needs nothing extra for this.

**The QR code** encodes `approve_url`, which carries a public handle, not the `pair_id` the client polls with, so a
photo of the code cannot collect the session.

## Limits and logging

Sign-in, exchange, provider start, pair start and pair poll are rate limited per source; a pairing polled faster than
once a second gets 429 `slow-down`. Passwords are stored memory-hard on the service. Nothing in a reply is a secret
except the refresh token, and the service's logs never carry tokens.
