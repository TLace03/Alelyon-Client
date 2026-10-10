# Native sign-in contract (v1, 2026-10-07; email codes 2026-10-09)

The identity service's routes for a native (desktop) client, as `src/client.rs` and `src/loopback.rs` call them.
The 2026-10-09 additions (`sign-up`, `verify-email`, `resend-code`, `beta-key`, `beta-key/redeem`) are served by the
identity service; this crate does not call them yet. `sign-up`, `verify-email` and `resend-code` were reworked the same
day after a security review (handles instead of passwords); no client had built against the earlier shape.

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
| `POST /identity/native/sign-in` | `{client_id, login, password, stay_signed_in, link_ticket?}` (login = email or username) | 200 session; 401 `invalid-credentials`; 429; 503 |
| `POST /identity/native/sign-up` | `{client_id, email, password, display_name?, agree: true, turnstile_token?}` | 202 `{"handle", "expires_at", "code_expires_at"}` (both RFC 3339); 400 `agreement-required`, `invalid-email`, `weak-password`, `invalid-request`, `invalid-client`; 403 `challenge-failed`; 429; 503 `mail-unavailable` |
| `POST /identity/native/verify-email` | `{client_id, handle, code, stay_signed_in}` (code = the 6 digits mailed) | 200 session (the account now exists); 400 `invalid-code`; 429; 503 |
| `POST /identity/native/resend-code` | `{client_id, handle, turnstile_token?}` | 202 `{"accepted": true, "code_expires_at": "<RFC 3339>"}`; 410 `sign-up-expired`; 403 `challenge-failed`; 429; 503 `mail-unavailable` |
| `POST /identity/native/beta-key` | `{client_id, refresh_token}` (no rotation) | 200 `{"state": "ready", "key": "ALN-...", "expires_at": "<RFC 3339>"}` or 200 `{"state": "unverified", "message"}` (an account whose email address is not proven, below); 401 `invalid-token` |
| `POST /identity/native/beta-key/redeem` | `{key}` (no client id and no session: the key is the credential) | 200 `{"state": "ok" \| "unknown-key" \| "expired" \| "already-redeemed"}`; 429 |
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
sign-in to that account from the same client, within 10 minutes, carries `link_ticket` (one use). A password sign-up
still awaiting its code is not an account: a provider that verified its address creates the account (after the consent
page below) and ends the address's pending sign-ups, so it never answers `account-exists` because of one.

**A new account needs consent.** The first sign-in with a provider that matches no account creates nothing at the
callback: the browser shows a "Create your Alelyon account" page with the Terms of Service and Privacy Notice and an
unticked box. Only Create account with the box ticked creates it, and the loopback then gets `code` and `state` as
usual. Cancel sends `error=access-denied`. The client needs nothing extra for this.

**The QR code** encodes `approve_url`, which carries a public handle, not the `pair_id` the client polls with, so a
photo of the code cannot collect the session.

## A password sign-up and its code (2026-10-09)

A password sign-up is not an account until the 6-digit code mailed to the address is entered. A native client:

1. Shows the sign-up form (email, display name, password, and a required box agreeing to the Terms of Service and
   Privacy Notice, linked) and sends it to `sign-up` with `agree: true`. While the service runs Cloudflare Turnstile
   `sign-up` and `resend-code` also need `turnstile_token`, solved on the identity service's hostname for the actions
   `native-sign-up` and `native-resend` (a token solved on the sign-up page's own widget is refused). No page mints
   those yet, so until one does the client opens `sign_up_url` (the service's sign-up page) instead.
2. Keeps the reply's `handle` in memory (it is the only proof the sign-up is this client's; never log it) and asks the
   person for the code from the email (input: 6 digits, `one-time-code`).
3. Sends it to `verify-email` with the `handle` and `stay_signed_in`. The reply is a session, as sign-in's is; the
   account now exists with its address verified.
4. To get a new code, calls `resend-code` with the `handle`. The older code stops working.

Rules the service applies (a client only needs to show the error's `message`):

- The reply to `sign-up` is the same for every address. For an address that already has an account it is a decoy's
  handle: no code is mailed, every code fails, and the account's owner may be told (once a day) that someone tried.
  The client cannot tell, and must not try to.
- Every failed code check is 400 `invalid-code`: a wrong, expired or malformed code, a decoy's handle, an unknown
  handle, and a sign-up that has ended. A code works for 10 minutes and once; a new code replaces the old one.
- A sign-up ends 24 hours after it started (its `expires_at`), after five codes that did not work, or once confirmed.
  The client counts the failures and watches `expires_at` to say "sign up again"; `resend-code` answers an ended
  sign-up with 410 `sign-up-expired`.
- `resend-code` sends one code a minute and five per sign-up; an address gets at most ten code emails and five new
  sign-ups in 24 hours, whoever asks, and one source at most six and three of them (429 with `Retry-After`
  otherwise). It answers 503 `mail-unavailable` when the service has no mail sender. The mail leaves after the reply,
  so a failed send is not reported; the person asks again.
- Until the code is confirmed there is no account: sign-in answers 401 `invalid-credentials`, as for an unknown
  address.
- A client that cannot keep a long session (the website's `alelyon-web`) gets 400 `invalid-request` for
  `stay_signed_in: true`, as at sign-in.

A sign-in through Google, GitHub, LinkedIn or another provider needs no code when the provider says it verified the
email address (see Linking for a provider sign-in that meets a password sign-up awaiting its code).

## The hosted beta key (2026-10-09)

A signed-in account with a verified email address gets its hosted DQC-OS beta key automatically: `beta-key` mints it
on the first ask and returns the same key while it is live (72 hours, not yet redeemed), so the client may ask on
every sign-in. An expired or redeemed key is replaced by a new one. An account whose real address is not proven (by a
sign-up code, a used password-reset link, or a provider that verified it) gets `{"state": "unverified"}`: in
practice a provider account whose provider gave no verified address, or an account an operator brought over from a
desktop that has not yet reset its password or linked such a provider. The installed app spends a key once with
`beta-key/redeem`; typed keys are accepted in any case, with or without the `ALN-` prefix and hyphens.

## Limits and logging

Sign-in, sign-up, verify-email, resend-code, exchange, provider start, pair start and pair poll are rate limited per
source; a pairing polled faster than once a second gets 429 `slow-down`. Passwords are stored memory-hard on the
service. Nothing in a reply is a secret except the refresh token and a sign-up's `handle`, and the service's logs
never carry tokens, handles or codes.
