# Social contract (v1, 2026-10-08)

Friends, presence and one-to-one chat for the desktop app, as `src/social.rs` calls them. Same conventions as the
[native sign-in contract](native-sign-in-contract.md): JSON in and out, UTF-8, under the service's base URL.

## Conventions

Every route is `POST /identity/native/social/<name>` with a JSON body carrying `client_id` (a registered native
client, `alelyon-desktop`) and `refresh_token` (the session the app holds). These routes check the token **without
rotating it**.

Errors have the native shape `{"error": "<code>", "message": "<words for a person>"}`. The common ones:

| status | code | when |
|---|---|---|
| 400 | `invalid-request` | a malformed body or field |
| 400 | `unknown-client` | `client_id` is not a registered native client |
| 401 | `invalid-token` | the session has ended |
| 409 | `username-needed` | the account has no username (every route except `profile`) |
| 429 | `rate-limited` | with `Retry-After` in whole seconds |
| 503 | `maintenance` | an identity maintenance notice is in its window (with the notice) |

A **person** is `{"id": "<opaque account id>", "username": "<username or null>", "display_name": "..."}`, never an
email address.

## Routes

| route | body (besides `client_id`, `refresh_token`) | replies |
|---|---|---|
| `profile` | `username?` | 200 `{"me": person}`. With `username`, sets the account's FIRST username (2-39 lowercase letters, digits, `.`, `_`, `-`): 400 `invalid-username`, 409 `username-taken`, 409 `username-set` (changing one is not in v1) |
| `search` | `query` | 200 `{"people": [person]}`: an exact, case-insensitive username match, so zero or one; never by email, never a prefix, never the caller, never a disabled account. A query that is not a valid username answers an empty list, not an error |
| `friends` | | 200 `{"friends": [{"person", "presence", "last_seen", "unread"}], "incoming": [{"id", "from", "sent_at"}], "outgoing": [{"id", "to", "sent_at"}]}`; friends sorted online, away, offline, then by display name; `unread` counts that friend's messages not read yet |
| `request` | `username` | 200 `{"state": "requested"\|"friends"}` (a crossing request makes them friends at once); 404 `no-such-person` (also the caller's own username, an account without a username, a disabled account); 409 `already-friends`, `already-requested`, `too-many-requests` (50 pending outgoing) |
| `respond` | `request_id`, `accept` (a JSON boolean) | **204**; only the recipient; declining deletes the request with no notice; 404 `no-such-request` |
| `cancel` | `request_id` | **204**; only the sender; 404 `no-such-request` |
| `remove` | `friend_id` | **204**; ends the friendship for both and deletes their messages; 404 `not-friends` |
| `presence` | `state`: `online`\|`away`\|`offline` | **204** |
| `messages` | `friend_id`, `before?` (a message id) | 200 `{"messages": [{"id", "from", "text", "sent_at"}], "more": bool}`: the newest 50 (older than `before`), oldest first; marks the friend's messages up to the newest returned as read; 404 `not-friends`; 400 for a `before` not in that conversation |
| `send` | `friend_id`, `text` | 200 `{"message": {"id", "from", "text", "sent_at"}}`; `text` is 1-2000 characters after trimming with no control character except newline (a tab or a carriage return is refused, 400 `invalid-request`); 404 `not-friends` |

`from` in a message is the sender's account id. Times are RFC 3339.

## Rate limits

Per account: search 30 and request 20 per 15 minutes; presence 6 and send 60 per minute. A refusal is 429
`rate-limited` with `Retry-After` in whole seconds (at least 1).

## Presence and polling

The service stores each account's last reported state and its time. A friend whose last report is older than 150
seconds is shown `offline`, whatever it said. This is computed when `friends` is read: nothing in the service runs on
a timer.

The app does the polling, and only while its window is open and signed in: `presence` on start (`online`) and every
60 seconds, `away` after 10 minutes without input, `offline` when it closes; `friends` every 30 seconds (every 10
seconds while a friends panel is open); `messages` every 5 seconds only while a conversation is open. Nothing while
closed or offline.

## Privacy and retention

- Messages are stored on the server **in plain text**. They are **not** end-to-end encrypted in v1: anyone with access
  to the service's database can read them.
- A conversation is kept until the friendship ends (`remove` deletes it for both people), and every social record
  goes with its account. Read marks are kept per reader and friend. Friend requests stay until answered or cancelled.
- Disabled accounts cannot be found, requested, or seen in request lists. A friendship that already exists stays
  listed, with presence `offline` and `last_seen` null.
- The service logs no token, no search query and no message text.
