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
| `profile` | `username?`, `display_name?` | 200 `{"me": person}`. With `username`, sets the account's FIRST username (2-39 lowercase letters, digits, `.`, `_`, `-`): 400 `invalid-username`, 409 `username-taken`, 409 `username-set` (changing one is not in v1). With `display_name`, changes the account's name: 1-64 characters once trimmed (each run of white space becomes one space), no control character and none of `[ ] ( )`, which a name line keeps for a title and tickers: 400 `invalid-name`; 5 a day (429 `rate-limited`) |
| `search` | `query` | 200 `{"people": [person]}`: an exact, case-insensitive username match, so zero or one; never by email, never a prefix, never the caller, never a disabled account. A query that is not a valid username answers an empty list, not an error |
| `friends` | | 200 `{"friends": [{"person", "presence", "last_seen", "unread"}], "incoming": [{"id", "from", "sent_at"}], "outgoing": [{"id", "to", "sent_at"}]}`; friends sorted online, away, dnd, offline, then by display name; `unread` counts that friend's messages not read yet; `plaintext_retires_at` (RFC 3339) only once the service has a date when unsealed messages end ("Unsealed messages retire") |
| `request` | `username` | 200 `{"state": "requested"\|"friends"}` (a crossing request makes them friends at once); 404 `no-such-person` (also the caller's own username, an account without a username, a disabled account); 409 `already-friends`, `already-requested`, `too-many-requests` (50 pending outgoing) |
| `respond` | `request_id`, `accept` (a JSON boolean) | **204**; only the recipient; declining deletes the request with no notice; 404 `no-such-request` |
| `cancel` | `request_id` | **204**; only the sender; 404 `no-such-request` |
| `remove` | `friend_id` | **204**; ends the friendship for both and deletes their messages; 404 `not-friends` |
| `presence` | `state`: `online`\|`away`\|`offline` | **204** |
| `status` | `status?`: `online`\|`away`\|`dnd`\|`offline` | 200 `{"status": ...}`: without `status`, the status the account chose (`online` until it chooses); with it, sets that choice (2026-10-10). `dnd` is do not disturb; `offline` appears offline to friends. 10 changes a minute per account (429 `rate-limited`); reads are not limited. Kept apart from `presence`, so a heartbeat never overwrites it |
| `messages` | `friend_id`, `before?` (a message id) | 200 `{"messages": [{"id", "from", "text", "sent_at"}], "more": bool}`: the newest 50 (older than `before`), oldest first; marks the friend's messages up to the newest returned as read; 404 `not-friends`; 400 for a `before` not in that conversation |
| `send` | `friend_id`, `text` | 200 `{"message": {"id", "from", "text", "sent_at"}}`; `text` is 1-2000 characters after trimming with no control character except newline (a tab or a carriage return is refused, 400 `invalid-request`); 404 `not-friends`; 410 `plaintext-retired` from `plaintext_retires_at` on |

`from` in a message is the sender's account id. Times are RFC 3339.

## Sealed messages (contract v2)

Messages sealed on the sending device for each device that may read them. The envelope is the identity client's
`src/sealed.rs` (and the website's twin): AES-256-GCM per message, the content key wrapped per device with HPKE
(RFC 9180) Auth mode. Bytes travel as unpadded base64url. The service stores what the client sealed and checks only
shape, size, ownership and device coverage: it cannot tell that the bytes were encrypted, it never sees a message's
text or key, and it still sees who writes to whom, when, and how much.

| route | body (besides `client_id`, `refresh_token`) | replies |
|---|---|---|
| `device-register` | `public_key` (32 bytes) | 200 `{"device": {"id", "public_key", "created_at"}}`; the id is the first 16 bytes of SHA-256 over `alelyon.friends.device/v1` + NUL + the key, in hex (the service derives it); registering a live key again answers the same device; 409 `device-taken` (another account's key), `device-revoked` (a removed key never returns), `too-many-devices` (16 live); 10 a day |
| `devices` | `friend_id?` | 200 `{"devices": [device]}`: this account's live devices, or a friend's; 404 `not-friends` |
| `device-revoke` | `device_id` | **204**; 404 `no-such-device` |
| `send-sealed` | `friend_id`, `message_id` (`msg_` + 20-64 of `A-Z a-z 0-9 _ -`, made by the client, as the envelope's associated data binds it), `sender_device`, `nonce` (12 bytes), `ciphertext` (17-8016 bytes), `keys`: `[{"device", "enc" (32 bytes), "wrapped" (48 bytes)}]` (1-32, one per device) | 200 `{"message": entry}` with the sender's own key or null; 409 `devices-changed` with `devices` (the friend's live devices) when the keys miss one of them or name a device neither person has: seal again for that list; 409 `no-devices` (the friend has no device yet), `message-exists`; 404 `no-such-device` (not the caller's live device), `not-friends`; counts against `send`'s limit |
| `messages` | as above, plus `device_id` | with `device_id` (a live device of the caller): both kinds, oldest first, each entry with `kind`: `{"kind": "text", "id", "from", "text", "sent_at"}` or `{"kind": "sealed", "id", "from", "sender_device", "nonce", "ciphertext", "key": {"enc", "wrapped"} or null, "sent_at"}` (null when the message was not sealed for that device); 404 `no-such-device`. Without `device_id` a v1 client sees text messages only |

Once both people in a conversation have a live device, `send` (plain text) is refused with 409 `sealed-required`, so
an older app cannot silently downgrade a sealed conversation. Sealed messages count as unread and page with text
messages (one `seq` per conversation). Removing a friend deletes the sealed messages and their keys; devices stay.
Deleting an account deletes its devices, and the sealed messages and keys of its conversations.

## Unsealed messages retire (ADR-0049 Phase 4b)

An operator may set a date when unsealed (v1) messages end. From when it is set, `friends` carries it as
`plaintext_retires_at`, and clients tell people the date. From that instant, `send` answers 410 `plaintext-retired`,
`messages` returns no text entries (without `device_id`, an empty page), unread counts sealed messages only, and the
service deletes every stored v1 message, a batch each housekeeping pass (every 10 minutes). Without a date nothing
changes. A friend with no device can then not be written to until they open Friends on an updated app or the website.

## Rate limits

Per account: device registrations 10 a day; search 30 and request 20 per 15 minutes; presence 6, status changes 10 and send 60 per minute. A refusal is 429
`rate-limited` with `Retry-After` in whole seconds (at least 1).

## Presence and polling

The service stores each account's last reported state and its time. A friend whose last report is older than 150
seconds is shown `offline`, whatever it said. This is computed when `friends` is read: nothing in the service runs on
a timer.

- **Chosen status** (2026-10-10). What a friend sees combines the app's last report with the status the person chose: a report older than 150 seconds, or `offline`, shows `offline` whatever was chosen; otherwise `online` shows the report (so the app can still go `away` when idle), `away` and `dnd` show that word, and `offline` shows `offline` with `last_seen` null (a moving last-seen time would give the choice away). The app reports only `online`, `away` and `offline`; `dnd` is never a report. An older app that does not know `dnd` shows such a friend as offline until it is updated.

The app does the polling, and only while its window is open and signed in: `presence` on start (`online`) and every
60 seconds, `away` after 10 minutes without input, `offline` when it closes; `friends` every 30 seconds (every 10
seconds while a friends panel is open); `messages` every 5 seconds only while a conversation is open. Nothing while
closed or offline.

## Privacy and retention

- v1 messages (`send`) are stored on the server **in plain text**: anyone with access
  to the service's database can read them.
  Sealed messages (contract v2) are stored as the sending device sealed them; the app and the website send them by default since ADR-0049 Phase 4.
- A conversation is kept until the friendship ends (`remove` deletes it for both people), and every social record
  goes with its account. Read marks are kept per reader and friend. Friend requests stay until answered or cancelled.
- Disabled accounts cannot be found, requested, or seen in request lists. A friendship that already exists stays
  listed, with presence `offline` and `last_seen` null.
- The service logs no token, no search query and no message text.
