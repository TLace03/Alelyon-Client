# Pages contract (2026-10-09)

This contract covers public profiles, organizations with tickers, titles, and
posts, as `src/pages.rs` calls them. It uses the same conventions as the
[social contract](social-contract.md): JSON in and out, UTF-8, under the
service's base URL.

## Conventions

Every signed-in route is `POST /identity/native/pages/<name>`. Its JSON body
carries `client_id` and `refresh_token`, and these are checked **without
rotating the token**, as the social routes check them. Every route needs an
account with a username (409 `username-needed` otherwise).

The service also answers four **public reads** that need no credential:

- `GET /identity/pages/person?username=`
- `GET /identity/pages/org?ticker=`
- `GET /identity/pages/post?id=`
- `GET /identity/pages/explore`

Each takes an optional `before`. A public read applies no block, because its
reader is anonymous.

Errors have the native shape `{"error": "<code>", "message": "<words for a
person>"}`. The common errors are the social contract's. In addition, a text
that breaks one of the rules below answers 400 `invalid-text`, with the rule's
own words and never the text itself.

## Shapes

- An **org** is `{"id", "ticker", "name"}`. A ticker is 2-5 letters A-Z,
  stored upper case, and unique.
- A **person** has these fields:
  - `id`, `username`, `display_name`;
  - `title`: the person's own text, or empty;
  - `title_org`: the org the title names, or null;
  - `title_verified`: an owner or admin of `title_org` confirmed exactly this
    title;
  - `tickers`: the orgs the person chose to show, in their order;
  - `name_line`, for example `[ Founder | CEO ] Ada Lovelace (LACE)`;
  - `has_page`.

  A person is never an email address. Draw the verified mark from
  `title_verified` only. A display name is free text and may itself contain
  brackets.
- A **post** has these fields:
  - `id`;
  - `kind`: `post` or `article`;
  - `author`: a person;
  - `org`: the org it was posted as, or null;
  - `title`: an article's title;
  - `text`;
  - `reply_to`;
  - `repost_of`: the post it reposts or quotes, as the reader may see it, or
    null;
  - `repost_of_id`;
  - `created_at`, `edited_at`;
  - `deleted`;
  - `likes`, `replies`, `reposts`;
  - `liked`: null for an anonymous reader.

  An empty `text` with a `repost_of_id` is a plain repost. A `deleted` post is
  a tombstone, kept because replies hang under it, and its text and title are
  empty.
- A **page of posts** is `{"posts": [post], "more"}`, newest first, 30 at a
  time. Pass the last id as `before` to read the next page.

## Rules

- **A page is opt-in.** Until `open`, nothing about an account is public, and
  posting answers 409 `page-needed`. `close` withdraws the page. Its posts are
  kept, unseen, and show again if the page is opened again.
- **Titles.** A title has up to 4 parts joined by ` | `, each part 1-30
  characters, 60 characters in all. It may not contain `[ ] ( )`. One
  surrounding `[ ]` pair is accepted and dropped. An owner or admin of the org
  the title names may confirm it. The mark lasts while the title's text equals
  the confirmed text, so editing the title drops the mark.
- **Membership needs both sides.** Either the person asks (`org-join`) and an
  admin approves (`org-invite` of that person), or an admin invites and the
  person accepts (`org-join`).
- **Last owner.** An organization's last owner cannot leave or step down.
- **Post lengths.** A post has 1-500 characters. An article has a title of
  1-150 characters and a body of up to 20000. Text is trimmed and may contain
  no control character except newline.
- **Blocks.** A block in either direction does the following for signed-in
  routes:
  - hides each side's posts from the other;
  - refuses follows, replies, likes and reposts between them;
  - ends the follows both ways;
  - hides the blocker's page from the blocked person.

## Routes

| route | body (besides `client_id`, `refresh_token`) | replies |
|---|---|---|
| `me` | | 200 `{"me": person, "bio", "memberships": [{"org", "role", "state", "shown", "confirmed_title", "since"}]}` |
| `open`, `close` | | 200, as `me` |
| `update` | `title?`, `title_org?` (a ticker; empty for none; an active membership), `bio?` | 200, as `me`; 409 `page-needed`, `not-a-member` |
| `tickers` | `tickers` (a list, in order) | 200, as `me`; each must be an active membership; every other one is hidden |
| `person` | `username`, `before?` | 200 `{"person", "bio", "followers", "following", "posts", "more", "you_follow", "you_block"}`; 404 `no-such-page` |
| `org` | `ticker`, `before?` | 200 `{"org", "about", "created_at", "followers", "members": [person], "posts", "more", "you_follow", "your_role", "your_state"}`; 404 `no-such-org` |
| `org-create` | `ticker`, `name`, `about?` | 200 `{"org"}`; 400 `invalid-ticker`; 409 `ticker-taken`, `ticker-reserved`, `too-many-orgs` (10 owned) |
| `org-update` | `ticker`, `name?`, `about?` | 200 `{"org"}`; 403 `not-org-admin` |
| `org-members` | `ticker` | 200 `{"members": [{"person", "role", "state", "confirmed_title", "since"}]}`; owners and admins only |
| `org-join` | `ticker` | 200 `{"state": "requested"\|"member"}` |
| `org-invite` | `ticker`, `username` | 200 `{"state": "invited"\|"member"}`; owners and admins only |
| `org-leave` | `ticker` | **204**; leaves, withdraws a request or declines an invitation; 409 `last-owner` |
| `org-remove` | `ticker`, `member_id` | **204**; only an owner removes an admin; nobody removes an owner |
| `org-role` | `ticker`, `member_id`, `role` (`owner`\|`admin`\|`member`) | **204**; owners only |
| `org-confirm-title` | `ticker`, `member_id`, `confirm` (a JSON boolean) | 200 `{"confirmed_title"}`; 409 `no-title-to-confirm` |
| `follow` | `username` or `ticker`, `follow` (a JSON boolean, default true) | **204** |
| `block` | `username`, `block` (a JSON boolean, default true) | **204** |
| `post` | `text`, `kind?`, `title?`, `reply_to?`, `repost_of?`, `as_org?` | 200 `{"post"}`; 409 `already-reposted` (a second plain repost) |
| `edit` | `post_id`, `text`, `title?` | 200 `{"post"}`; 409 `cannot-edit-repost` |
| `delete` | `post_id` | **204**; the author, or an owner or admin of the org it was posted as |
| `like` | `post_id`, `like` (a JSON boolean, default true) | **204** |
| `thread` | `post_id`, `before?` | 200 `{"post", "parent", "replies", "more"}`: replies oldest first |
| `feed` | `before?` | 200 a page of posts: the reader's own and those of the people and organizations they follow |
| `explore` | `before?` | 200 a page of posts: every top-level post |
| `report` | `reason` (`spam`\|`harassment`\|`impersonation`\|`illegal`\|`other`), `note?`, one of `post_id`, `username`, `ticker` | 200 `{"report_id"}` |

## Rate limits

The signed-in routes are limited per account:

| routes | limit |
|---|---|
| reads (`me`, `person`, `org`, `org-members`, `thread`, `feed`, `explore`) | 240 per minute |
| `post` | 30 per hour |
| `org-create` | 5 per day |
| `report` | 20 per hour |
| every other route | 300 per hour |

The public reads are limited to 120 per minute per source address. A refusal
is 429 `rate-limited`, with `Retry-After` in whole seconds.

## Privacy

- A page and its posts are **public**: anyone can read them, signed in or not.
  They are stored on the server in plain text.
- A report goes to Alelyon's operators. A post they hide, or an organization
  they suspend, is gone from every read.
- The service logs no token, no post text and no query.
