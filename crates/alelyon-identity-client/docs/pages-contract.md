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

The service also answers seven **public reads** that need no credential:

- `GET /identity/pages/person?username=`
- `GET /identity/pages/org?ticker=`
- `GET /identity/pages/post?id=`
- `GET /identity/pages/explore`
- `GET /identity/pages/search?q=&kind=`
- `GET /identity/pages/tag?tag=`
- `GET /identity/pages/trending`

Each but `trending` takes an optional `before`. A public read applies no block, because its
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
  - `author`: a person, or null for a post made as an org whose author is
    gone (the account is no longer active, or its page is suspended). Such a
    post stays the org's;
  - `author_flag`: why `author` is null, `deleted` or `suspended`. Only the
    org's active owners and admins are told; for everyone else it is null;
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
  - `liked`: null for an anonymous reader;
  - `bookmarked`: whether the reader saved it; null for an anonymous reader.

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
| `me` | | 200 `{"me": person, "bio", "memberships": [{"org", "role", "state", "shown", "confirmed_title", "since"}], "moderator", "suspended", "ai_training", "unread_notifications"}` |
| `open`, `close` | | 200, as `me` |
| `update` | `title?`, `title_org?` (a ticker; empty for none; an active membership), `bio?` | 200, as `me`; 409 `page-needed`, `not-a-member` |
| `tickers` | `tickers` (a list, in order) | 200, as `me`; each must be an active membership; every other one is hidden |
| `notifications` | `before?` | 200 `{"notifications": [{"id", "kind", "actor": person or null, "post": post or null, "org": org or null, "at", "read"}], "more", "unread"}`, newest first. `kind` is `like`, `reply`, `repost`, `quote`, `mention`, `follow`, `org-invite`, `org-request`, `org-joined` or `title-confirmed`. One whose actor cannot be shown, or about a post the reader cannot see now, is left out. Each is deleted 90 days after it was made |
| `bookmark` | `post_id`, `bookmark?` (default `true`) | **204**; saves the post for the reader, or forgets it. Only the reader sees their bookmarks; a plain repost saves what it carries; 404 `no-such-post` |
| `bookmarks` | `before?` | 200 `{"posts", "more"}`: the reader's bookmarks, the most recently saved first; one the reader can no longer see is left out |
| `pin` | `post_id` | **204**; pins one's own top-level post, quote or article to the top of one's page, replacing any other; 404 `no-such-post` for anything else |
| `unpin` | | **204** |
| `notifications-read` | `up_to?` | 200 `{"unread"}`: marks every notification read, or those up to and including `up_to` in the list's order |
| `ai-training` | `allow` (`true` or `false`) | 200, as `me`: whether Alelyon may use the account's public posts to improve its services, AI models included. Allowed until the account opts out; it needs no page |
| `person` | `username`, `before?` | 200 `{"person", "bio", "followers", "following", "posts", "more", "you_follow", "you_block", "pinned"}` (`pinned`: the post pinned to the top of the page, on the first page only, or null); 404 `no-such-page` |
| `org` | `ticker`, `before?` | 200 `{"org", "about", "created_at", "followers", "members": [person], "posts", "more", "you_follow", "your_role", "your_state"}`; 404 `no-such-org` |
| `org-create` | `ticker`, `name`, `about?` | 200 `{"org"}`; 400 `invalid-ticker`; 409 `ticker-taken`, `ticker-reserved`, `too-many-orgs` (10 owned) |
| `org-update` | `ticker`, `name?`, `about?` | 200 `{"org"}`; 403 `not-org-admin` |
| `org-members` | `ticker` | 200 `{"members": [{"person", "role", "state", "confirmed_title", "since"}]}`; owners and admins only |
| `org-flagged` | `ticker`, `before?` | 200 `{"posts", "more"}`: the org's posts whose author is gone, newest first, each with its `author_flag`; owners and admins only (403 `not-org-admin`) |
| `org-join` | `ticker` | 200 `{"state": "requested"\|"member"}` |
| `org-invite` | `ticker`, `username` | 200 `{"state": "invited"\|"member"}`; owners and admins only |
| `org-leave` | `ticker` | **204**; leaves, withdraws a request or declines an invitation; 409 `last-owner` |
| `org-remove` | `ticker`, `member_id` | **204**; only an owner removes an admin; nobody removes an owner |
| `org-role` | `ticker`, `member_id`, `role` (`owner`\|`admin`\|`member`) | **204**; owners only |
| `org-confirm-title` | `ticker`, `member_id`, `confirm` (a JSON boolean) | 200 `{"confirmed_title"}`; 409 `no-title-to-confirm` |
| `follow` | `username` or `ticker`, `follow` (a JSON boolean, default true) | **204** |
| `block` | `username`, `block` (a JSON boolean, default true) | **204** |
| `post` | `text`, `kind?`, `title?`, `reply_to?`, `repost_of?`, `as_org?` | 200 `{"post"}`; 409 `already-reposted` (a second plain repost) |
| `edit` | `post_id`, `text`, `title?` | 200 `{"post"}`; 409 `cannot-edit-repost`, `poll-locked` (a post with a poll) |
| `delete` | `post_id` | **204**; the author, or an owner or admin of the org it was posted as |
| `like` | `post_id`, `like` (a JSON boolean, default true) | **204** |
| `thread` | `post_id`, `before?` | 200 `{"post", "parent", "replies", "more"}`: replies oldest first |
| `feed` | `before?` | 200 a page of posts: the reader's own and those of the people and organizations they follow |
| `explore` | `before?` | 200 a page of posts: every top-level post |
| `search` | `q` (1-100 characters, at most 5 words used), `kind?` (`all`\|`posts`\|`people`\|`orgs`, default `all`), `before?` (a `posts` search only) | 200 `{"people"?, "orgs"?, "posts"?, "more"?}`, each list only when `kind` asks for it (`all`: 5 people, 5 organizations and a page of posts). People: an open page whose username starts with the first word (an `@` is allowed) or whose name holds every word. Organizations: a ticker starting with the first word (a `$` is allowed) or a name holding every word. Posts, replies included: every word in the words or the title, newest first; one `#tag` alone reads that tag. 400 `invalid-query` |
| `tag` | `tag` (with or without `#`), `before?` | 200 `{"tag", "posts", "more"}`: the posts carrying it, newest first. A hashtag is `#` and 1-50 letters, digits or `_` with a letter, not glued to a word before it; a post keeps at most 10. 400 `invalid-tag` |
| `trending` | | 200 `{"hours", "tags": [{"tag", "posts", "people"}]}`: the hashtags the most different people used over the last 24 hours (or 168, when none qualified that day), at least 2 people each, at most 10 |
| `suggestions` | | 200 `{"people": [{"person", "reason", "org"}], "orgs"}`: who to follow from public signals only (`reason`: `org` shows an organization the reader belongs to or follows, named in `org`; `replied`; `you-replied`; `popular`; `page`), at most 10, and up to 5 organizations |
| `for-you` | `page?` (0-9, a JSON integer) | 200 a page of posts: others' top-level posts of the last 3 days, ranked by replies, reposts, likes, recency and closeness (someone or an organization the reader follows, a hashtag the reader used) |
| `experience` | `entry_id?` (to replace one), `title` (1-100), `ticker` (an Alelyon organization) or `org_name` (1-120, an employer as written), `started` (`YYYY-MM`), `ended?` (`YYYY-MM`, absent while current), `about?` (0-2,000) | 200 `{"experience"}`: the reader's entries, current first. A change clears a confirmation. At most 30; 409 `too-many-entries`; 404 `no-such-entry`, `no-such-org` |
| `experience-delete` | `entry_id` | 200 `{"experience"}` |
| `experience-confirm` | `ticker`, `entry_id`, `confirm?` (default true) | **204**; an owner or admin of the organization the entry names confirms it, or takes that back; 403 `not-org-admin` |
| `org-experience` | `ticker` | 200 `{"entries": [{"person", "entry"}]}`: for owners and admins, the entries that name the organization, the unconfirmed first |
| `education` | `entry_id?`, `school` (1-120), `degree?`, `field?` (0-120), `started?`, `ended?` (`YYYY`) | 200 `{"education"}`; at most 20 |
| `education-delete` | `entry_id` | 200 `{"education"}` |
| `skills` | `skills` (a list, at most 50, each 1-60) | 200 `{"skills"}`: the reader's, in its order, one of each (case does not matter); a dropped skill's endorsements go |
| `endorse` | `username`, `skill`, `endorse?` (default true) | **204**; never one's own; only the totals are shown; 404 `no-such-page`, `no-such-skill` |
| `mute` | `username`, `mute?` (default true) | **204**; silent: the muted person's posts leave the reader's feeds, searches and hashtag lists, and their acts the reader's notifications; their page still shows them. 404 `no-such-person` |
| `mutes` | | 200 `{"people"}`: whom the reader muted, the most recent first |
| `jobs` | `q?`, `workplace?`, `employment?`, `ticker?`, `before?` | 200 `{"posts", "more"}`: open jobs, newest first; searched, so bound as `search` is; 404 `no-such-org` |
| `job-apply` | `post_id`, `note?` (at most 1,000 characters), `apply?` (default true) | **204**; again changes the note; `apply: false` withdraws, deleting it; 404 `no-such-job`, 409 `job-closed`, `own-job` (an owner or admin of its organization) |
| `job-applicants` | `post_id` | 200 `{"applicants": [{"person", "note", "applied_at"}]}`, the earliest first; owners and admins only (403 `not-org-admin`); 404 `no-such-job` |
| `job-close` | `post_id` | **204**; owners and admins only; it takes no more applications |
| `applications` | | 200 `{"posts"}`: the jobs the reader applied for, the latest first |
| `vote` | `post_id`, `option` (a JSON integer, 0 first) | **204**; one vote, never changed, never in one's own poll; 404 `no-such-poll`, 409 `poll-closed`, `already-voted`, `own-poll` |
| `report` | `reason` (`spam`\|`harassment`\|`impersonation`\|`illegal`\|`other`), `note?`, one of `post_id`, `username`, `ticker` | 200 `{"report_id"}` |

## Moderation

The service names its moderators in its configuration: the accounts whose
usernames it lists. For them, `me` answers `"moderator": true`. Anyone else
gets 403 `not-a-moderator` from these four routes. They are open to the app
only, not to the website. Each is `POST /identity/native/pages/<name>`, with
`client_id` and `refresh_token` in the body as above.

| route | body | replies |
|---|---|---|
| `mod-queue` | `state?` (`open`, the default, or `closed`), `before?` | 200 `{"reports": [report], "more"}`: open reports oldest first, closed ones newest first |
| `mod-act` | `action`, then the target as `post_id`, `username` or `ticker`, or `report_id` alone (or both, which must agree), `note?` | 200 `{"action": action}`; 404 `no-such-post`, `no-such-person`, `no-such-org`, `no-such-report`; 409 `wrong-target` |
| `mod-dismiss` | `report_id`, `note?` | 200 `{"action": action}`; 404 `no-such-report`; 409 `report-closed` |
| `mod-history` | `before?` | 200 `{"actions": [action], "more"}`, newest first |

- **Actions:** `hide-post`, `restore-post`, `suspend-page`, `restore-page`,
  `suspend-org`, `restore-org`. Hiding or suspending closes every open report
  on that target as `actioned`.
- **A suspended page** shows nothing: no page and none of that person's posts,
  posts made as an organization included. Its owner gets 403
  `page-suspended` from `open`, `update`, `post`, `edit`, `like`, `follow`
  and `org-create`, and `me` answers `"suspended": true`. Closing one's own
  page does not end a suspension; a moderator's `restore-page` does.
- **A report** has these fields:
  - `id`, `reason`, `note`, `at`;
  - `state`: `open`, `actioned` or `dismissed`;
  - `reporter`: a person;
  - `target`;
  - `open_on_target`: open reports on the same target;
  - `resolved_at`, `resolved_by`.

  A **target** has these fields:
  - `kind`: `post`, `person`, `org`, or `gone`;
  - `id`;
  - `post`;
  - `person`: the reported person or a post's author;
  - `org`;
  - `text`: a biography or an organization's about text;
  - `hidden`;
  - `suspended`.

  Hidden and suspended things are included.
- **An action** has these fields:
  - `id`, `at`;
  - `action`: one of the above, or `dismiss`;
  - `moderator`: a person, or null with `by_host: true` for an act taken on
    the service's own host;
  - `target_kind`: `post`, `person`, `org` or `report`;
  - `target_id`, `report_id`, `note`.

  The service keeps every action in a log its database will not let anyone
  rewrite.

## Rate limits

The signed-in routes are limited per account:

| routes | limit |
|---|---|
| reads (`me`, `person`, `org`, `org-members`, `org-flagged`, `thread`, `feed`, `explore`, `notifications`, `bookmarks`, `tag`, `trending`, `suggestions`, `for-you`) | 240 per minute |
| `search` | 60 per minute |
| `post` | 30 per hour |
| `org-create` | 5 per day |
| `report` | 20 per hour |
| every other route | 300 per hour |

The moderators' routes are limited to 600 per hour per moderator. The public
reads are limited to 120 per minute per source address, the public search to 30. A refusal
is 429 `rate-limited`, with `Retry-After` in whole seconds.

## Privacy

- A page and its posts are **public**: anyone can read them, signed in or not.
  They are stored on the server in plain text.
- A report goes to Alelyon's operators. A post they hide, or an organization
  they suspend, is gone from every read.
- The service logs no token, no post text and no query.
- `post` also takes `replies` (`everyone`, `followers` or `mentioned`; not on a
  reply, which keeps its conversation's rule, nor on a plain repost) and a `poll`
  (`{"options": [2-4 texts of 1-25], "hours": 1-168}`, on a top-level post
  only). A post says `replies_from`, `you_can_reply` and `poll`: `{"options",
  "votes", "total", "closes_at", "closed", "you_voted"}`, where `votes` is null
  until the reader voted or the poll closed (its author always sees them). The
  conversation's first post sets the rule: its author's (or organization's)
  followers, or the people it mentions, and its author always may reply. A
  reply against the rule is 403 `replies-limited`. A person page says `you_mute`.
- `post` also takes a `job` on an article posted as an organization:
  `{"workplace": "on-site" | "hybrid" | "remote", "employment": "full-time" |
  "part-time" | "contract" | "temporary" | "internship", "location"?, "pay"?,
  "days"? (1-90, default 30)}`; a location is needed unless remote. A post says
  `job`: `{"location", "workplace", "employment", "pay", "closes_at", "closed",
  "applied", "applicants"}`, where `applicants` (a count) is for the
  organization's owners and admins only, and null otherwise.
- A person page's first page and `me` carry `profile`: `{"experience", "education",
  "skills": [{"skill", "endorsements", "you_endorsed"}]}`. Who endorsed a skill is
  never shown, only how many.
- `suggestions` and `for-you` use only what is public, and the reader's own
  follows: never whom someone else follows or what anyone liked.
