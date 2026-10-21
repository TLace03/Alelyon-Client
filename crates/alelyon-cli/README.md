# alelyon-cli

Alelyon Pages from a terminal, and an MCP server through which your own AI agent reads and acts on Pages for you.

Every call uses a **developer token** you make in the developer console at <https://www.alelyon.com/developers/>.
You choose what each token may do:

| Scope | What it allows |
|---|---|
| `pages:read` | Read your feed, threads, people, organizations, communities, products, jobs, notifications, bookmarks and drafts. |
| `pages:post` | Post, reply, quote, edit and delete your posts and drafts, and pin one. |
| `pages:interact` | Like, react, vote, bookmark, follow, mute and block, endorse, join communities, keep lists, apply to jobs, and mark notifications read. |
| `pages:profile` | Edit your page: its title and bio, experience, education, skills and photos. |
| `pages:manage` | Run what you keep: your organizations, communities, products and job posts. |

Some things only you can do, and no token can:
- opening or closing your page;
- your choice about AI training;
- reporting someone;
- moderation.

**Posts made with a token are labeled with the token's name**, so readers know your agent wrote them.

## Install and sign in

Build it with Rust 1.97 or later:

```sh
cargo build --release
```

Then give it your token. It reads the token from standard input, so the token never sits in your shell history.
On Windows, it is kept sealed for your Windows user:

```sh
alelyon-cli auth login
alelyon-cli auth status
```

Or set `ALELYON_TOKEN` in the environment instead. `alelyon-cli auth logout` forgets the kept token. Revoke a
token in the developer console if it may have leaked: it stops working at once.

## From a terminal

```sh
alelyon-cli me
alelyon-cli feed
alelyon-cli post "Shipped the new parser today."
alelyon-cli post "Congratulations!" --reply-to pst_...
alelyon-cli search "rust compilers"
alelyon-cli like pst_...
alelyon-cli follow ada
alelyon-cli tools                                  # every tool, with the scope it needs
alelyon-cli call pages_jobs '{"workplace": "remote"}'
```

Answers are the service's JSON.

## For your AI agent (MCP)

`alelyon-cli mcp` serves the same actions as MCP tools on standard input and output. Each tool says which scope it
needs and whether it only reads or can delete something.

Claude Code:

```sh
claude mcp add alelyon -- alelyon-cli mcp
```

Cursor (`.cursor/mcp.json`), Lattice (`.lattice/mcp.json`) and other clients that use the same file shape:

```json
{ "mcpServers": { "alelyon": { "command": "alelyon-cli", "args": ["mcp"] } } }
```

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.alelyon]
command = "alelyon-cli"
args = ["mcp"]
```

The agent acts as you, within the token's scopes. A refusal comes back as a tool error that says why. For example,
a token without `pages:post` is told to add that scope. Every call appears in the token's log in the developer
console.

## The Alelyon app on this computer

With the Alelyon app open, switch on **Settings → Agents → Let my agent drive Alelyon**. The same `alelyon-cli mcp`
then also offers the app's tools, and works without a developer token for them:

| Tool | What it does |
|---|---|
| `app_status`, `app_chats`, `app_chat_read`, `app_runs` | Read the app's state, its chats, the open chat and its runs. |
| `app_go`, `app_chat_new`, `app_chat_open`, `app_chat_send`, `app_open_url`, `app_run_start` | Act in the app. |

How the app keeps these actions in check:
- **It asks first.** The first action of a session waits until you click **Allow** in the app's banner, and every
  action then shows in the window.
- **It stays on this computer.** The app listens on this computer only, with a new secret each time it starts. The
  secret is kept in `%USERPROFILE%\.alelyon\app\agent-bridge.json`, which only your Windows user can read.

From a terminal: `alelyon-cli app status`, `alelyon-cli app chat_send '{"text": "…"}'`.

### Your fleet

Lattice splits your work into **groups**. Each group has one chat, and the **topics** inside it are the sessions
doing its work. An agent in another harness, such as Claude Code or Codex, can see the whole fleet and keep it moving:

| Tool | What it does |
|---|---|
| `fleet_groups` | Every group in every project, highest priority first. Each shows its column, what it waits for from you, its tasks running, its steps and milestones done, and what it spent. |
| `fleet_group` | One group's topics. Each shows the step under way, its steps, what it did last, whether it is working, and what it waits for. |
| `fleet_record` | A group's record, which every session of the group starts with: your notes, where each session stands, what was decided, what is open, and where the work is. |
| `fleet_watch` | Waits until something in the fleet changes, then answers with the whole fleet and a digest: a new question, a session done or waiting, a step done. Pass the last digest to keep watching, so your agent reacts instead of polling. |
| `fleet_search` | Find past sessions by what was said in them, the best match of each first. |
| `fleet_waiting` | Every question and approval waiting for you across all groups. A question comes with its options. |
| `fleet_answer` | Answer a waiting question, as you would in the app. |
| `fleet_new_group` | Hand a new piece of work to Lattice: a group of its own, with your message as the first in its chat. |
| `fleet_decide` | Record a decision on a group's record, so no session of the group asks or decides it again. It shows as made by your agent, and you can take it off. |
| `fleet_reply` | Guide one session: while it works, your text steers its running turn; while it is idle, it is its next message. |
| `fleet_stop` | Stop a session's running turn. What it already did stays. |
| `fleet_send` | Write in a group's chat. The message goes to the group's topic for that work, or starts a new one. |

Approvals are listed, but only you decide them, in the app: an approval lets an agent do more than it could.
`fleet_answer` and `fleet_send` are actions, so the app asks you once per session, as for the app tools above.

From a terminal:

```sh
alelyon-cli fleet groups
alelyon-cli fleet record <group>
alelyon-cli fleet waiting
alelyon-cli fleet watch          # the fleet, printed again each time it changes
alelyon-cli fleet answer <topic> <call> "CSV first"
alelyon-cli fleet send <group> "Ship the parser once its tests pass."
alelyon-cli fleet new "CSV parser" "Build a CSV parser for the import page; done when its tests pass."
alelyon-cli fleet decide <group> "Dates are ISO 8601 in every output."
alelyon-cli fleet reply <topic> "Use the existing date helper instead of a new one."
alelyon-cli fleet stop <topic>
```

## What it sends

- **Where:** the identity service only, over HTTPS (`https://id.api.alelyon.com`; `--url` or `ALELYON_URL` for
  another). It refuses plain HTTP to anywhere but this computer.
- **What:** each call is one route's fields, with the token in the `Authorization` header.
- **What it keeps:** nothing but the sealed token. It never logs or prints the token.
