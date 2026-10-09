# Lattice (native)

Lattice is the Alelyon desktop app's chat and coding agent: agent runs and their traces,
a chat that reads, edits and runs code in a folder you choose, and a window drawn on the
GPU. It is written in Rust.

- **The runtime** is a port of the OpenAI Agents SDK (openai-agents-python 0.22.3, MIT;
  see `crates/lattice-agents/NOTICE`).
- **The window** is drawn with [iced](https://iced.rs) over wgpu, and only when something
  changes.

## Crates

| Crate | What it holds |
|---|---|
| `lattice-protocol` | The contract between the runtime and a window: runs, their events and trace spans in the SDK's export shape; chat threads, turns and answer events; agent conversations, staged changes, approvals and checkpoints. Neither service trait has a delete. |
| `lattice-agents` | The SDK port: agents, tools, handoffs, guardrails, streamed runs, cancellation, an OpenAI-compatible Chat Completions model, a scripted test model, local-only tracing and the agent graph. |
| `lattice-core` | The client services: the state folder, keys and the model registry, the run store and manager, the plain chat core and the agent chat (staged edits with a review, a policy that allows, asks or refuses each tool call, checkpoints, commands, skills, plug-ins, projects, MCP servers, a local llama.cpp server it manages, and the agent's browser). Files are never deleted: a file is moved aside instead. |
| `lattice-sys` | The only crate allowed `unsafe`: thin Win32 calls (file identity, opening without following links, `ReplaceFileW`, Job Objects and process spawn, Credential Manager, and the desktop for auto mode). |
| `lattice-app` | The native window over a `RunService`: the run list, a timeline of each run's spans, the agent graph and span detail. The `lattice` binary wires in the real runtime, or demonstration data with `--demo`. |

## Build, run, test

Rust 1.97 or later. The lockfile is committed; build with `--locked`.

```bash
cd lattice
cargo test --locked --workspace                     # every crate's tests; no window, no GPU device
cargo run --release -p lattice-app -- --demo        # demonstration data, no runtime
cargo run --release -p lattice-app -- --dev-model   # the real runtime, plus a scripted development model
```

Some tests compare Lattice with the Python implementation it was ported from, or start
helpers that exist only in Alelyon's source repository. Outside it they print `SKIPPED`
and why, and return; the golden files they were recorded into are here and are checked
by the tests that read them. Windows is the platform Lattice is built and tested on.
