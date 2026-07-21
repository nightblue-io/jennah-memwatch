# memwatch - an autonomous watcher that runs on a schedule

A small demo **agent** that consumes Jennah's public memory APIs the way any
external agent would: plain HTTP/JSON through the `jennah-proxy` gateway,
authenticated with a `jennah_sk_` API key. No Jennah server internals are
imported - this is a standalone Go module, so it doubles as a reference for
outside integrators.

**memwatch** is *proactive and headless*. You give it a subject to watch and walk
away. Run it on a cron; each run it observes what's noteworthy now, **diffs that
against what it already recorded in Jennah**, and reports only **what's new since
the last run**. The entity graph accretes across runs and the execution log
becomes a run-by-run timeline.

The point it makes: **Jennah is the durable brain an unattended agent runs its
entire life against.** Kill it, cron it, resume it days later - it never
re-reports what it already knows.

## What it does each run

```
(cron fires) ─┐
              ├─ memory:query (log, limit 1) ── when did I last run?
              ├─ memory:query (graph, 1 hop)  ── which entities do I already track?
   observe →  │
   diff →     ├─ brain.observe ── LLM reports notable developments (entities + relations)
   commit     ├─ per item: memory:query (semantic) ── nearest past chunk closer than
              │             the dedup threshold? → KNOWN (skip), else → NEW
              └─ memory:commit ── the NEW items as vector chunks + their
                                  entities/relations as graph nodes/edges
                                  + one run record to the log (atomic)
```

- **Vector chunks** hold each new development, so next run's semantic diff can tell
  new from already-seen (cosine distance; smaller = more similar).
- **Graph** grows an entity map: every named entity is linked to the stable
  `subject` anchor via `MENTIONS`, and entity-to-entity `relations` become edges -
  so a traversal from the anchor enumerates the watch list, and a two-hop walk
  reads back *how* things relate.
- **Execution log** is the run timeline: one step per run (`"N new, M known"`),
  newest first - which is also how the banner knows when you last ran.

Cross-run memory is just **reusing the same `agent_instance_id`**, persisted to
`memwatch-state.json` along with the pinned subject (so cron entries don't repeat
`-subject`). Graph writes are idempotent server side with content-hashed ids, so an
entity re-seen on a later run just converges instead of fragmenting - the client
keeps no id ledger.

> **Where the "news" comes from.** For a self-contained demo the developments come
> from the chat model itself (LLM-as-source). A real deployment swaps a web-search /
> RSS / news API in at the `brain.observe` seam - every Jennah memory call stays
> exactly the same.

## Prerequisites

1. A Jennah API key for an **approved, entitled** enterprise. Mint one after
   logging in (console or `jnh`):
   `POST /v1/apikeys {"label":"memwatch"}` → copy the `secret` (shown once).
2. A chat model - Anthropic, or Gemini (via **Google AI Studio** with an API key,
   or via **Vertex AI** with a GCP project + ADC).

The observing brain is pluggable: only the LLM differs, every Jennah memory call is
identical. `-provider auto` (the default) picks **Anthropic** when an Anthropic key
is configured, otherwise **Gemini**; force it with `-provider gemini|anthropic`.
Within Gemini, Vertex is used when `GOOGLE_GENAI_USE_VERTEXAI=true` or when
`GOOGLE_CLOUD_PROJECT` is set and no Studio key is present.

The agent's home region is chosen at creation with `-region` (or `$JENNAH_REGION`);
it's applied only on first launch, since an agent is pinned to one region for its
lifetime, and empty uses the platform default. List the available regions with
`jnh agents regions`. The target region must have managed embeddings configured
(prod `db0001` / `us-central1` does) - the demo sends plain text and lets the
server embed it.

## Run

```sh
export JENNAH_API_KEY=jennah_sk_...

# first run: name the subject (remembered thereafter). Anthropic:
export ANTHROPIC_API_KEY=sk-ant-...
go run . -subject "the AI agent memory / context platform market"

# …or Gemini via Google AI Studio (API key):
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
go run . -subject "the AI agent memory / context platform market"

# …or Gemini via Vertex AI (GCP project + ADC, no API key):
gcloud auth application-default login          # once
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=my-gcp-project
export GOOGLE_CLOUD_LOCATION=us-central1       # optional; defaults to "global"
go run . -subject "the AI agent memory / context platform market"

# subsequent runs: subject is remembered, just run it again
go run .
go run . -verbose                          # show observations, dedup decisions, receipts
go run . -show                             # print the entity graph + run timeline, then exit
go run . -provider gemini                  # force a provider regardless of which keys are set
go run . -max-items 12 -dedup-distance 0.1 # more items per run; stricter "already known"
go run . -endpoint http://127.0.0.1:8090   # against a local proxy instead
go run . -region us-central1               # pin the agent's home region (or $JENNAH_REGION)
```

On start it prints the chosen brain and the last run, e.g.:

```
chat model: anthropic/claude-sonnet-5
watching "the AI agent memory / context platform market" — last run 2026-07-22T06:00:11Z

3 new since last run:
  • Foo raises Series B to expand its agent-memory offering
      …
  • Bar ships a vector+graph unified store
      …

(5 already known, skipped)
```

## Run it on a schedule

The whole point is unattended, repeated runs. Build once and cron the binary
(env in the crontab so the job has your keys):

```crontab
# every hour, on the hour
JENNAH_API_KEY=jennah_sk_...
ANTHROPIC_API_KEY=sk-ant-...
0 * * * * cd /path/to/jennah-memwatch && ./jennah-memwatch >> memwatch.log 2>&1
```

```sh
go build .   # produces ./jennah-memwatch
```

Each firing appends a run to the log and only surfaces genuinely new developments;
`./jennah-memwatch -show` any time to review the accumulated graph and timeline.

## Notes

- Each provider defaults to a snappy/cheap model (`claude-sonnet-5`,
  `gemini-2.5-flash`); edit `anthropicModel` in `brain_anthropic.go`
  (→ `anthropic.ModelClaudeOpus4_8`) or `geminiModel` in `brain_gemini.go`
  (→ `gemini-2.5-pro`) for max capability. Backends live behind the `brain`
  interface in `brain.go`.
- The observing tool is **forced** (Anthropic `tool_choice`, Gemini
  `FunctionCallingConfig` mode `ANY`), so the model always returns structured
  developments rather than prose - the two backends decode the identical shape.
- The subject is **pinned at creation** (it's the `subject` anchor node's label);
  a later `-subject` is ignored in favor of the remembered one. Delete
  `memwatch-state.json` to start watching something else from scratch.
- `-verbose` surfaces the memory activity live: how many entities are already
  tracked, each dedup decision (`known (d=…)` / `NEW (d=…)`), and the commit
  receipt (`log=… vec=… nodes=… edges=…`) - handy when demoing.
- A run with nothing new still writes its log step, so the timeline faithfully
  records that the watcher ran.
- Fusion (`link:true`) is intentionally not used - it returns Unimplemented.
