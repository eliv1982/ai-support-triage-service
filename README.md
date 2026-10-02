# ai-support-triage-service

[![CI](https://github.com/eliv1982/ai-support-triage-service/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eliv1982/ai-support-triage-service/actions/workflows/ci.yml)

A small portfolio/reference demo of **fail-safe LLM-assisted support-ticket triage**: a FastAPI service that asks a chat model for a draft triage of a customer message, validates the answer strictly, and escalates to a human whenever the model or the provider cannot be trusted.

It is a demo, not a production helpdesk. See [Limitations](#limitations-security-and-data-handling) for exactly what it does and does not cover.

## What it does

`POST /triage` takes a ticket text, a channel and a caller-supplied `client_id`, and returns a **draft** triage result:

- a `category`: `billing`, `support`, `complaint` or `other`;
- a short `draft_reply`;
- a `confidence`: `high`, `medium` or `low`;
- an `escalate` flag.

Every triaged ticket is stored in SQLite. Nothing is sent to the customer and no support action is executed: the result is meant to be reviewed by a person.

If the provider fails, times out, or returns something that does not match the expected schema, the service does not guess. It returns a fixed fail-safe result (`other`, `low` confidence, `escalate: true`) and records that it did so.

## Key engineering ideas

- **Model output is treated as untrusted.** JSON mode is requested, but the reply is still parsed and validated against a strict Pydantic model: exact keys, no extra keys, no type coercion (the string `"false"` is not a boolean), enum-checked `category` and `confidence`, non-empty `draft_reply`. Anything else falls back.
- **Fail-safe, not fail-open.** Provider error, timeout and invalid output all end in the same escalating result, so a failure costs one manual review instead of a wrong automatic answer.
- **Provenance is recorded.** Each row has `used_fallback` and, for fallbacks, a stable `error` code (`provider_error`, `provider_timeout`, `invalid_llm_json`, `invalid_llm_schema`). Raw SDK exception text is never stored or logged, since provider messages can quote credentials or the ticket itself.
- **Bounded provider calls.** Explicit timeout (`OPENAI_TIMEOUT_SECONDS`), SDK retries disabled, one OpenAI client reused for the whole process.
- **Minimal prompt.** The model receives only the ticket text and the channel. `client_id` is a local rate-limit key and is not sent.
- **Local rate limiting.** Thread-safe sliding window per `client_id`; idle clients are evicted lazily, so memory follows recently active clients rather than every id ever seen. Rejected requests never reach the provider and are not stored.
- **Responsive under load.** Triage is a synchronous endpoint running in the worker thread pool; `/health` is not, and a test checks it still answers while every worker is busy with triage.
- **Testable without a network.** The triage service, rate limiter and DB session are injected. API tests use a fake LLM client, a fixture blocks non-loopback sockets, and a loopback stub provider exercises the real SDK transport (timeouts, no retries, error statuses).
- **Plain operations.** Exactly pinned dependencies, a non-root Docker image, and CI that runs the suite on every push and pull request.

## Request flow

```text
Client
  |  POST /triage  {text, channel, client_id}
  v
FastAPI request validation ---------- invalid ------> 422
  v
In-memory rate limiter (per client_id) -- over limit -> 429   (nothing stored)
  v
TriageService  --- text + channel only --->  OpenAI-compatible Chat Completions
  v
Strict Pydantic validation of the model's JSON
  |-- valid -----------------------------------------> model's triage
  '-- provider error / timeout / invalid output -----> fail-safe triage (escalate: true)
  v
SQLite: one `tickets` row  (used_fallback, error code)
  v
200  {category, draft_reply, confidence, escalate}
```

## API

### `POST /triage`

| Field | Type | Constraints |
| --- | --- | --- |
| `text` | string | 1-2000 characters after surrounding whitespace is trimmed; blank text is rejected |
| `channel` | string | `email`, `form` or `chat` |
| `client_id` | string | 1-128 characters, one visible token: no spaces, control characters or invisible characters. UUIDs, e-mail addresses and ids like `acme:user.42` are fine. It is a label for rate limiting, **not** authentication |

POSIX shells:

```bash
curl -X POST http://127.0.0.1:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for my last invoice.", "channel": "email", "client_id": "client-123"}'
```

PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/triage `
  -ContentType "application/json" `
  -Body '{"text": "I was charged twice for my last invoice.", "channel": "email", "client_id": "client-123"}'
```

Example response (the wording is model-generated and will vary):

```json
{
  "category": "billing",
  "draft_reply": "Thank you for contacting us. We are reviewing the duplicate charge and will get back to you shortly.",
  "confidence": "high",
  "escalate": false
}
```

This is a **draft triage result**, not an executed support action. On a normal result `escalate` is the model's judgement; the service itself only forces it to `true` on the fail-safe path. The response does not say whether the fail-safe was used; that is recorded in the database (see [Stored data](#stored-data)).

### `GET /health`

Returns `{"status": "ok"}` while the process is up and serving requests. It is a liveness check only: it does not touch the database and does not call the provider, so it is not a readiness probe for either.

FastAPI's interactive docs are served at `/docs`.

### Behaviour

| Situation | HTTP | Stored as a ticket | Fail-safe used |
| --- | --- | --- | --- |
| Valid model result | 200 | yes | no (`used_fallback = 0`) |
| Provider error or timeout | 200, fail-safe result | yes | yes (`provider_error` / `provider_timeout`) |
| Invalid model output (not JSON, or wrong schema) | 200, fail-safe result | yes | yes (`invalid_llm_json` / `invalid_llm_schema`) |
| Request validation failure | 422 | no | n/a, provider not called |
| Rate limit exceeded | 429 `{"detail": "Rate limit exceeded"}` | no | n/a, provider not called |

The fail-safe result is category `other`, confidence `low`, `escalate: true`, with a fixed holding message as the draft reply. That message is currently Russian ("Thank you for your request. We have passed it to an operator for manual review and will reply as soon as possible."), a leftover from the original demo, while the prompt and everything else are English.

## Local setup

Requires Python 3.11 or newer. CI and the Docker image use 3.11, which is the only version the project is tested on.

POSIX shells (macOS/Linux):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # enough to run the service
pip install -r requirements-dev.txt      # also installs the test dependencies
cp .env.example .env
```

PowerShell (Windows):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # enough to run the service
pip install -r requirements-dev.txt      # also installs the test dependencies
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`, then start the service (same command in both shells):

```bash
uvicorn app.main:app
```

It listens on `http://127.0.0.1:8000`. Check it with `GET /health`, then send the request from the [API](#post-triage) section.

## Configuration

Settings are read from the environment or from a `.env` file in the working directory. `.env` is git-ignored and excluded from the Docker build context; never commit it.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | none, required | Provider API key. Startup fails if it is missing, blank, or still the `your_api_key_here` placeholder from `.env.example`. The format is not checked, so any other non-placeholder string works with an OpenAI-compatible local server that does not authenticate |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Chat Completions endpoint |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name |
| `OPENAI_TIMEOUT_SECONDS` | `10` | HTTP timeout for the provider call, applied by the client to each network phase (connect, write, read) rather than as one deadline for the whole call. There are no retries: a timeout or error goes straight to the fail-safe. Raise it for slower models |
| `RATE_LIMIT_PER_MINUTE` | `5` | Accepted requests per `client_id` in any 60-second window |
| `DATABASE_URL` | `sqlite:///./app.db` | SQLite file, relative to the working directory. Only SQLite is supported |

## Testing

```bash
python -m pytest
```

Run it from the repository root after `pip install -r requirements-dev.txt`. Use `python -m pytest`, not bare `pytest`: the former puts the repository root on `sys.path`, and a bare `pytest` fails with `ModuleNotFoundError: No module named 'app'`.

The tests need no real API key and make no external network calls: the test setup substitutes a fake key, replaces the LLM client, and refuses non-loopback connections. `pytest.ini` limits collection to `tests/`. CI runs the same command on Python 3.11.

## Docker

```bash
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
# set OPENAI_API_KEY in .env
docker compose up --build
```

The service is then on `http://127.0.0.1:8000`.

- The container runs as an unprivileged user (uid 10001). The application code in `/app` is baked into the image and is not bind-mounted from the host.
- SQLite lives under `/data` (`/data/app.db`), in the named volume `triage-data`, so data survives `docker compose down`. `docker compose down -v` deletes it.
- `.dockerignore` keeps `.env`, `*.db`, `.git`, virtual environments and caches out of the image.
- Compose reads `OPENAI_API_KEY` and the other settings from your `.env`.

To look at the stored tickets (the image has no `sqlite3` client, but it has Python):

```bash
docker compose exec api python -c "import sqlite3; print(sqlite3.connect('/data/app.db').execute('SELECT id, category, escalate, used_fallback, error FROM tickets').fetchall())"
```

This Compose file is a convenience for running the demo locally. It is not a hardened deployment: no TLS, no authentication, no health check or resource limits, and port 8000 is published on all host interfaces.

## Stored data

Each triaged ticket is one row in the `tickets` table: `id`, `created_at`, `client_id`, `channel`, `text`, the four result fields, `error` and `used_fallback`.

`used_fallback` is `1` when the stored result is the fail-safe rather than the model's answer. Content alone cannot tell them apart, because a model may answer exactly like the fail-safe. For fallbacks, `error` holds the stable reason code listed in the [Behaviour](#behaviour) table; it is `NULL` for normal results and never contains provider text.

To inspect a local database (works in any shell, no `sqlite3` client needed):

```bash
python -c "import sqlite3; print(sqlite3.connect('app.db').execute('SELECT id, client_id, category, escalate, used_fallback, error FROM tickets').fetchall())"
```

There is no migration framework. On startup the service creates the table if needed and, for a database made by an earlier version, adds the `used_fallback` column in place (rows with a non-empty `error` are marked `used_fallback = 1`). Nothing is deleted or rewritten, so older rows keep whatever they held, including raw provider error text from earlier versions.

## Limitations, security and data handling

This is a portfolio/reference demo, not a production support service.

- **No authentication.** Anyone who can reach the port can call `/triage`, which spends provider quota. `client_id` is supplied by the caller and is a label, not an identity: choosing a different one gets a fresh rate-limit allowance. The limiter is a cost and politeness guard, not a security boundary. Run the service on a trusted machine or network; the Compose file publishes the port on all interfaces, so change the mapping to `127.0.0.1:8000:8000` if that matters.
- **Rate limiter is in-memory and per process.** State is lost on restart and is not shared between workers or containers.
- **SQLite suits this demo, not a horizontally scaled deployment.** It is a single file with no migration framework.
- **Ticket text goes to the configured LLM provider** (OpenAI by default). `client_id` does not.
- **Ticket data is stored locally in plain SQLite**: `client_id`, ticket text and the generated reply. There is no automatic PII redaction and no retention or deletion policy.
- **Logs** contain `client_id`, channel and text length. They do not contain the ticket text or provider error messages.
- **The reply is a draft.** It should be reviewed by a person before anything reaches a customer. The prompt asks for 1-6 sentences, but that is a request to the model, not something that is validated.
- **Validation is client-side.** The service uses JSON mode and validates the result itself; it does not use provider-side Structured Outputs.
- **Throughput is demo-scale.** Each triage call holds a worker thread for as long as the provider call takes; the timeout is per network phase, not a hard cap on the whole call.

## Repository structure

```text
app/
  main.py               FastAPI app: /health and /triage
  schemas.py            request/response models and the strict LLM-output model
  config.py             settings and API-key startup validation
  services/
    triage_service.py   prompt, provider call, output validation, fail-safe
    llm_client.py       OpenAI client factory (explicit timeout, no retries)
  rate_limiter.py       in-memory sliding-window limiter
  database.py           engine, session, init_db (create + additive upgrade)
  models.py             SQLAlchemy `tickets` table
  repository.py         save_ticket
  dependencies.py       process-wide service and limiter providers
  logging_config.py     logging setup
tests/                  network-free pytest suite (fake LLM client, loopback provider stub)
Dockerfile              non-root image
docker-compose.yml      local demo run with a named data volume
.github/workflows/ci.yml
requirements.txt        runtime dependencies (pinned)
requirements-dev.txt    runtime + test dependencies
.env.example            configuration template (contains no secret)
```

## Possible next steps

Not implemented, and not needed for the demo: an endpoint to list and review stored tickets, authentication, an English (or localised) fail-safe message, schema migrations, and a shared rate-limit store for multi-process deployments.
