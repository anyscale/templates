# Streamlit UI (optional demo frontend)

The main notebook ends at the Ray Serve `/recommend` endpoint. `streamlit_app.py`
is an optional browser frontend on top of that endpoint: upload a product image,
see the BLIP caption and the top-5 recommendations come back from the service.

It is intentionally **not** part of the main notebook flow — the notebook proves the
Ray pipeline; this is just a nicer way to click through it during a demo.

## What it does

- Renders an upload box, sends the image as base64 to `POST {SERVE_URL}/recommend`.
- Shows the returned caption and ranked recommendations with similarity scores.
- Pings `GET {SERVE_URL}/health` to show whether the backend is reachable.

## Prerequisites

The recommendation service must already be running and reachable — either:

- **In the workspace**, from the notebook or `serve run serve_app:app` (defaults to `http://localhost:8000`), or
- **As an Anyscale Service** (`anyscale service deploy -f setup/service.yaml`), in which case
  point the UI at the service's base URL and pass its bearer token.

## Run it

```bash
# Streamlit isn't in the template's requirements.txt (it's not needed on the
# cluster) — install it wherever you run the UI:
pip install streamlit requests pillow

# Point at the backend. Local default is http://localhost:8000.
export SERVE_URL="http://localhost:8000"

streamlit run streamlit_app.py
```

### Against a deployed Anyscale Service

A deployed service sits behind an authenticated HTTPS URL. The app reads `SERVE_URL`;
the service token must be sent as a bearer header. Set the base URL to the service URL
and add the token (the app passes through standard `requests`, so an
`Authorization: Bearer <token>` header is required — extend `streamlit_app.py` to read
a `SERVE_TOKEN` env var if you wire this up):

```bash
export SERVE_URL="https://<your-service>.anyscale-services.com"
export SERVE_TOKEN="<bearer-token-from-anyscale-service-status>"
streamlit run streamlit_app.py
```

## Configuration

| Env var      | Default                 | Purpose                                   |
|--------------|-------------------------|-------------------------------------------|
| `SERVE_URL`  | `http://localhost:8000` | Base URL of the Ray Serve recommender     |
| `SERVE_TOKEN`| _(unset)_               | Bearer token, required for Anyscale Services |

## Why it's split out

- Keeps the notebook focused on the Ray Data / Train / Serve story, not UI plumbing.
