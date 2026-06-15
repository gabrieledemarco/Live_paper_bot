# Deploying the Live Paper Trading System

## Prerequisites
- [Fly.io account](https://fly.io) (free tier covers both apps)
- [Supabase](https://supabase.com) or [Neon](https://neon.tech) free-tier Postgres database
- Docker installed locally (for testing)

## Step 1: Provision Postgres

### Option A: Supabase (recommended)
1. Create a free Supabase project
2. Go to Project Settings → Database → Connection string
3. Copy the URI (starts with `postgresql://`)
4. Save as `DATABASE_URL`

### Option B: Neon
1. Create a free Neon project
2. Copy the connection string
3. Save as `DATABASE_URL`

## Step 2: Freeze the Strategy Bundle

Run the freeze script locally (requires cached OHLCV data):

```bash
python -m live.freeze_strategy
```

This trains P(win) models on the full historical timeline and saves to
`artifacts/btc_bundle/`. Commit and push this directory.

## Step 3: Deploy to Fly.io

### Install flyctl
```bash
# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex

# macOS/Linux
curl -L https://fly.io/install.sh | sh
```

### Login
```bash
flyctl auth login
```

### Deploy the API
```bash
flyctl launch --from live/deploy/fly.api.toml --no-deploy
flyctl secrets set DATABASE_URL="<your-supabase-connection-string>"
flyctl deploy --config live/deploy/fly.api.toml
```

### Deploy the Trader
```bash
flyctl launch --from live/deploy/fly.trader.toml --no-deploy
flyctl secrets set DATABASE_URL="<your-supabase-connection-string>"
flyctl deploy --config live/deploy/fly.trader.toml
```

### Prevent sleeping (trader must be always-on)
```bash
flyctl machine update <machine-id> --auto-stop-machines=false
```

## Step 4: Access the Dashboard

Open the API app URL in your browser (e.g., `https://live-api.fly.dev`).
The static `web/` files are served from the API container.

## Step 5: Monitoring

- `/health` — DB liveness + last heartbeat timestamp
- `/kpis?run_id=<id>` — performance KPIs
- Dashboard auto-refreshes every 5 seconds

## Local Testing

```bash
docker-compose -f live/docker-compose.yml up --build
```

This boots Postgres + trader + API. Open http://localhost:8000 for the dashboard.

## Secrets & Security

- The **only secret** is `DATABASE_URL` (your Postgres connection string)
- No exchange API keys are needed (public Binance endpoints only)
- The API is read-only (CORS open for the dashboard)
