# Deploy su Render (free tier)

Blueprint (render.yaml) richiede un piano a pagamento. Su free tier si deploya
manualmente creando ogni servizio dal Dashboard di Render.

## Prerequisiti

- Un account [Render](https://dashboard.render.com) (free tier incluso)
- Il repository su GitHub con il bundle già generato:
  `python -m live.freeze_strategy`

---

## Step 1: Creare il Database Postgres

1. Vai su https://dashboard.render.com
2. **New + → PostgreSQL**
3. Compila:
   - **Name:** `live-trader-db`
   - **Database:** `live_trader`
   - **User:** lascia default
   - **Plan:** Free ($0/mese)
4. **Create Database**
5. Aspetta che lo stato diventi **Available** (~2 minuti)
6. Copia la **Internal Database URL** (la stringa che inizia con `postgresql://`)
   — la userai dopo

---

## Step 2: Deployare il Web Service (API + Dashboard)

1. **New + → Web Service**
2. Connetti il tuo GitHub repo (`gabrieledemarco/Live_paper_bot`)
3. Compila:
   - **Name:** `live-api`
   - **Region:** `Frankfurt (EU)` (più vicino a Binance)
   - **Branch:** `main`
   - **Runtime:** `Docker`
   - **Dockerfile Path:** `live/Dockerfile.api`
   - **Plan:** Free ($0/mese)
4. **Environment Variables:**
   - `Key:` `DATABASE_URL` → `Value:` incolla la Internal Database URL dello Step 1
5. **Advanced → Health Check Path:** `/health`
6. **Create Web Service**

Render builda l'immagine Docker e avvia il servizio. Al termine mostra l'URL
(e.g. `https://live-api.onrender.com`). Aprilo nel browser — vedrai la dashboard.

---

## Step 3: Deployare il Worker (trader loop)

1. **New + → Background Worker**
2. Connetti lo stesso repo
3. Compila:
   - **Name:** `live-trader`
   - **Region:** `Frankfurt (EU)`
   - **Branch:** `main`
   - **Runtime:** `Docker`
   - **Dockerfile Path:** `live/Dockerfile.trader`
   - **Plan:** Free ($0/mese)
4. **Environment Variables:**
   - `Key:` `DATABASE_URL` → `Value:` stessa Internal Database URL dello Step 1
5. **Create Background Worker**

Il worker parte subito. Nei log vedrai:
```
LiveTrader initialized; run_id=... bundle_hash=1125d970807d
LiveTrader loop started
```

---

## Step 4: Verificare

- **API + Dashboard:** https://live-api.onrender.com
- **Salute DB:** https://live-api.onrender.com/health
- **Logs worker:** Render Dashboard → live-trader → Logs

La dashboard mostra KPI, equity curve, trades e signals, aggiornamento ogni 5s.

---

## Costi

| Servizio | Free tier | Limiti |
|---|---|---|
| Postgres `live-trader-db` | $0 | 1 GB storage, 256 MB RAM |
| Web Service `live-api` | $0 | 512 MB RAM, sleep dopo 15 min idle |
| Worker `live-trader` | $0 | 512 MB RAM, **sempre acceso** |

Il worker (trader) resta sempre acceso anche sul free tier — solo i web service
vanno in sleep.

---

## Opzionale: tenere sveglio il Web Service

La dashboard non è raggiungibile quando il web service è in sleep.
Per tenerlo sveglio 24/7 (costa ~$7/mese), cambia il piano in **Starter**
dal Dashboard: live-api → Settings → Instance Type → Starter ($7/mese).

In alternativa, usa un cron-job gratuito (es. cron-job.org) che pinga
`/health` ogni 10 minuti — Render non dorme se riceve traffico ogni < 15 min.

---

## Troubleshooting

| Problema | Causa | Fix |
|---|---|---|
| `connection refused` | DATABASE_URL sbagliata | Usa **Internal** DB URL (non External) |
| Trader non parte | Bundle mancante | Esegui `freeze_strategy` e pusha su git |
| Dashboard mostra "—" | Web service in sleep | Apri l'URL del servizio per risvegliarlo |
| Build fallisce | psycopg2 non trovato | Verifica che requirements.txt contenga `psycopg2-binary` |

---

## Local Testing

```bash
docker-compose -f live/docker-compose.yml up --build
```

Postgres + trader + API in locale su http://localhost:8000.
