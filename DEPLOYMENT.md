# Deployment guide

Where and how to run this bot 24/7, starting from free.

---

## What it actually needs

Measured on a fully bootstrapped process (3 providers configured, database open, cogs loaded,
a 400-line knowledge base cached):

| Resource | Measured | Notes |
|---|---|---|
| RAM | **65 MB** | Grows slowly with cached configs + active ticket queues |
| Disk | **53 KB** | SQLite; grows with ticket history, not with messages |
| Open files | 13 | No sockets held beyond Discord + AI APIs |
| CPU | ~0% idle | Spikes only during an inference call, which is network-bound |
| Python | 3.10+ | Developed on 3.11 |

So **any 512 MB host is comfortable**, and 1 GB gives you room for several bots. The AI calls are
network-bound — the LLM decides your latency (300 ms on Cerebras, 1–2 s on Groq 70B), not your CPU.

### The one hard requirement

A Discord bot holds a **persistent WebSocket to Discord's gateway**. That single fact rules out
most "free hosting" advice:

- **No inbound ports needed.** The bot only makes outbound HTTPS/WSS connections, so there is
  nothing to expose and no firewall rules to open.
- **The process must never sleep.** Anything that idles your container disconnects the bot.
- **The filesystem should persist**, or you must use PostgreSQL instead of SQLite.

Outbound access to these hosts on port 443 is required:

```
discord.com  gateway.discord.gg  *.discord.gg  cdn.discordapp.com
api.groq.com  api.cerebras.ai  generativelanguage.googleapis.com
```

---

## Options, ranked

| # | Host | Cost | Sleeps? | Card? | Verdict |
|---|---|---|---|---|---|
| 1 | **Oracle Cloud Always Free** | $0 forever | No | Yes | **Best free.** Real VM, 24/7, 200 GB disk |
| 2 | **Google Cloud e2-micro** | $0 forever | No | Yes | Solid alternative, 1 GB RAM / 30 GB disk |
| 3 | **Your own hardware** (Pi, old laptop, home server) | $0 + electricity | No | No | **Best if cloud signup fails.** Full control, no approvals |
| 4 | **Fly.io** | ~$0 for one small VM | No | Yes | Good, but Docker-based and the allowance is fiddly |
| 5 | Hetzner / Vultr / DigitalOcean | €3.29–4/mo | No | Yes | Cheapest *hassle-free* option; no free-tier anxiety |
| 6 | Free Pterodactyl hosts (Wispbyte, HeavenCloud, etc.) | $0 | No | No | Works for a hobby bot, but expect manual renewal every 4–14 days and small hosts that come and go |
| ✗ | **Render, Replit, Railway, Koyeb free tiers** | $0 | **Yes / credits** | varies | **Do not use** — they idle your process or run out of trial credit |
| ✗ | Vercel, Netlify, Cloudflare Workers, Lambda | $0 | n/a | varies | **Cannot work.** Serverless can't hold a gateway connection |

> Note on the many "Best free Discord bot hosting 2026" articles: most of the top results are
> written by the hosts themselves. Treat their comparison tables as marketing. The Oracle/Google
> free tiers are the only ones with a long public track record and no renewal treadmill.

---

## Option 1 — Oracle Cloud Always Free (recommended)

**Important 2026 change:** Oracle cut the Always Free **Ampere A1 (ARM)** allowance from
4 OCPU / 24 GB to **2 OCPU / 12 GB**, enforced from **18 August 2026** — instances above the new
limit are **automatically terminated**. The **AMD micro** instances (2× `VM.Standard.E2.1.Micro`,
1/8 OCPU + 1 GB RAM each) and the 200 GB of block storage were **not** changed.

For this bot that is irrelevant in practice — 65 MB RAM means the *AMD micro* shape has ~15× the
memory you need, and it is usually **easier to provision** than ARM (ARM capacity in popular
regions frequently returns "Out of host capacity"). Pay-As-You-Go tenancies still get 3,000
OCPU-hours + 18,000 GB-hours of A1 free per month.

### Steps

1. Sign up at <https://www.oracle.com/cloud/free/>. Credit/debit card required for identity
   verification — **you are not charged** on Always Free shapes. Approval can be rejected on the
   first attempt; retrying a day later often works.
2. **Compute → Instances → Create instance**
   - Image: **Ubuntu 24.04** (or Debian 12)
   - Shape: **Specialty and previous generation → `VM.Standard.E2.1.Micro`** (Always Free eligible)
   - Networking: accept the default VCN; it assigns a public IP
   - Add your **SSH public key** (or let Oracle generate a key pair — download it)
3. Open the SSH port (Oracle blocks inbound by default, in *two* places):
   - OCI console → **Networking → Virtual cloud networks → your VCN → Security Lists →
     Add Ingress Rule**: source `0.0.0.0/0`, TCP port `22`
   - Then **inside the VM**, Ubuntu's iptables also blocks it:
     ```bash
     sudo iptables -I INPUT -p tcp --dport 22 -j ACCEPT
     sudo netfilter-persistent save
     ```
4. Deploy (see [Install on a VM](#install-on-a-vm) below).

**Caveats to know up front:** Oracle reclaims *idle* Always Free compute in some regions, so keep
the instance doing something (a running bot counts). Regional capacity for ARM is genuinely hard
to get. If either bites you, use Option 3.

---

## Option 2 — Google Cloud e2-micro

<https://cloud.google.com/free> — one `e2-micro` VM (2 shared vCPU, 1 GB RAM, 30 GB standard
disk) is always free in `us-west1`, `us-central1` or `us-east1`. Card required, $0 as long as you
stay in the free region/shape and don't add extras. Same deployment steps as Oracle. Watch out:
the $300 new-account credit expires after 90 days, but the e2-micro free allowance does not.

---

## Option 3 — Your own hardware

Genuinely the most reliable free option, and it sidesteps every cloud approval problem. A
Raspberry Pi 3/4/5, an old laptop, or any always-on PC works. The bot uses 65 MB, so even a Pi
with 1 GB is fine.

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git
git clone https://github.com/WaquarShaikh5555/Discord-bots.git /opt/ticket-bot
```

Then follow [Install on a VM](#install-on-a-vm). On a Pi, prefer the `arm64` build — everything
here is pure Python plus `aiosqlite`, so there is nothing to compile.

Only real downsides: your home internet going down takes the bot offline, and a laptop is not
free if it runs 24/7 (a Pi costs roughly ₹50–80/month in electricity).

---

## Install on a VM

Works identically on Oracle, Google Cloud, Hetzner or a Raspberry Pi.

```bash
# 1. Fetch the code
sudo mkdir -p /opt/ticket-bot && sudo chown $USER /opt/ticket-bot
git clone https://github.com/WaquarShaikh5555/Discord-bots.git /opt/ticket-bot
cd /opt/ticket-bot
git checkout arena/01a0c499-discord-bots      # or main, once merged

# 2. Virtualenv + dependencies
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
# SQLite only — add this if you use PostgreSQL instead:
# .venv/bin/pip install "asyncpg>=0.29"

# 3. Secrets. Create the file from the template, then edit it.
cp .env.example .env
nano .env                # paste DISCORD_BOT_TOKEN and GROQ_API_KEY
chmod 600 .env           # readable only by the owner — this file is your credentials

# 4. Validate BEFORE installing the service
.venv/bin/python -m scripts.doctor --live
#    --live sends one real request per provider, so you learn immediately
#    whether each key is valid. Expects a green "Ready." line.

# 5. Create the data directory for SQLite
mkdir -p data

# 6. Run it once in the foreground to watch the first boot
.venv/bin/python -m bot
#    Ctrl-C to stop.
```

### Run it as a service (survives reboots and crashes)

```bash
sudo useradd --system --home /opt/ticket-bot --shell /usr/sbin/nologin ticketbot
sudo chown -R ticketbot:ticketbot /opt/ticket-bot

sudo cp deploy/ticket-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ticket-bot

# Watch it
journalctl -u ticket-bot -f
systemctl status ticket-bot
```

`deploy/ticket-bot.service` already sets `Restart=always`, caps memory at 512 MB, and hardens the
unit (`ProtectSystem=strict`, `NoNewPrivileges`, write access limited to `data/`).

Useful commands:

```bash
sudo systemctl restart ticket-bot          # after a code or .env change
sudo systemctl stop ticket-bot             # stop the bot
journalctl -u ticket-bot --since "1 hour ago" | grep -i escalat
```

### Docker instead

The repo ships a `Dockerfile` and `docker-compose.yml`:

```bash
cp .env.example .env && nano .env
docker compose up -d --build      # bot + PostgreSQL
docker compose logs -f bot
```

Note `docker-compose.yml` overrides `DATABASE_URL` to point at its bundled PostgreSQL. For a
SQLite-only container, remove the `db` service and mount a volume at `/app/data`:

```bash
docker build -t ticket-bot .
docker run -d --name ticket-bot --restart unless-stopped \
  --env-file .env -v /opt/ticket-bot/data:/app/data ticket-bot
```

---

## Database: SQLite or PostgreSQL?

| Situation | Use |
|---|---|
| One bot instance on a VM with a persistent disk | **SQLite** — zero setup, zero cost, plenty fast at this scale |
| Host has an ephemeral filesystem (Render, Fly without a volume) | **PostgreSQL**, or your data vanishes on every deploy |
| Multiple bot instances / sharding | **PostgreSQL** (SQLite is per-process; in-process caches would diverge) |
| You want a managed DB with a UI and backups | **Supabase** free tier → PostgreSQL |

Switching is one line:

```bash
.venv/bin/pip install "asyncpg>=0.29"
# .env
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/tickets
```

Tables are created automatically on boot either way. **Supabase caveat:** free projects pause after
about a week of inactivity, which would silently break the bot — a constantly-writing bot normally
stays active, but check before relying on it. If you use Supabase, prefer the **pooler** connection
string (port 5432 for session mode) and note that `asyncpg` needs session mode, not transaction mode.

---

## How much traffic does the free tier cover?

Groq's free tier is ~**14,400 requests/day**, and this bot spends **one request per member
question** — not per message, because debouncing merges a burst of messages into one call.

| Scenario | Requests/day | Fits in Groq free? |
|---|---|---|
| Small server, 50 tickets/day | ~50 | Easily |
| Mid-size, 500 tickets/day | ~500 | Yes, 3% of quota |
| Busy, 2,000 tickets/day | ~2,000 | Yes, 14% of quota |
| Very busy, 14,000+/day | ~14,000 | At the limit — add Cerebras/Gemini to the chain |

The bot defends the quota for you: per-provider RPM token buckets, a persisted daily budget that
skips a provider once it is spent, and automatic failover. `/ai-status` shows live consumption.
If you outgrow one provider, just add a second API key — the chain is already `groq,cerebras,gemini`.

---

## Backups

Everything the bot knows lives in one file:

```bash
# SQLite: safe to copy while running thanks to WAL mode
sqlite3 data/tickets.db ".backup '/root/backups/tickets-$(date +%F).db'"

# or simply
cp data/tickets.db /root/backups/tickets-$(date +%F).db
```

A weekly cron is enough — the file is tens of kilobytes:

```bash
echo '0 4 * * 0 root sqlite3 /opt/ticket-bot/data/tickets.db ".backup /root/backups/tickets-$(date +\%F).db"' \
  | sudo tee /etc/cron.d/ticket-bot-backup
```

Knowledge bases are the only thing that is genuinely hard to recreate, so `/view-kb full` (which
downloads the KB as a file) is a handy manual export.

---

## Updating

```bash
cd /opt/ticket-bot
git pull
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.doctor
sudo systemctl restart ticket-bot
```

Schema changes are applied automatically on boot (`create_all` is idempotent). If you later add or
rename columns, introduce Alembic — the current tables match `schema.sql` exactly.

---

## Pre-flight checklist

- [ ] **Message Content Intent** enabled in the Discord developer portal — the #1 cause of "bot is
      online but never answers"
- [ ] Bot invited with scopes `bot` **and** `applications.commands`
- [ ] Bot's role sits **above** the staff role in Server Settings → Roles (or Discord blocks the ping)
- [ ] `.env` has `chmod 600` and is not committed (`git status` must not list it)
- [ ] `python -m scripts.doctor --live` reports green for your primary provider
- [ ] `/setup-kb`, `/set-staff-role` and `/set-ticket-category` all run in the target server
- [ ] A test question in a ticket channel gets an answer, and an unanswerable one escalates
- [ ] `systemctl enable ticket-bot` so it comes back after a reboot
