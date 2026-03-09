# Infrastructure Repository - Claude Guidelines

## Project Overview

Docker Compose based infrastructure managing databases for two applications:

- **家計簿アプリ** (`finance-db`): PostgreSQL 15, host port 5432, DB: `finance_db`
- **プロジェクト管理ツール** (`pm-db`): PostgreSQL 15, host port 5433, DB: `pm_db`
- **PgAdmin**: Web UI at http://localhost:8080

## Common Commands

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View logs (all services)
docker compose logs -f

# View logs (specific service)
docker compose logs -f finance-db
docker compose logs -f pm-db

# Connect to Finance DB
docker compose exec finance-db psql -U user -d finance_db

# Connect to PM DB
docker compose exec pm-db psql -U user -d pm_db

# Check service status
docker compose ps

# Restart a specific service
docker compose restart finance-db
```

## Architecture

- All services share a `app-network` bridge network
- Data persisted in named volumes: `finance_db_data`, `pm_db_data`
- PgAdmin: http://localhost:8080 — login: `admin@example.com` / `admin`
- DB credentials (dev): user=`user`, password=`password`

## Warnings

- `docker compose down -v` will **DELETE ALL DATABASE DATA** — use with caution
- Do NOT expose ports 5432/5433 to public networks in production
- Change PgAdmin default credentials (`admin@example.com` / `admin`) before production use
- DB credentials in `docker-compose.yml` are for development only — never use in production
