# ====== Config ======
# Default: container/CPU mode uses .env
ENV_FILE ?= .env

# Project name used by docker-compose (for hard nuke)
PROJECT_NAME ?= llm-server

# helper to load .env vars in recipes
define dotenv
	set -a; \
	[ -f $(ENV_FILE) ] && . $(ENV_FILE); \
	set +a;
endef

# FastAPI app (host)
API_PORT      ?= 8000
# public entry
NGINX_PORT    ?= 8080
PG_HOST       ?= 127.0.0.1
PG_PORT       ?= 5433
PG_USER       ?= llm
PG_DB         ?= llm
REDIS_HOST    ?= 127.0.0.1
REDIS_PORT    ?= 6379

# You can pass API_KEY on the command line: make seed-key API_KEY=sk_live_...
API_KEY       ?=

# ====== Phony targets ======
.PHONY: \
  dev-local dev-cpu dev-tmux \
  up-local up-cpu down restart ps status \
  logs logs-nginx logs-postgres logs-redis \
  migrate revision seed-key \
  api-local curl test env \
  clean clean-volumes nuke

# ====== One-shot developer experience ======
dev-local: ENV_FILE=.env.local ## Local dev: MPS LLM on host + Docker infra
dev-local: up-local
	@echo "✅ Infra (postgres/redis/prom/grafana/nginx) is up for LOCAL mode."
	@echo "👉 In another terminal: ENV_FILE=.env.local make api-local"
	@echo "👉 If you need to run migrations: ENV_FILE=.env.local make migrate"
	@echo "👉 Then test via Nginx: make curl API_KEY=<your-key>"

dev-cpu: ENV_FILE=.env ## Dev: CPU LLM in container + infra
dev-cpu: up-cpu migrate-docker
	@echo "✅ Infra + API container are up and DB is migrated (CPU mode)."
	@echo "👉 Seed an API key: make seed-key API_KEY=$$(openssl rand -hex 24)"
	@echo "👉 Then test via Nginx: make curl API_KEY=<your-key>"

# Optional: auto-run API (local) + logs in tmux panes (local mode)
dev-tmux: ENV_FILE=.env.local
dev-tmux: up-local migrate
	@command -v tmux >/dev/null || (echo "tmux not found. Install it or use 'make dev-local'." && exit 1)
	tmux new-session -d -s llmdev "make api-local"
	tmux split-window -h "make logs-nginx"
	tmux select-pane -t 0
	tmux attach-session -t llmdev

# ====== Containers ======
# Local MPS mode: API runs on host, Docker runs infra + Nginx/Prometheus/Grafana
up-local: ## Start docker services for LOCAL mode (no API container)
	docker compose -f docker-compose.yml -f docker-compose.local.yml up -d \
	  postgres redis prometheus grafana pgadmin nginx
	@echo "⏳ Waiting for Postgres @ $(PG_HOST):$(PG_PORT) ..."
	@for i in $$(seq 1 30); do \
		pg_isready -h $(PG_HOST) -p $(PG_PORT) -d $(PG_DB) -U $(PG_USER) >/dev/null 2>&1 && break; \
		sleep 1; \
	done || (echo "❌ Postgres not ready on $(PG_HOST):$(PG_PORT)"; exit 1)
	@echo "✅ Docker services (LOCAL mode) are up."

# CPU mode: API + LLM run in the `api` container
up-cpu: ## Start docker services for CPU mode (API container + infra)
	docker compose up -d postgres redis api prometheus grafana pgadmin nginx
	@echo "⏳ Waiting for Postgres @ $(PG_HOST):$(PG_PORT) ..."
	@for i in $$(seq 1 30); do \
		pg_isready -h $(PG_HOST) -p $(PG_PORT) -d $(PG_DB) -U $(PG_USER) >/dev/null 2>&1 && break; \
		sleep 1; \
	done || (echo "❌ Postgres not ready on $(PG_HOST):$(PG_PORT)"; exit 1)
	@echo "✅ Docker services (CPU mode) are up."

# Keep the old 'up' as an alias to CPU mode if you like
up: up-cpu

down: ## Stop docker services (keep volumes)
	docker compose down

restart: down up ## Restart services (CPU mode by default)

ps: ## List compose services
	docker compose ps

status: ps ## Alias

logs: ## Tail all docker logs
	docker compose logs -f

logs-nginx:
	docker compose logs -f nginx

logs-postgres:
	docker compose logs -f postgres

logs-redis:
	docker compose logs -f redis

# ====== DB Migrations ======
# Uses ENV_FILE to load DATABASE_URL before running alembic
migrate: ## Alembic upgrade to head (uses ENV_FILE’s DATABASE_URL)
	@$(dotenv) \
	uv run python -m alembic upgrade head

revision: ## Autogenerate a new migration (edit message: make revision m="msg")
	@if [ -z "$(m)" ]; then echo 'Usage: make revision m="your message"'; exit 1; fi
	@$(dotenv) \
	uv run python -m alembic revision --autogenerate -m "$(m)"

migrate-docker: ## Run Alembic migrations inside the API container
	docker compose exec api python -m alembic upgrade head

# ====== Seed admin API key ======
seed-key: ## Insert 'admin' role + API key into Postgres (requires API_KEY=...)
	@if [ -z "$(API_KEY)" ]; then echo '❌ Provide API_KEY, e.g. make seed-key API_KEY=$$(openssl rand -hex 24)'; exit 1; fi
	docker exec -i llm_postgres psql -U $(PG_USER) -d $(PG_DB) -v ON_ERROR_STOP=1 \
	  -c "DO $$$$ BEGIN IF NOT EXISTS (SELECT 1 FROM roles WHERE name='admin') THEN INSERT INTO roles (name) VALUES ('admin'); END IF; END $$$$;"
	docker exec -i llm_postgres psql -U $(PG_USER) -d $(PG_DB) -v ON_ERROR_STOP=1 \
	  -c "INSERT INTO api_keys (key, label, active, role_id, quota_used, quota_monthly, quota_reset_at) SELECT '$(API_KEY)', 'bootstrap', TRUE, r.id, 0, NULL, NULL FROM roles r WHERE r.name = 'admin' ON CONFLICT (key) DO NOTHING;"
	@echo "✅ Seeded API key: $(API_KEY)"

# ====== Local runner (host: MPS mode) ======
api-local: ENV_FILE=.env.local ## Run the App API (FastAPI) on $(API_PORT) using local MPS config
api-local:
	@$(dotenv) \
	ENV=dev PORT=$(API_PORT) uv run serve

# ====== Quick checks ======
curl: ## Example request via Nginx (requires API_KEY to be seeded)
	@if [ -z "$(API_KEY)" ]; then echo 'Tip: make curl API_KEY=<your-key>'; fi
	@url="http://localhost:$(NGINX_PORT)/api/v1/generate"; \
	echo "➡️  Requesting $$url"; \
	curl -sS --fail "$$url" \
	  -H "Content-Type: application/json" \
	  -H "X-API-Key: $(API_KEY)" \
	  -d '{ "prompt": "Write a haiku about autumn leaves.", "max_new_tokens": 32, "temperature": 0.7, "top_p": 0.95 }' \
	  | jq .

test: ## Sanity: hit API directly; then via Nginx (needs API_KEY)
	@if [ -n "$(API_KEY)" ]; then \
	  echo "➡️  Direct API:"; \
	  curl -s http://127.0.0.1:$(API_PORT)/v1/generate \
	    -H "Content-Type: application/json" \
	    -H "X-API-Key: $(API_KEY)" \
	    -d '{ "prompt": "ping", "max_new_tokens": 4 }' | jq . ; \
	  echo "➡️  Through Nginx:"; \
	  curl -s http://localhost:$(NGINX_PORT)/api/v1/generate \
	    -H "Content-Type: application/json" \
	    -H "X-API-Key: $(API_KEY)" \
	    -d '{ "prompt": "ping", "max_new_tokens": 4 }' | jq . ; \
	else \
	  echo "⚠️  Set API_KEY to test: make test API_KEY=<key>"; \
	fi

env: ## Print key env vars (using ENV_FILE)
	@$(dotenv) \
	echo "ENV_FILE=$(ENV_FILE)"; \
	echo "ENV=$$ENV  DEBUG=$$DEBUG"; \
	echo "DATABASE_URL=$$DATABASE_URL"; \
	echo "REDIS_URL=$$REDIS_URL"; \
	echo "MODEL_ID=$$MODEL_ID MODEL_DEVICE=$$MODEL_DEVICE"; \
	echo "API_PORT=$(API_PORT)  NGINX_PORT=$(NGINX_PORT)"

# ====== Cleanup (careful!) ======
clean: ## Stop containers & remove orphans (keep volumes)
	docker compose down --remove-orphans

clean-volumes: ## Stop containers & REMOVE VOLUMES (DB/Redis data LOST)
	@read -p "This will DELETE volumes (DB/Redis). Are you sure? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	  docker compose down -v; \
	else \
	  echo "Aborted."; \
	fi

nuke: ## Hard nuke: remove containers, networks, volumes for this project (even if compose is broken)
	@echo "🔨 Hard nuke for Docker resources with project '$(PROJECT_NAME)'"
	@echo

	@echo "⛔ Stopping & removing containers..."
	@ids=$$(docker ps -aq --filter "label=com.docker.compose.project=$(PROJECT_NAME)"); \
	if [ -n "$$ids" ]; then \
		echo "   Containers: $$ids"; \
		docker stop $$ids >/dev/null 2>&1 || true; \
		docker rm -f $$ids >/dev/null 2>&1 || true; \
	else \
		echo "   No matching containers found."; \
	fi

	@echo
	@echo "🌐 Removing networks..."
	@net_ids=$$(docker network ls -q --filter "label=com.docker.compose.project=$(PROJECT_NAME)"); \
	if [ -n "$$net_ids" ]; then \
		echo "   Networks: $$net_ids"; \
		docker network rm $$net_ids >/dev/null 2>&1 || true; \
	else \
		echo "   No matching networks found."; \
	fi

	@echo
	@read -p "❗ Delete volumes for project '$(PROJECT_NAME)' as well? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		echo "💥 Removing volumes..."; \
		vol_ids=$$(docker volume ls -q --filter "label=com.docker.compose.project=$(PROJECT_NAME)"); \
		if [ -n "$$vol_ids" ]; then \
			echo "   Volumes: $$vol_ids"; \
			docker volume rm $$vol_ids >/dev/null 2>&1 || true; \
		else \
			echo "   No matching volumes found."; \
		fi; \
	else \
		echo "Skipping volume deletion."; \
	fi

	@echo
	@echo "✅ Hard nuke complete. You can now safely run 'docker compose up' again."