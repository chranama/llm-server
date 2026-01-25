# ====== Config ======
ENV_FILE ?= .env
PROJECT_NAME ?= llm-extraction-platform

define dotenv
	set -a; \
	[ -f $(ENV_FILE) ] && . $(ENV_FILE); \
	set +a;
endef

# Host ports (published ports only; DB/Redis are NOT published)
API_PORT ?= 8000
UI_PORT ?= 5173
PGADMIN_PORT ?= 5050
PROM_PORT ?= 9090
GRAFANA_PORT ?= 3000
PROM_HOST_PORT ?= 9091

# DB defaults (for exec-inside-postgres helpers)
PG_USER ?= llm
PG_PASSWORD ?= llm
PG_DB ?= llm

API_KEY ?=
EVAL_ARGS ?=

# ====== Paths ======
COMPOSE_DIR ?= deploy/compose
COMPOSE_YML ?= $(COMPOSE_DIR)/docker-compose.yml
BACKEND_DIR ?= backend
TOOLS_DIR ?= tools
COMPOSE_DOCTOR ?= $(TOOLS_DIR)/compose_doctor.sh

# ====== Compose command (single file + profiles) ======
COMPOSE := COMPOSE_PROJECT_NAME=$(PROJECT_NAME) docker compose --env-file $(ENV_FILE) -f $(COMPOSE_YML)

# Convenience macro: run compose with profiles
# Example: $(call dc,infra api,up -d)
define dc
	$(COMPOSE) $(foreach p,$(1),--profile $(p)) $(2)
endef

.PHONY: \
  init init-env env config \
  infra-up infra-down infra-ps infra-logs \
  api-up api-down api-gpu-up api-gpu-down api-ps api-logs \
  ui-up ui-down \
  admin-up admin-down \
  obs-up obs-down obs-host-up obs-host-down \
  eval-run eval-shell eval-host-run eval-host-shell \
  dev-cpu dev-gpu dev-local \
  migrate-docker revision-docker \
  seed-key seed-key-from-env \
  api-local \
  test-unit test-integration test-all \
  doctor \
  clean clean-volumes nuke

# ====== Setup ======
init: init-env

init-env:
	@if [ -f .env ]; then \
		echo "✔ .env already exists (using $(ENV_FILE))"; \
	elif [ -f .env.example ]; then \
		echo "📄 Creating .env from .env.example"; \
		cp .env.example .env; \
	else \
		echo "❌ .env.example not found; create .env manually."; \
		exit 1; \
	fi
	@echo "✅ Env file ready: .env"

env:
	@$(dotenv) \
	echo "ENV_FILE=$(ENV_FILE)"; \
	echo "PROJECT_NAME=$(PROJECT_NAME)"; \
	echo "API_PORT=$(API_PORT) UI_PORT=$(UI_PORT)"; \
	echo "PG_USER=$(PG_USER) PG_DB=$(PG_DB)"; \
	echo "PGADMIN_PORT=$(PGADMIN_PORT) PROM_PORT=$(PROM_PORT) GRAFANA_PORT=$(GRAFANA_PORT) PROM_HOST_PORT=$(PROM_HOST_PORT)"; \
	echo "MODELS_YAML=$$MODELS_YAML"

config:
	@$(COMPOSE) config >/dev/null && echo "✅ compose config OK"

# ====== Infra (postgres + redis) ======
infra-up:
	@$(call dc,infra,up -d)
	@echo "✅ infra up (postgres/redis)."

infra-down:
	@$(call dc,infra,down --remove-orphans)
	@echo "✅ infra down"

infra-ps:
	@$(call dc,infra,ps)

infra-logs:
	@$(call dc,infra,logs -f --tail=200)

# ====== API modes (dockerized) ======
api-up:
	@$(call dc,infra api,up -d --build)
	@echo "✅ api up (docker) @ http://localhost:$(API_PORT)"
	@echo "ℹ️  Capabilities are controlled by models.yaml."

api-down:
	@$(call dc,infra api,down --remove-orphans)
	@echo "✅ api down"

api-gpu-up:
	@$(call dc,infra api-gpu,up -d --build)
	@echo "✅ api_gpu up (docker, nvidia)"

api-gpu-down:
	@$(call dc,infra api-gpu,down --remove-orphans)
	@echo "✅ api_gpu down"

api-ps:
	@$(COMPOSE) ps

api-logs:
	@$(COMPOSE) logs -f --tail=200

# ====== UI / Admin / Obs ======
ui-up:
	@$(call dc,ui,up -d --build)
	@echo "✅ ui up @ http://localhost:$(UI_PORT)"

ui-down:
	@$(call dc,ui,down --remove-orphans)
	@echo "✅ ui down"

admin-up:
	@$(call dc,infra admin,up -d)
	@echo "✅ pgadmin up @ http://localhost:$(PGADMIN_PORT)"

admin-down:
	@$(call dc,infra admin,down --remove-orphans)
	@echo "✅ pgadmin down"

obs-up:
	@$(call dc,obs,up -d)
	@echo "✅ obs up @ prometheus http://localhost:$(PROM_PORT) | grafana http://localhost:$(GRAFANA_PORT)"

obs-down:
	@$(call dc,obs,down --remove-orphans)
	@echo "✅ obs down"

obs-host-up:
	@$(call dc,obs-host,up -d)
	@echo "✅ obs-host up @ prometheus http://localhost:$(PROM_HOST_PORT) | grafana http://localhost:$(GRAFANA_PORT)"

obs-host-down:
	@$(call dc,obs-host,down --remove-orphans)
	@echo "✅ obs-host down"

# ====== Golden paths ======
dev-cpu: api-up migrate-docker
	@echo "✅ dev-cpu ready"
	@echo "👉 seed key: make seed-key-from-env"
	@echo "👉 health:   curl -sS http://localhost:$(API_PORT)/healthz"
	@echo "👉 ready:    curl -sS http://localhost:$(API_PORT)/readyz"
	@echo "👉 model:    curl -sS http://localhost:$(API_PORT)/modelz"

dev-gpu: api-gpu-up migrate-docker
	@echo "✅ dev-gpu ready"
	@echo "👉 model:    curl -sS http://localhost:$(API_PORT)/modelz"

# dev-local: infra in docker, API on host (MPS), eval against host API
# NOTE: because postgres isn't published to the host, your host API must be configured
# to reach postgres via docker network OR you must publish postgres ports in compose.
dev-local: ENV_FILE=.env.local
dev-local: infra-up
	@echo "✅ dev-local infra ready"
	@echo "⚠️  Your compose does NOT publish Postgres to host."
	@echo "   If host API needs Postgres on localhost, publish ports or run host API inside docker."
	@echo "👉 run api on host:  ENV_FILE=.env.local make api-local"
	@echo "👉 eval host api:    ENV_FILE=.env.local make eval-host-run"

# ====== DB Migrations (docker-only) ======
migrate-docker:
	@# Run alembic inside whichever API service is running.
	@# Prefer api, else api_gpu.
	@$(COMPOSE) ps --services | grep -q '^api$$' && $(COMPOSE) exec -T api python -m alembic upgrade head || true
	@$(COMPOSE) ps --services | grep -q '^api_gpu$$' && $(COMPOSE) exec -T api_gpu python -m alembic upgrade head || true
	@echo "✅ migrations applied (docker)"

revision-docker:
	@if [ -z "$(m)" ]; then echo 'Usage: make revision-docker m="your message"'; exit 1; fi
	@$(COMPOSE) ps --services | grep -q '^api$$' && $(COMPOSE) exec -T api python -m alembic revision --autogenerate -m "$(m)" || true
	@$(COMPOSE) ps --services | grep -q '^api_gpu$$' && $(COMPOSE) exec -T api_gpu python -m alembic revision --autogenerate -m "$(m)" || true

# ====== Seed API key (exec inside postgres container) ======
seed-key:
	@if [ -z "$(API_KEY)" ]; then echo '❌ Provide API_KEY, e.g. make seed-key API_KEY=$$(openssl rand -hex 24)'; exit 1; fi
	@# Ensure infra is up (postgres)
	@$(call dc,infra,up -d postgres >/dev/null)
	@# Insert role + key inside postgres container
	@$(call dc,infra,exec -T postgres sh -lc "psql -U $(PG_USER) -d $(PG_DB) -v ON_ERROR_STOP=1 -c \"INSERT INTO roles (name) SELECT 'admin' WHERE NOT EXISTS (SELECT 1 FROM roles WHERE name = 'admin');\"")
	@$(call dc,infra,exec -T postgres sh -lc "psql -U $(PG_USER) -d $(PG_DB) -v ON_ERROR_STOP=1 -c \"INSERT INTO api_keys (key, name, label, active, role_id, quota_used, quota_monthly, quota_reset_at) \
	      SELECT '$(API_KEY)', 'bootstrap', 'bootstrap', TRUE, r.id, 0, NULL, NULL \
	      FROM roles r WHERE r.name = 'admin' \
	      ON CONFLICT (key) DO NOTHING;\"")
	@echo "✅ Seeded API key: $(API_KEY)"

seed-key-from-env:
	@$(dotenv) \
	if [ -z "$$API_KEY" ]; then \
		echo "❌ API_KEY not set in $(ENV_FILE)."; \
		exit 1; \
	fi; \
	API_KEY="$$API_KEY" $(MAKE) seed-key

# ====== Local API runner (host) ======
api-local:
	@$(dotenv) \
	cd $(BACKEND_DIR) && \
	APP_ROOT=.. PORT=$(API_PORT) uv run serve

# ====== Tests ======
test-unit:
	@$(dotenv) \
	cd $(BACKEND_DIR) && \
	PYTHONPATH=src uv run pytest -q -m unit

test-integration:
	@$(dotenv) \
	cd $(BACKEND_DIR) && \
	PYTHONPATH=src uv run pytest -q -m integration

test-all:
	@$(dotenv) \
	cd $(BACKEND_DIR) && \
	PYTHONPATH=src uv run pytest -q

# ====== Eval ======
# eval against dockerized api (service name `api`)
eval-run: api-up
	@$(call dc,eval,run --rm eval sh -lc "pip install -e /work/eval && eval $(EVAL_ARGS)")

eval-shell: api-up
	@$(call dc,eval,run --rm --entrypoint sh eval)

# eval against host api (dev-local / MPS)
eval-host-run:
	@$(call dc,eval-host,run --rm eval_host sh -lc "pip install -e /work/eval && eval $(EVAL_ARGS)")

eval-host-shell:
	@$(call dc,eval-host,run --rm --entrypoint sh eval_host)

# ====== Compose doctor ======
doctor:
	@command -v bash >/dev/null || (echo "❌ bash not found"; exit 1)
	@if [ ! -x "$(COMPOSE_DOCTOR)" ]; then \
		echo "❌ $(COMPOSE_DOCTOR) not found or not executable."; \
		echo "   Create tools/compose_doctor.sh and chmod +x it."; \
		exit 1; \
	fi
	@$(COMPOSE_DOCTOR)

# ====== Cleanup ======
clean:
	@$(COMPOSE) down --remove-orphans

clean-volumes:
	@read -p "This will DELETE volumes (DB/Redis/Grafana). Are you sure? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	  $(COMPOSE) down -v --remove-orphans; \
	else \
	  echo "Aborted."; \
	fi

nuke:
	@echo "🔨 Hard nuke for Docker resources with project '$(PROJECT_NAME)'"
	@ids=$$(docker ps -aq --filter "label=com.docker.compose.project=$(PROJECT_NAME)"); \
	if [ -n "$$ids" ]; then docker rm -f $$ids >/dev/null 2>&1 || true; fi
	@net_ids=$$(docker network ls -q --filter "label=com.docker.compose.project=$(PROJECT_NAME)"); \
	if [ -n "$$net_ids" ]; then docker network rm $$net_ids >/dev/null 2>&1 || true; fi
	@read -p "❗ Delete volumes for project '$(PROJECT_NAME)' as well? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		vol_ids=$$(docker volume ls -q --filter "label=com.docker.compose.project=$(PROJECT_NAME)"); \
		if [ -n "$$vol_ids" ]; then docker volume rm $$vol_ids >/dev/null 2>&1 || true; fi; \
	fi
	@echo "✅ Hard nuke complete."