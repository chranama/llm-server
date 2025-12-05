# LLM Server

A Production-Style LLM API Gateway & Inference Runtime  
---

**LLM Server** is a self-hosted, production-inspired platform for serving Large Language Models behind a secure, observable, scalable API layer. It mirrors the architecture used by modern AI infrastructure teams, packaged into a portfolio-grade systems engineering project.

The system demonstrates not only **model inference**, but the broader operational concerns required for real-world LLM deployments:

- API Gateway with Authentication & Roles
- Quotas & Rate Limits
- Structured Logging
- Observability (Prometheus + Grafana)
- Caching (SQLite + optional Redis)
- Multi-model orchestration infrastructure
- Flexible runtime (Local CPU/MPS or Docker CPU mode)

---

# 📚 Documentation

LLM Server’s full documentation is organized into topic-specific markdown files under `docs/`.

You can explore each section directly:

### **1. Introduction**  
[`docs/00-intro.md`](docs/00-intro.md)

### **2. Architecture Overview**  
[`docs/01-architecture.md`](docs/01-architecture.md)

### **3. Features Overview**  
[`docs/02-features.md`](docs/02-features.md)

### **4. Observability (Prometheus + Grafana)**  
[`docs/03-observability.md`](docs/03-observability.md)

### **5. Caching Layer**  
[`docs/04-caching.md`](docs/04-caching.md)

### **6. Multi-Model Infrastructure**  
[`docs/05-multimodel.md`](docs/05-multimodel.md)

### **7. API Stability and Versioning
[`docs/06-api-versioning.md`](docs/06-api-versioning.md)

### **8. Project Structure**  
[`docs/07-project-structure.md`](docs/07-project-structure.md)

### **9. Makefile Guide**  
[`docs/08-makefile.md`](docs/08-makefile.md)

### **10. Quickstart (Container Mode)**  
[`docs/09-quickstart-container.md`](docs/09-quickstart-container.md)

### **11. Quickstart (Local Mode)**  
[`docs/10-quickstart-local.md`](docs/10-quickstart-local.md)

### **12. Admin & Operations Guide**  
[`docs/11-admin-ops.md`](docs/11-admin-ops.md)

### **13. Testing Guide**  
[`docs/12-testing.md`](docs/12-testing.md)


---

# 🚀 Quick Start (Container Mode)

The easiest way to start the full stack (API + Postgres + Redis + Prometheus + Grafana + Nginx) is:

```bash
cp .env.example .env
make init
make up
make seed-key API_KEY=$(openssl rand -hex 24)
```

Then test:

```bash
make curl API_KEY=<your-key>
```

### Public endpoints via Nginx (default port `8080`)

- `http://localhost:8080/api`
- `http://localhost:8080/healthz`
- `http://localhost:8080/readyz`

### Admin dashboards

- Grafana → `http://localhost:8080/grafana`
- Prometheus → `http://localhost:8080/prometheus`
- pgAdmin → `http://localhost:8080/pgadmin`

*(All internal ports are intentionally hidden behind Nginx.)*

---

# 💻 Quick Start (Local Mode)

This runs the LLM on your host machine (CPU or MPS), while Docker runs the supporting services:

```bash
cp .env.example .env.local
make dev-local
```

In a separate terminal:

```bash
ENV_FILE=.env.local make api-local
```

---

# 🔑 API Stability & Versioning

LLM Server uses a **versioned API** under the `/v1/` namespace.

### **Stable Endpoints**
These can be relied on and will not change without a major version bump:

| Endpoint | Purpose |
|---------|---------|
| **POST /v1/generate** | Run inference on the configured model |
| **GET /v1/healthz** | Liveness probe |
| **GET /v1/readyz** | Readiness probe |
| **GET /v1/me/usage** | Usage & quota for the calling API key |
| **GET /v1/models** | List available models (static or multimodel) |

### **Experimental Endpoints**
These may change as capabilities evolve, but are still documented:

| Endpoint | Status |
|---------|--------|
| `/v1/admin/*` | Experimental admin routes (role-based) |
| Multimodel selection (`model:` in body) | Experimental until models.yaml is enabled |

---

# 🛠 Development & Testing

Run the full test suite:

```bash
make test
```

View specific test files under `tests/` to explore how the system validates:

- Authentication
- Rate limiting
- Quotas
- Generate API behavior
- Health & metrics endpoints
- Integration behavior

---

# 📝 Notes

- Local mode defaults to **Llama 3.1 8B**.  
- Docker mode defaults to **Llama 3.2 1B** for compatibility and low resource use.  
- Multi-model architecture is implemented even if `models.yaml` is disabled by default.  
- All observability services (Prometheus, Grafana, pgAdmin) run behind a single Nginx entrypoint for security.  
- This project is designed as a **realistic systems engineering portfolio piece**, not just an inference demo.

---

# 🤝 Contributing

Contributions, issues, and improvements are welcome!  
This repository is intentionally designed to be extended with additional model providers, caching backends, authentication strategies, and evaluation harnesses.

---

# 📄 License

MIT License — see `LICENSE` for details.