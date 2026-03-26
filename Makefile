# ───────────────────────────────────────────────────────────────────────
# DistFL — Developer Makefile
# ───────────────────────────────────────────────────────────────────────

FRONTEND_DIR := fl_client/web/ui
STATIC_DIR   := fl_client/web/static

.PHONY: help frontend-build sync-static build install install-dev run dev clean test

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Frontend ──────────────────────────────────────────────────────────

frontend-build: ## Build React frontend into static assets
	cd $(FRONTEND_DIR) && npm ci && npm run build

sync-static: ## (No-op) Vite already builds directly into static/
	@echo "Vite outputs to $(STATIC_DIR) — nothing to sync."

# ── Python package ────────────────────────────────────────────────────

build: frontend-build ## Build wheel + sdist (includes frontend)
	python -m build

install: ## pip install the package
	pip install .

install-dev: ## pip install in editable mode with dev extras
	pip install -e ".[dev]"

# ── Run ───────────────────────────────────────────────────────────────

run: ## Launch the DistFL UI (production bundle)
	distfl ui

dev: ## Run Vite dev server + FastAPI bridge concurrently
	@echo "Starting FastAPI bridge on :5050 and Vite dev server on :5173 …"
	@trap 'kill 0' INT; \
		distfl ui --no-browser & \
		cd $(FRONTEND_DIR) && npm run dev & \
		wait

# ── Quality ───────────────────────────────────────────────────────────

test: ## Run pytest suite
	python -m pytest tests/ -v

# ── Cleanup ───────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info distfl_client.egg-info
	rm -rf $(STATIC_DIR)/assets $(STATIC_DIR)/index.html
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."
