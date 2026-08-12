VENV_DIR := .venv

UV := $(shell command -v uv 2> /dev/null)
ifndef UV
	UV = $(VENV_DIR)/bin/uv
endif

.PHONY: sync

all: sync

sync:
	uv sync --dev

test: sync
	uv run pytest

clean:
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +

plots: $(UV)
	$(UV) run python -m benchmarks.generate_publication_plots
