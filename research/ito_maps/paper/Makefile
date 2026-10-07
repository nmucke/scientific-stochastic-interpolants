# Rebuild ignored PDF/PNG figures from tracked data before compiling LaTeX.
PYTHON ?= $(if $(wildcard ../../../.venv/bin/python),../../../.venv/bin/python,python3)
VALIDATION_SOURCES := $(wildcard ../results/validation_baselines/*/evaluation.json ../results/validation_psd/*/evaluation.json ../results/validation_rollout/*/evaluation.json)

TEST_SEED0_SOURCES := $(wildcard ../results/test_one_step/*/evaluation.json)

.PHONY: all artifacts clean
all: artifacts
	TEXINPUTS=.:template: BSTINPUTS=.:template: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

artifacts: generated/validation_table.tex figures/ns_compression.pdf generated/test_seed0_table.tex generated/test_seed0_calibration.tex

generated/validation_table.tex: build_validation_table.py $(VALIDATION_SOURCES)
	$(PYTHON) build_validation_table.py

figures/ns_compression.pdf: build_compression_figure.py ../results/ns_path_compression/results.json
	$(PYTHON) build_compression_figure.py

clean:
	latexmk -c main.tex


generated/test_seed0_table.tex generated/test_seed0_calibration.tex &: build_seed0_results.py $(TEST_SEED0_SOURCES)
	$(PYTHON) build_seed0_results.py
