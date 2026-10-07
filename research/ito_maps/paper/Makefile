# Rebuild ignored PDF/PNG figures from tracked data before compiling LaTeX.
PYTHON ?= $(if $(wildcard ../../../.venv/bin/python),../../../.venv/bin/python,python3)
VALIDATION_SOURCES := $(wildcard ../results/validation_baselines/*/evaluation.json ../results/validation_psd/*/evaluation.json ../results/validation_rollout/*/evaluation.json)

TEST_SEED0_SOURCES := $(wildcard ../results/test_one_step/*/evaluation.json)
STUDY_SOURCES := ../results/final_report/one_step/summary.csv ../results/final_report/rollout/all_leads.csv ../results/final_report/one_step/paired_crps.json ../results/final_report/rollout/paired_crps.json $(wildcard ../results/training/*_seed0/history.json)
REPLICATION_SOURCES := $(wildcard ../results/final_report/*/student_training_seeds/*.json ../results/final_report/*/student_training_seeds/*.csv ../results/training/*/history.json) ../results/final_report/one_step/summary.csv ../results/final_report/rollout/all_leads.csv
REPLICATION_OUTPUTS := generated/replication_table.tex generated/replication_individuals.tex generated/replication_claims.tex generated/replication_training_cost.tex generated/replication_paired.tex
LONG_MAP_SOURCES := $(wildcard ../results/long_horizon/flow_map/*.json ../results/long_horizon/follmer_ito_map/*.json) ../results/long_horizon/independent_completed_map_audit.json ../results/long_horizon/reference_enstrophy.json ../results/long_horizon/spatial_enstrophy_spectra_maps_preview.json
LONG_MAP_OUTPUTS := figures/long_horizon_maps.pdf figures/long_horizon_maps_skill.pdf figures/first_step_enstrophy_spectrum.pdf figures/last_step_enstrophy_spectrum.pdf generated/long_horizon_maps_table.tex generated/long_horizon_maps_provenance.json
VIDEO_CASE_SOURCES := ../../video_forecasting/results/kth_protocol.json ../../video_forecasting/results/clevrer_protocol.json $(wildcard ../../video_forecasting/results/training/*/metadata.json ../../video_forecasting/notes/*_data_preview.png)
VIDEO_CASE_OUTPUTS := figures/video_case/setup_table.tex figures/video_case/training_table.tex figures/video_case/dataset_previews.pdf figures/video_case/asset_provenance.json

.PHONY: all artifacts clean replication-artifacts
all: artifacts
	TEXINPUTS=.:template: BSTINPUTS=.:template: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

artifacts: generated/validation_table.tex figures/ns_compression.pdf generated/test_seed0_table.tex generated/test_seed0_calibration.tex figures/seed0_quality_cost.pdf figures/seed0_rollout.pdf generated/seed0_claims.tex generated/seed0_training_cost.tex $(REPLICATION_OUTPUTS) $(LONG_MAP_OUTPUTS) $(VIDEO_CASE_OUTPUTS)

generated/validation_table.tex: build_validation_table.py $(VALIDATION_SOURCES)
	$(PYTHON) build_validation_table.py

figures/ns_compression.pdf: build_compression_figure.py ../results/ns_path_compression/results.json
	$(PYTHON) build_compression_figure.py

clean:
	latexmk -c main.tex


generated/test_seed0_table.tex generated/test_seed0_calibration.tex &: build_seed0_results.py $(TEST_SEED0_SOURCES)
	$(PYTHON) build_seed0_results.py

figures/seed0_quality_cost.pdf figures/seed0_rollout.pdf generated/seed0_claims.tex generated/seed0_training_cost.tex &: build_study_figures.py $(STUDY_SOURCES)
	$(PYTHON) build_study_figures.py

# Every frozen seed and training archive must be complete before rebuilding.
replication-artifacts: $(REPLICATION_OUTPUTS)

$(REPLICATION_OUTPUTS) &: build_replication_tables.py $(REPLICATION_SOURCES)
	$(PYTHON) build_replication_tables.py

$(LONG_MAP_OUTPUTS) &: build_long_horizon_figures.py ../experiments/report_long_horizon.py $(LONG_MAP_SOURCES)
	$(PYTHON) build_long_horizon_figures.py

$(VIDEO_CASE_OUTPUTS) &: build_video_case.py $(VIDEO_CASE_SOURCES)
	$(PYTHON) build_video_case.py
