# Build the manuscript. The paper currently has no results-dependent artifacts:
# all Navier--Stokes results are \pending{} placeholders until the new runs land.
#
# To add generated tables/figures later, list them in ARTIFACTS and give each a
# rule of the form
#   generated/<name>.tex: build_<name>.py <sources>
#   	$(PYTHON) build_<name>.py
# and \input/\includegraphics them from the .tex sources.
#
# Archived (initial-stage results and the dropped second application): the
# former build_*.py scripts, generated/ and figures/ outputs and the matching
# .tex sources live in ../archive/paper/ (pre_placeholder_tex/ holds the
# pre-placeholder versions of the current sources).
# They read ../results, which has moved to ../archive/results, and are not built.

PYTHON ?= $(if $(wildcard ../../../.venv/bin/python),../../../.venv/bin/python,python3)
LATEX_ENV := TEXINPUTS=.:template: BSTINPUTS=.:template:

ARTIFACTS :=

.PHONY: all artifacts clean overleaf
all: artifacts
	$(LATEX_ENV) latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

artifacts: $(ARTIFACTS)

clean:
	$(LATEX_ENV) latexmk -c main.tex
	rm -rf overleaf overleaf.zip

# Self-contained Overleaf upload: sources plus the used ICML style files flattened
# into the project root (Overleaf does not search template/). Test-compiled
# without TEXINPUTS so a missing file fails here rather than on Overleaf.
OVERLEAF_STYLES := template/icml2026.sty template/icml2026.bst template/algorithm.sty \
	template/algorithmic.sty template/fancyhdr.sty
overleaf: artifacts
	rm -rf overleaf overleaf.zip && mkdir overleaf
	cp -r main.tex macros.tex references.bib sections appendix $(OVERLEAF_STYLES) overleaf/
	cd overleaf && latexmk -pdf -interaction=nonstopmode -halt-on-error -quiet main.tex >/dev/null \
		&& latexmk -C main.tex >/dev/null && rm -f main.bbl
	cd overleaf && zip -qr ../overleaf.zip .
	rm -rf overleaf
	@echo "wrote overleaf.zip"
