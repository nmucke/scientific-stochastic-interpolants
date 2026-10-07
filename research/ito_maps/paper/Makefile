.PHONY: all clean
all:
	TEXINPUTS=.:template: BSTINPUTS=.:template: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
clean:
	latexmk -c main.tex
