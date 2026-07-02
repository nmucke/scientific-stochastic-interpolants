# Add TeX Live binaries to the path, common for MacTeX installations
export PATH := /usr/local/texlive/2026/bin/universal-darwin:$(PATH)

.PHONY: build watch clean

# Main target file
TARGET = main

# ---
# Main commands
# ---

# Build the PDF
build:
	latexmk -pdf -interaction=nonstopmode $(TARGET).tex

# Continuously watch for changes and recompile
watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(TARGET).tex

# Clean up temporary files
clean:
	latexmk -c $(TARGET).tex
