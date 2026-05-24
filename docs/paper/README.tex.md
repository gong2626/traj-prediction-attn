# LaTeX Draft Layout

Files:

- `main.tex`: main manuscript entry
- `paper_refs_v1.bib`: bibliography
- `sections/*.tex`: section files
- `tables/*.tex`: auto-generated tables from evaluation CSVs
- `build.sh`: latexmk wrapper

Build command:

```bash
cd docs/paper
latexmk -pdf -bibtex -interaction=nonstopmode main.tex
```

Current limitation:

- The current environment does not have `latexmk` / `pdflatex` installed, so this repository currently contains a prepared LaTeX manuscript source, not a verified compiled PDF.

Recommended next additions:

1. add author/affiliation block
2. add one method figure under `figures/`
3. add one calibration figure and one candidate trade-off figure
4. switch class/template once target venue is fixed
