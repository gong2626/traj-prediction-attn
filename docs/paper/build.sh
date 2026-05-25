#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

target="${1:-en}"

case "$target" in
  en)
    latexmk -pdf -bibtex -interaction=nonstopmode main.tex
    ;;
  zh)
    latexmk -xelatex -bibtex -interaction=nonstopmode main_zh.tex
    ;;
  all)
    latexmk -pdf -bibtex -interaction=nonstopmode main.tex
    latexmk -xelatex -bibtex -interaction=nonstopmode main_zh.tex
    ;;
  *)
    echo "Usage: $0 [en|zh|all]" >&2
    exit 2
    ;;
esac
