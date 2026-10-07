#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/figures/tikz"
pdflatex -interaction=nonstopmode -halt-on-error contours_main.tex
pdflatex -interaction=nonstopmode -halt-on-error contours_all.tex
cp contours_main.pdf ../fig_sketch_2d_main.pdf
cp contours_all.pdf ../fig_sketch_2d_fits.pdf
