# Report

`report.tex` — the report source (LaTeX, `article` class). Fill in each
`\todo{...}` (they render in red) from the `results/` files named in the
section comments.

## Build the PDF

The report is built **locally** (never on the cluster — the cluster only runs
the benchmarks). Easiest engine is **tectonic** (self-contained, auto-downloads
packages):

```bash
brew install tectonic          # if not already installed
cd report && tectonic report.tex    # -> report/report.pdf
```

Alternative with a full **pdflatex** install (`brew install --cask basictex`,
then `sudo tlmgr install siunitx booktabs`): run `pdflatex report.tex` twice so
cross-references resolve.

Packages used in the preamble: `geometry, amsmath, booktabs, graphicx,
siunitx, listings, xcolor, hyperref`.

## archive/

Superseded artifacts from the earlier version, kept for reference:

- `DSA5208_Final_Project_Report_OLD.pdf` — the previous PDF (describes the old,
  now-removed implementation).
- `build_pdf_report.py` — the old hand-written ReportLab generator (replaced by
  the LaTeX flow).
- `REPORT_TEMPLATE.md` — an earlier Markdown draft.
- `generate_figures.py` — earlier figure script (current figures come from
  `analysis/plot_results.py`).
