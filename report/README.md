# Report Build

`main.tex` is the IEEE conference-format final report.

## Build

```bash
cd report
latexmk -pdf main.tex
# or: pdflatex main && pdflatex main
```

Figures regenerate from `results/` with:

```bash
python scripts/report_figures.py
```

## Authoring Policy

Per the ASEN 5264 AI policy, **prose and math must be authored by Sean
and Andrew.** `main.tex` should be read end to end before submission,
with every numerical claim checked against `results/` and
`report/figures/eval_summary.csv`.

Figures, tables, numerical results, and the bibliography skeleton are
produced by scripts and can be used as-is, after verification against
`results/` and `ENGINEERING_LOG.md`.

## Submission checklist (Phase 9 gate)

- [x] Draft markers removed
- [ ] Final human read-through by both authors
- [x] Every numeric claim cross-checked with `results/`
- [ ] Page count 4–8 (not counting references)
- [ ] ≥5 citations with relation-to-project blurbs in Related Work
- [ ] Contributions paragraph with real split
- [x] Release statement (exact wording, pick one)
- [x] From-scratch vs off-the-shelf disclosure
- [x] AI-assistance disclosure
- [ ] PDF builds with no warnings beyond IEEE boilerplate
- [ ] Read-through by both authors after freeze
