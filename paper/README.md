# Local research report

Source: `procedure-acquisition.tex`, generated `results.tex`, and versioned bibliography `references.bib`.
Build with Tectonic 0.17.0 from the repository root:

```
tectonic --outdir output/pdf paper/procedure-acquisition.tex
pdftoppm -scale-to 1200 -png output/pdf/procedure-acquisition.pdf tmp/pdfs/review
```

This session reused the sibling study's existing Tectonic binary at
`../update-source-selection/tmp/tectonic`; no new TeX installation was needed.
The experiment is independent of that compiler location. Tectonic may fetch
TeX support files on first use. The final PDF is a local report, not externally submitted.
