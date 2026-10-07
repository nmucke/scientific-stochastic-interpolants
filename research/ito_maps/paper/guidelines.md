# ICML author requirements and template provenance

Checked 2026-10-07. The official available author kit verified here is **ICML 2026**. Its January 2026 deadlines have passed. Searches did not locate official 2027 author instructions, and the candidate 2027 instructions URL was unavailable. This is not proof that none exist elsewhere. Recheck the intended next submission cycle before final formatting or planning deadlines.

## Verified sources

- [2026 Author Instructions](https://icml.cc/Conferences/2026/AuthorInstructions): eight-page main body, unlimited references and appendices in one PDF; anonymized LaTeX submission; 50 MB submission limit. Final accepted papers may use nine main-body pages and a 20 MB PDF. Supplementary code/data must be anonymous and submitted with the paper. Anonymous repository branches must be frozen at submission. Do not include acknowledgments or identifying repository links in review copies. The camera-ready process includes a format checker, publication consent, and plain-language summary. Main-track impact statements are required; final financial-conflict disclosures apply where relevant.
- [2026 Call for Papers](https://icml.cc/Conferences/2026/CallForPapers): impact statement and appendices are outside the eight-page body limit. Claims need reproducible empirical or sound theoretical support and appropriate attribution. Generative AI assistance is allowed, with human authors responsible for all content; LLMs cannot be authors, and the conference encourages describing substantial methodological use. Dual/concurrent submissions are restricted. Reciprocal-review requirements apply. Accepted submissions and their original submitted versions/reviews become public. The verified old deadlines were abstract January 23 and full paper January 28, 2026, AoE; these are **not** prospective deadlines.
- [Official example PDF](https://media.icml.cc/Conferences/ICML2026/Styles/example_paper.pdf): preserve the supplied style, use a single-paragraph abstract preferably four to six sentences, table captions above and figure captions below, and complete references. Use letter paper. Keep essential claims in the main text since appendices need not be read.

## Official kit and discrepancy

Downloaded [official ICML 2026 ZIP](https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip) into `template/`; exact URL, retrieval date, and SHA-256 are in `template/provenance.json`. Style files are unmodified. `main.tex` uses anonymous review mode and no identifying author details.

The example PDF still mentions a 10 MB size limit and Type-1 fonts only, while the live Author Instructions specify 50 MB for submission, 20 MB camera-ready, and explicitly no Type-3 font check. Prefer current live conference instructions and keep this discrepancy visible until the next year's kit is available. Our current PDF is far below both limits.

## Draft build and submission gates

Run `make -C research/ito_maps/paper`; the Makefile sets TeX/BibTeX search paths to the vendor template directory. A successful build verifies syntax only. Before submission: reconcile every methodological statement with implemented behavior, replace the results-status paragraph with actual recorded measurements, include uncertainty and experiment settings, recheck bibliography and conference requirements, inspect figures and equations, and have the human authors review responsibility, authorship, and disclosure. Do not submit this initial proposal-style draft.
