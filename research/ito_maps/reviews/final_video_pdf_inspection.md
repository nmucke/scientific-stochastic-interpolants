# Final video PDF inspection

Passed on 2026-10-07. The reviewed `paper/main.pdf` has 28 pages: seven main
pages including the video figure, references begin on page 8, and appendices
begin on page 9. This satisfies the eight-page main-text limit excluding
references. The video case appears on page 5, the complete FVD table on page 6,
and both datasets’ matched truth/prediction frame sequences on page 7.

Main and appendix labels are readable with no overlaps or overfull boxes. All
four model rows occur in both main panels; full appendix panels retain
persistence and the previously fixed cases/member. The detailed video setup
and metrics are in Appendix K (pages 22–28). Observed context and future frames
are distinguished, retained-frame indices are explicit, and completion SD is
not presented as a confidence interval.

Independently recomputed video ratios agree with abstract/main prose: Itô8
versus SI64 has FVD increases of 14.826% / 42.777%, speedups 6.0748× / 5.8966×,
and CRPS increases 0.1246% / 17.086% for KTH / CLEVRER. The deterministic-map
and persistence comparisons also agree with the audited reports. No quality
equivalence or general stochastic-map superiority is claimed. Existing NS
abstract numbers match the generated replication claims. Shared training
wall times remain separate from GPU-exclusive video inference timings.

One minor wording issue was reported to the editor and corrected before final
build: oracle metrics are in the accompanying reports, not the manuscript’s
image-metric tables. No paper edits or GPU work were performed in this review.

PDF SHA-256: `1bb6478e96beba803a534a237c4137a9d5a9a6bd5deacd2cf371611be1d09f40`. Full page locations, source-audit hash and
recomputed ratios are recorded in `final_video_pdf_inspection.json`.
