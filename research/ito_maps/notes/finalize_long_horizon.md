# Final four-model NS audit and report command

Run from the repository root, only after all four production trajectories have
completed:

```bash
.venv/bin/python research/ito_maps/experiments/finalize_long_horizon.py
```

A metadata-only readiness check creates no output:

```bash
.venv/bin/python research/ito_maps/experiments/finalize_long_horizon.py --check-only
```

The command requires `flow_map`, `follmer_ito_map`,
`follmer_stochastic_interpolant`, and `flow_matching` under
`research/ito_maps/results/long_horizon`. It enforces their respective budgets
1, 8, 64, and 64; 4,096 complete finite leads; eight members; trajectory 180;
origin 5; history 5; no feedback clipping; matching initial history, archive truth,
normalization, random seed, and stored-time spacing. It requires final restart
state and first/final saved raw snapshots. Incomplete, shortened, mismatched,
or nonfinite runs fail clearly before any derived outputs are written.
There is no implicit wait or automatic replacement with shorter trajectories.

After preflight it invokes the existing implementations, without redefining
metrics:

1. `audit_long_horizon_units.py` independently checks all raw hashes, physical
   scalar metrics, supported truth, inputs, and restart state. Output:
   `independent_all_model_audit.json`.
2. `spatial_enstrophy_spectra.py` computes leads 1 and 4096, including per-member
   raw-vorticity spectra, Parseval checks, paired first-lead truth, and held-out
   archive context. Output: `spatial_enstrophy_spectra_all_models.json`.
3. `report_long_horizon.py` consumes the audited spectrum JSON and existing
   `reference_enstrophy.json`. Output: `final_report/`, including separate first/last
   spatial-spectrum figures and supplemental scalar/spread diagnostics.
4. `finalization_manifest.json` records final derived-file hashes.

All outputs live under the selected result root. The queue-generated legacy
`report/` remains unchanged; this command uses the separate `final_report/`
directory so its spatial-spectrum inventory cannot conflict with the earlier
report without spectra. `--root` and `--archive`
allow alternate locations. The command is CPU-only and disables CUDA visibility
in child processes; it never samples a model, changes raw states, or builds the
paper. The root agent handles manuscript regeneration separately.

Reruns recompute audits and spectra into temporary staging and compare their
JSON contents with any existing products. Existing reports must have the same
file inventory and byte-identical scientific content; PDF creation/modification
timestamps alone are ignored for comparison. Existing files are never replaced.
A previously recorded finalization manifest additionally protects exact file
hashes, including original PDF metadata. A mismatch fails and leaves prior
artifacts untouched. An exclusive lock prevents concurrent finalizers.
Historical two-map and intermediate audits remain separate.

Validation: four focused tests cover successful metadata gating, shortened and
nonfinite-run refusal, unchanged existing-audit preservation, and report mismatch
refusal. Two actual completed-map report renders also matched under the PDF
comparison rule. A readiness check on production data correctly refused the
still-running SI reference. Full four-model finalization has not been run while
references remain incomplete.
