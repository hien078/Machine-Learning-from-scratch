# Data provenance

`corpus.txt` (~11 KB) contains 18 of Shakespeare's *Sonnets* (1609), transcribed with
modernized spelling: sonnets 1, 2, 12, 18, 29, 30, 33, 55, 60, 65, 71, 73, 104, 106,
116, 130, 138, and 147, separated by blank lines. William Shakespeare died in 1616;
the text is unambiguously in the **public domain** worldwide.

## No-download policy

- The corpus is embedded in the repository; training reads only this file.
- The training script performs **no network access** of any kind.
- Any larger corpus (e.g. a future `--full` mode on the complete works) would require a
  download and therefore explicit user approval first, per repo policy (CLAUDE.md §1:
  dataset downloads and any write inside `data/` require asking).
