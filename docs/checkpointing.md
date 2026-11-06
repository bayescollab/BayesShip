# Checkpointing and resuming

BayesShip can stop a run partway through (killed, timed out or signalled) and continue it later. A resumed run
continues the same chains: it does not repeat burn-in, and it keeps the samples, temperatures and proposal state
from before the stop.

This page covers:
1. what is written to disk, and when;
2. how checkpointing is switched on and which options control it.

The code is in `src/bayesshipSampler.cpp` (`sample`, `commitBatch`, `writeCheckpoint`, `loadCheckpoint`) and
`src/dataUtilities.cpp` (batch and history-window files).

---

## 1. What is stored, and when

### Files

Every name below starts with `<outputDir><outputFileMoniker>`. The sampler adds no separator, so `outputDir`
needs a trailing `/`. `<M>` stands for the moniker.

```
<outputDir>/
├── <M>Prior_output.hdf5          prior exploration (if priorIterations > 0 and writePriorData)
├── <M>Burn_output.hdf5           burn-in samples (if burnIterations > 0)
├── <M>_batches/
│   ├── batch_000000.hdf5         samples of sampling batch 0
│   ├── batch_000001.hdf5         ...one file per batch, never rewritten
│   └── batch_tmp.hdf5            (only while a batch is being written)
├── <M>_checkpoints/
│   ├── gen_000041/               older complete generation (kept as a fallback)
│   ├── gen_000042/               newest complete generation
│   │   ├── sampler.json                                   sampler state (see below)
│   │   ├── historyWindow.hdf5                             recent hot-chain steps (if the window is active)
│   │   ├── <M>_checkpoint_gaussianProposalVariables.json  Gaussian proposal widths
│   │   ├── <M>_checkpoint_blockFisherProposalRef.json     pointer to the Fisher state (Fisher proposals only)
│   │   └── MANIFEST.json                                  size + hash of every file above
│   └── gen_000043.tmp/           (only while a generation is being written)
├── <M>_fisher/
│   └── blockFisherProposal.json  Fisher matrices + eigen-decompositions (or fisherProposal.json)
├── <M>_stat.txt                  acceptance/swap statistics, rewritten every batch
└── <M>_output.hdf5               full cold-chain history, written only when the run finishes
```

### Timeline of a run

| Phase | Written | Checkpoint? |
|---|---|---|
| Prior exploration | `<M>Prior_output.hdf5` at the end | No |
| Burn-in | `<M>Burn_output.hdf5` at the end | No |
| Each sampling batch | One batch file, then one checkpoint generation, then the stat files | Yes, after every batch |
| Run reaches its target | `<M>_output.hdf5` (the whole history in one file) | — |
| Run stops early | Nothing extra: the last generation already describes everything | — |

Checkpoints are only written once sampling has started. If a run is killed during prior exploration or burn-in,
nothing can be resumed and it has to start over. A stop signal received during burn-in is acted on after the
first sampling batch has been committed.

The batch length is `batchSize`. If `batchSize` is 0, or larger than `iterations`, the whole run is one batch:
there is one checkpoint at the end and no chance to stop early. When `independentSamples` is used with
`batchSize = 0`, the batch length adapts to the autocorrelation length (it starts at `independentSamples`).

### What happens after each batch (`commitBatch`)

The order is chosen so that a kill at any point leaves the disk in a usable state:

1. **Write the samples.** Rows added since the previous batch are written to `_batches/batch_tmp.hdf5`. That
   means the stored chains only: cold chains when `coldOnlyStorage`, otherwise all of them. The file is
   fsync'd, hashed, and renamed to `batch_NNNNNN.hdf5`. If this step fails, those rows are retried with the
   next batch.
2. **Write a checkpoint generation** in `gen_NNNNNN.tmp/`:
   - `sampler.json`;
   - the hot-chain history window;
   - each proposal's own checkpoint file;
   - `MANIFEST.json`, which records every file's size and FNV-1a hash.

   Everything is fsync'd, then the directory is renamed to `gen_NNNNNN/`. A generation therefore either exists
   complete or does not exist at all.
3. **Prune.** Keep the newest `checkpointGenerationsToKeep` generations and delete leftover `.tmp` directories.
4. **Rewrite the stat files** atomically (temporary file, then rename).

If the run is killed between steps 1 and 2, the new batch file is not listed in any checkpoint. On resume it is
simply overwritten.

If writing fails three batches in a row (for example, the disk is full), the run stops with an error rather
than carrying on with a stale checkpoint.

### Contents of `sampler.json`

- **Layout:** `maxDim`, `minDim`, `RJ`, `ensembleSize`, `ensembleN`, `coldOnlyStorage`.
- **Temperatures:** `betaSchedule`, and `betas` (the temperatures after burn-in).
- **Random number seed:** `seed`.
- **Settings at the time of writing:** `swapProb`, `threadPool`, `randomizeSwapping`, `averageDynamics`. These
  are recorded only for reference; a resume does not use them (see section 2).
- **Progress:**
  - `samplingStarted`;
  - `rowsStored`, and `rowsStoredPerChain` (steps already in batch files);
  - `batchFiles`: the name, size and hash of every batch file this generation relies on.
- **Each chain's final state:** `finalPosition`, which holds the parameters, the log likelihood and the log
  prior (plus status and model ID for RJ).
- **Evidence:** `evidenceTotals`, each chain's log-likelihood sum and step count since the start of sampling.
  Chains whose history is not stored need these for the evidence to carry over.
- **Hot-chain window:** `historyWindow`, giving the file name, the first chain and the number of rows.
- **Prior ranges:** `priorRanges`, if they were set.

### Proposal state

| Proposal | Stored | Where |
|---|---|---|
| Gaussian | Adapted per-chain widths | `<M>_checkpoint_gaussianProposalVariables.json` in each generation |
| Fisher / block Fisher | Fisher matrices and their eigen-decompositions (frozen after burn-in) | `<M>_fisher/`. It is written once and rewritten only when a matrix changes. Each generation holds a small reference file with that file's size and hash. |
| Differential evolution, GMM, KDE | Nothing of their own | They learn from chain history: the batch files for cold chains, `historyWindow.hdf5` for hot chains. |

### Hot-chain history window

When `coldOnlyStorage` is on (and the run is not RJ), only cold chains go to the batch files. Each hot chain
keeps only its last `hotChainHistoryLength` steps in memory, trimmed after every batch. This window feeds the
history-based proposals.

Each generation saves the window as `historyWindow.hdf5` and restores it on resume. The log likelihoods of the
trimmed steps are added to the evidence offsets, so the evidence still counts every step.

### Integrity checks and fallback

On resume, the newest generation is checked first:
- every file in its manifest must have the recorded size and hash;
- every batch file listed in `sampler.json` must exist and match its recorded size and hash.

If any check fails, the sampler prints a WARNING and tries the next older generation.

If generations exist but **none** passes, the sampler stops with an error instead of starting over silently.
If there are no generation directories at all, it falls back to the older single-file format,
`<M>_checkpoint.json`.

### Partial output

`<M>_output.hdf5` is only written when the run finishes. For a run that is stopped or still going, use

```
python python/merge_batches.py <outputDir> <moniker> [merged.hdf5]
```

It joins the batch files listed by the newest readable generation into the usual `MCMC_OUTPUT` layout.

---

## 2. Turning it on and setting options

### Checkpointing is always on

Every call to `sample()` writes batch files and checkpoint generations; there is no switch to turn this off.
HDF5 support is required for the batch files and the history window (the `_HDF5` build flag). Without it, only
`sampler.json` and the proposal files are written, and earlier samples cannot be reloaded on resume.

### Resuming is automatic

At the start of `sample()`, if `<M>_checkpoints/gen_*` (or a legacy `<M>_checkpoint.json`) exists in
`outputDir` and `ignoreExistingCheckpoint` is false, the sampler resumes:

1. It loads the newest valid generation (see the fallback rules above).
2. It advances the seed by `101 × ensembleSize × ensembleN`, so the resumed run does not replay the random
   numbers of the run it continues. The Gaussian proposal's generators are reseeded from the new seed.
3. It skips prior exploration and burn-in.
4. It re-evaluates the likelihood and prior at every chain's saved position and compares them with the
   recorded values. If any differ by more than `resumeValueRelTol`, it stops: the likelihood or prior has
   changed, for example through different code, config or build.
5. It reloads the stored samples from the batch files, the hot-chain window and the evidence totals.
6. It continues until the original target (`iterations` or `independentSamples`) counted over the whole run.

The run must be set up with the same `maxDim`, `ensembleSize`, `ensembleN` and `coldOnlyStorage` as the run
being continued. A mismatch is an error.

### Which values are used on resume

| Taken from the **checkpoint** | Taken from the **current settings** |
|---|---|
| Layout, `betaSchedule`, `betas`, chain positions, `seed` (then advanced), `priorRanges` | `swapProb`, `threadPool`, `randomizeSwapping`, `averageDynamics` (a `NOTE:` line is printed when one differs from the checkpoint) |
| | Also every option not stored in the checkpoint: `iterations`, `independentSamples`, `batchSize`, `threads`, proposal weights, and the options below |

This lets a resumed run use more threads, a different swap rate, or a larger target.

### Options (members of `bayesshipSampler`)

| Member | Default | Meaning |
|---|---|---|
| `outputDir` | `""` | Directory for everything above. Needs a trailing `/`. |
| `outputFileMoniker` | `"BayesShip"` | Prefix of every file name. A resume must use the same moniker. |
| `batchSize` | 0 | Steps between checkpoints (see the timeline). |
| `ignoreExistingCheckpoint` | false | true starts fresh even if a checkpoint exists. **Prefer an empty `outputDir`:** a fresh run writes `batch_000000.hdf5` onward over the old run's batch files, which leaves the old generations unusable. |
| `checkpointGenerationsToKeep` | 2 | Complete generations kept on disk. |
| `hotChainHistoryLength` | 2000 | Hot-chain window length (only with `coldOnlyStorage`, non-RJ). 0 keeps the full history in memory and saves no window. |
| `resumeValueRelTol` | 1e-10 | Relative tolerance for the likelihood/prior check on resume. A negative value disables the check. |
| `wallTimeLimitSeconds` | 0 | Stop after a batch once this many seconds since `sample()` began leave no room for another batch. 0 = no limit of its own. |
| `wallTimeMarginSeconds` | 120 | Safety margin added to the budget for the next batch. |
| `handleStopSignals` | true | Catch SIGTERM, SIGUSR1 and SIGINT while sampling, and stop after the current batch. |
| `stoppedEarly` | (output) | Set by `sample()`: true if it returned because of a signal or the time limit before reaching its target. |

### Environment variables

| Variable | Effect |
|---|---|
| `SLURM_JOB_END_TIME` | Set by SLURM. The job's end time is always respected, together with any other limit. |
| `BAYESSHIP_WALL_TIME_LIMIT` | Seconds; used when `wallTimeLimitSeconds` is 0. |
| `BAYESSHIP_WALL_TIME_MARGIN` | Seconds; overrides `wallTimeMarginSeconds`. |

### Stopping early

After every committed batch, the sampler stops early if either:
- a stop signal arrived; or
- the remaining time is less than `2 × (slowest batch so far) + margin`. The remaining time is the earlier of
  the wall-time limit and the SLURM end time.

When it stops early:
- `sample()` prints `STOPPED EARLY ... Checkpoint generation N is complete`;
- it sets `stoppedEarly = true` and returns;
- it does not write `<M>_output.hdf5`.

A second signal kills the process straight away; the last complete generation is still valid.

Signals only reach the sampler if they are delivered to the process. Under SLURM, use
`#SBATCH --signal=B:USR1@300`. If the program runs under a batch shell, that shell must pass the signal on to
it: start the program in the background, `trap` the signal and forward it with `kill`, then `wait`.

---

## Limitations

- **RJ runs:**
  - Earlier samples stay in the batch files but are not reloaded, so the final `_output.hdf5` only covers the
    steps since the last restart.
  - There is no hot-chain window.
- **GMM:** the fitted mixture is not stored. On resume it is fitted again from the reloaded or windowed
  history.
- **Bit-for-bit reproducibility:** resuming does not give the same samples as an uninterrupted run, because of
  the seed advance. Separately, the Armadillo generator used inside the GMM proposal is not seeded per chain,
  so even fresh runs with the same seed are not exactly reproducible once GMM is active.
- **Older checkpoints:** a checkpoint written before these changes (`_checkpoint.json`, or a
  generation without the log likelihoods or evidence totals) still loads. The checks it can't support are
  skipped, with a `NOTE:`.
