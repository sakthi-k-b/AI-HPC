# AI-HPC Hybrid Solver: 2D Poisson Equation

A GPU iterative solver (Red-Black Gauss-Seidel in Numba CUDA) that generates convergence trajectories, and a decoder-only Transformer (causal masked self-attention) trained on them to anticipate the solver's remaining convergence from a mid-solve checkpoint. Everything lives in one notebook and runs end to end on a Kaggle T4 in a few minutes.

**Full write-up with all numbers, figures and limitations: [REPORT.md](REPORT.md)**

## What it does

1. Solves ∇²u = f on the unit square (50×50 grid, zero Dirichlet boundaries) with three solvers: pure-Python Gauss-Seidel, compiled Numba CPU Gauss-Seidel, and Red-Black Gauss-Seidel on the GPU. All stop on the same relative residual (1e-6).
2. Studies where the GPU actually pays off (grid sizes 50 to 1600).
3. Generates 100 CUDA convergence trajectories of varied Poisson problems (snapshot every 10 iterations).
4. Trains a Transformer on the iteration history. Each token is one saved grid state. Heads predict the next increment, the jump to the converged state, and the log-residual.
5. Hands off from the solver at a checkpoint (relative residual 1e-3, about 48% of the iterations), predicts the rest, and compares against the HPC result, the analytical solution, and two non-learned baselines (freeze at checkpoint, Aitken extrapolation).
6. Measures a hybrid run: CUDA to checkpoint, then Transformer jump, then CUDA warm-start to the same tolerance.

## Headline results (Kaggle Tesla T4, one run, seed 42)

| Question | Result |
|---|---|
| GPU vs pure-Python sequential GS (50×50) | 27× faster (0.43 s vs 11.7 s) |
| GPU vs compiled Numba CPU (50×50) | **5.4× slower** (0.43 s vs 0.08 s); GPU overtakes between N=100 and N=200, reaching 17× at N=1600 |
| Transformer jump vs freezing at the checkpoint (10 held-out problems) | lower error on 10/10; geometric-mean error ratio 0.42 |
| Transformer jump vs Aitken extrapolation | **Aitken is about 150× more accurate** (0/10 wins for the Transformer) |
| Autoregressive rollout vs freezing | worse (ratio 3.3; better on only 2/10) |
| Unseen benchmark sin(πx)sin(πy): iterations after checkpoint | 1820 → 880 when warm-started from the AI jump (-52%); hybrid 0.33 s vs 0.43 s for full CUDA (1.29×) |
| Same warm-start idea averaged over the 10 held-out problems | 874 → 999 iterations (no benefit); restarting from the Aitken estimate needs 85 |
| Final accuracy of the hybrid result vs analytical | 3.17e-4 vs 3.16e-4 for full CUDA (the discretisation floor) |

In short: the learned jump clearly beats doing nothing, but a one-line classical extrapolation beats it by two orders of magnitude on this problem class, and the warm-start benefit does not carry over to the held-out set. The report discusses why, and what a more convincing follow-up looks like.

## Repository contents

```
ai_hpc_poisson.ipynb   executed notebook (all outputs from the run above)
REPORT.md              full results, analysis, limitations
figures/               plots extracted from the notebook run
```

The notebook also writes `results.json`, `poisson_transformer.pt` (best weights) and `standard_problem.png` to `/kaggle/working`.

## Reproduce

1. Open the notebook on Kaggle, set **Accelerator → GPU T4**, and choose Run All. No external data is needed; every dataset is generated inside the notebook.
2. Versions used: Python 3, PyTorch 2.10.0 (CUDA 12.8), Numba 0.60.0, NumPy, Matplotlib (the standard Kaggle GPU image).
3. Runtime on a T4: solver baselines and scaling study about 1 minute (pure-Python baseline about 12 s), dataset generation 27 s, Transformer training 150 s, evaluation and timing a few seconds.
4. Seeds are fixed (`SEED = 42`, trajectory seeds 100 onward). GPU nondeterminism can still cause small run-to-run differences in the learned model; the HPC solvers are deterministic.
5. Without a GPU the notebook falls back to a NumPy red-black solver and runs, but timings are then not GPU timings. Setting the environment variable `AIHPC_QUICK=1` (or 2) runs a tiny smoke-test configuration.

All main knobs (`N`, `TOL`, `CHKPT_R`, `SEQ_LEN`, `EPOCHS`, `N_TRAJ`, loss weights) are in the first code cell.

## Method in brief

- **Problem:** f = −2π² sin(πx) sin(πy), exact solution sin(πx) sin(πy). Training problems are random sums of sine modes (indices 1 to 3) with known exact solutions; a pure (1,1) mode, the benchmark, is never generated.
- **Residual:** relative, max|Au − f| / max|f|, checked every 10 iterations by all solvers. Checkpoint = first snapshot with residual ≤ 1e-3; tolerance 1e-6.
- **Normalisation:** each trajectory is divided by the amplitude of its checkpoint state (known at hand-off time).
- **Model:** token = one flattened state concatenated with its increment since the previous snapshot (asinh-compressed), linear projection to 256 dims, log-iteration sinusoidal encoding, 4 pre-LN decoder blocks, 4 heads, causal mask, 4.67M parameters. Outputs are asinh-space changes that are added to the current state.
- **Training:** 80/10/10 split by trajectory, teacher forcing on sliding windows of 16 snapshots, AdamW with warm-up and cosine decay, early stopping on validation loss.
- **Error metric:** RMSE over the grid divided by the checkpoint scale, so 1e-3 means 0.1% of the solution amplitude.

## Known limitations

- Many training problems are single-mode, and after amplitude normalisation, problems with the same mode are the same trajectory (up to sign). Several test results are identical to each other, so train/test overlap is likely and the test set is smaller in effective size than its 10 entries. See REPORT.md, section 6.
- One run, one seed; no confidence intervals.
- Hybrid timing is shown for the single benchmark problem only.
- On the 50×50 grid a compiled CPU solver is faster than the GPU, so the hybrid timing gain is relative to the GPU solver, not to the best CPU baseline.
