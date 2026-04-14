# Exported chat context — ef50_zernike SIGINT / checkpoint behavior

**Date of session:** 2026-04-11
**Working directory:** `/home/hisham26/Desktop/RL`
**File the user had open in IDE at the time:**
`results/models/ef50_zernike_20260410_192850/ef50_zernike.yaml`

This chat was a short debugging / clarification session about why pressing
Ctrl+C on a running ef50_zernike QD job repeatedly prints
`[ef50-zernike] SIGINT received — checkpointing then exiting.` and where the
resulting checkpoint is saved. It is being exported as part of a larger
collection of agent-task transcripts the user is gathering to feed into a
larger model while writing the Results / Methods section of their paper.

---

## User's question

> why is this happenning, where is the checkpoint getting saved to??

Triggered by the terminal output:

```
^C[ef50-zernike] SIGINT received — checkpointing then exiting.
^C[ef50-zernike] SIGINT received — checkpointing then exiting.
^C[ef50-zernike] SIGINT received — checkpointing then exiting.
^C[ef50-zernike] SIGINT received — checkpointing then exiting.
^C[ef50-zernike] SIGINT received — checkpointing then exiting.
```

---

## Investigation — files and code grepped

A grep for `SIGINT received — checkpointing` showed the same cooperative
SIGINT pattern is implemented across many of the QD runners in the repo:

- `optical_rl/qd/triplet_batched.py` — `[triplet-batched]`
- `optical_rl/qd/ef50_zernike.py` — `[ef50-zernike]`
- `optical_rl/qd/cmamega_ext/runner_cmamega.py` — `[cmamega]`
- `optical_rl/qd/ef50_zernike_parallel.py` — `[ef50-zernike]`
- `optical_rl/qd/cmamega_triplet/runner_triplet_cmamega.py` — `[triplet-cmamega]`
- `optical_rl/qd/triplet_cooke_zernike.py` — `[triplet-zernike]`

In every case the structure is the same: install a SIGINT handler that sets a
flag, check the flag at the top of the main generation loop, and flush a
checkpoint in a `finally` block.

### Relevant code in `optical_rl/qd/ef50_zernike.py`

Config option for periodic checkpointing (line 225-226):

```python
# Stop-and-go checkpointing. See ef50_zernike_parallel.py for full doc.
checkpoint_every: int = 50
```

Checkpoint path construction (line 1209):

```python
ckpt_path = (Path(result_dir) / "checkpoint.pkl") if result_dir is not None else None
```

Checkpoint builder / flusher (lines ~1218-1234):

```python
def _build_ckpt() -> "CMAMEGACheckpoint":
    return CMAMEGACheckpoint(
        ...
    )

def _flush_checkpoint() -> None:
    ...
    save_checkpoint(_build_ckpt(), ckpt_path)
```

SIGINT handler install (lines 1237-1247):

```python
prev_handler = signal.getsignal(signal.SIGINT)

def _handle_sigint(signum, frame):  # noqa: ARG001
    sigint_received["flag"] = True
    print(
        "[ef50-zernike] SIGINT received — checkpointing then exiting.",
        flush=True,
    )

signal.signal(signal.SIGINT, _handle_sigint)
```

Main loop check + periodic flush + final flush in `finally`
(lines ~1249-1373):

```python
try:
    while self.completed < target_completed:
        if sigint_received["flag"]:
            break
        ...
        if checkpoint_every > 0 and self.generation % checkpoint_every == 0:
            _flush_checkpoint()
finally:
    try:
        _flush_checkpoint()
    except Exception as e:
        print(f"[ef50-zernike] WARNING: final checkpoint failed: {e}", flush=True)
    signal.signal(signal.SIGINT, prev_handler)
```

Resume side (lines ~1569-1609) loads from `resume_dir / "checkpoint.pkl"`
via `optical_rl.qd._checkpoint.load_checkpoint`, with a fallback warning for
legacy runs that only have `archive.pkl`.

---

## Answer given to the user

**This is expected behavior, not a bug — the run traps Ctrl+C so it can exit
cleanly instead of being killed mid-generation.**

What's happening:

- A SIGINT handler is installed that just sets `sigint_received["flag"] = True`
  and prints the `[ef50-zernike] SIGINT received — checkpointing then exiting.`
  line.
- The main loop checks that flag at the top of each generation and breaks out.
- The `finally` block then calls `_flush_checkpoint()`.
- Each repeated `^C` re-enters the handler (so you see the line printed
  again each time), but the process only actually exits when the *current*
  generation finishes and the loop's flag check fires at the top of the next
  iteration.

**Where the checkpoint is saved:** `Path(result_dir) / "checkpoint.pkl"`.
For the run whose YAML the user had open, that resolves to:

```
results/models/ef50_zernike_20260410_192850/checkpoint.pkl
```

**Practical advice given:**

- If you want it to die faster, just wait — extra `^C` presses don't help
  once the flag is already set.
- If you want a hard kill, `Ctrl+\` (SIGQUIT) or `kill -9` will work, but
  you'll lose all progress since the last *periodic* checkpoint, which is
  written every `checkpoint_every=50` generations (configurable in the
  YAML's `qd:` section).
- Normal Ctrl+C is the safe path: it always writes a final checkpoint via
  the `finally` block before exiting.

---

## Notes for the writeup model

Things in this transcript that may be useful when writing the Methods /
Results section of the paper:

1. **Cooperative-shutdown checkpointing is a shared design pattern across
   the QD runners in this codebase** (ef50_zernike, ef50_zernike_parallel,
   triplet_batched, triplet_cooke_zernike, cmamega_ext/runner_cmamega,
   cmamega_triplet/runner_triplet_cmamega). All of them install a SIGINT
   handler that just flips a flag, check the flag at generation
   boundaries, and flush a `checkpoint.pkl` in a `finally` block. This
   means runs are interruptible and resumable without losing more than the
   work since the last periodic flush.
2. **Periodic checkpoint cadence is `checkpoint_every` generations**
   (default 50 for ef50_zernike), set in the YAML `qd:` section.
3. **Checkpoint format:** pickled `CMAMEGACheckpoint` objects, written via
   `optical_rl.qd._checkpoint.save_checkpoint` and read back via
   `load_checkpoint`. The runner records a `config_hash` so resumes can
   detect config drift; there is also a legacy fallback for old runs that
   only persisted `archive.pkl`.
4. **Per-run output directory layout** (inferred from this run):
   `results/models/ef50_zernike_<YYYYMMDD_HHMMSS>/` containing at least
   `ef50_zernike.yaml` (the frozen config) and `checkpoint.pkl` (latest
   state). The runner is `optical_rl/qd/ef50_zernike.py`, the underlying
   algorithm is CMA-MEGA (the checkpoint dataclass is `CMAMEGACheckpoint`).
5. **The specific run referenced in this session** is
   `ef50_zernike_20260410_192850`, started 2026-04-10 19:28:50.
