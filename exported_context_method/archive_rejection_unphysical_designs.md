# Context Export: Rejecting Unphysical Designs (Entrance Pupil Failures) in MAP-Elites Archive

## Project Background
- Project: `/home/hisham26/Desktop/RL` — optical lens design via Quality-Diversity (MAP-Elites / CMA-MEGA) optimization.
- Relevant subsystem: `optical_rl/qd/` (Zernike-based EF50 lens designs, archive-based QD search).
- Current artifact under inspection: `results/models/ef50_zernike_20260409_232730/archive_cells.json`.
- Related context file already in repo: `exported_context_method/seeding_out_of_bounds_measures.md`.

## The User's Current Reasoning (verbatim intent)

### 1. What we currently do during seeding
During the seeding process, we **avoid filling the archive with any setups that fail to calculate the entrance pupil**. Instead of assigning them a very large negative fitness (e.g. `-1e9`) and inserting them anyway, we simply reject them. The motivation:

- We don't want to waste compute mutating an individual that is **completely unphysical**.
- More importantly: the **archive bin (descriptor / measure) the failed individual maps into is itself unreliable**, because the entrance pupil calculation feeding the descriptor was wrong. So even the "location" of the design in behavior space is bogus.

### 1a. Suspected root cause of "BS designs" in current results
The user suspects that many of the garbage designs visible in current archives came from cases where:
- The entrance pupil calculation **failed**, but
- The ray bundle still ended up using the **first surface** in its entirety as a fallback aperture, which gave the design *some* numerical measure / fitness anyway.
- Result: unphysical individuals occupied bins they had no business occupying, were sampled as parents, and their lineage propagated junk through the archive.
- This is a **huge waste of time and compute** and likely a major source of the noise in the published-quality results.

### 2. The proposed extension to mutation / offspring evaluation
The user is asking whether the same rejection rule should apply **outside seeding** — i.e., during the mutation / offspring evaluation loop:

- Currently (presumed): if a mutated child fails the entrance pupil calc, it may still get assigned `fitness = -1e9` and inserted into a bin.
- Even though the fitness is sentinel-low, the child still **occupies measure space**. If it later gets sampled as a parent (even rarely), that's wasted compute on a fundamentally unphysical design.
- Proposal: **outright reject** any mutated offspring whose entrance pupil calculation fails, never insert into the archive at all.
- Rationale: we are not interested in seeing these designs; they shouldn't even appear in the archive.

### 3. Open question — out-of-bounds measures
Separately, the user asks: **what currently happens when the computed measure (descriptor) falls outside the archive bounds?** This is unresolved in the conversation and needs to be checked against the code. There may already be a related note in `exported_context_method/seeding_out_of_bounds_measures.md`.

## What the user wants from the next agent
- Confirm whether the reasoning above is technically sound (no sycophancy — be realistic).
- Decide where to head next: implement rejection in the mutation path, or investigate the out-of-bounds-measure handling first, or both.
- Eventually feed all of this into the **results section** of the paper, since the "BS designs" issue and how it was diagnosed/fixed is part of the methodology story.

## Relevant code locations to investigate (not yet inspected in this chat)
- `optical_rl/qd/ef50_zernike.py` and `optical_rl/qd/ef50_zernike_parallel.py` — main EF50 Zernike QD loop.
- `optical_rl/qd/seed_parallel_experiment.py` — seeding logic; this is presumably where the entrance-pupil rejection rule already lives.
- `optical_rl/qd/map_elites.py` — archive insertion logic; check how out-of-bounds descriptors and `-1e9` sentinel fitnesses are handled.
- `optical_rl/qd/entrance_pupil_experiment.py` and `verify_pupil_sampling.py` — entrance pupil computation and its failure modes.
- `optical_rl/qd/refine_ef50_archive.py` — archive post-processing, possibly relevant to filtering junk.
- Archive being inspected: `results/models/ef50_zernike_20260409_232730/archive_cells.json`.

## Key questions the next agent should answer
1. In the current mutation/offspring evaluation path, what fitness/descriptor is assigned when entrance pupil calc fails? Is the individual inserted into the archive?
2. When a valid-fitness individual produces a descriptor outside the archive's measure bounds, is it (a) clamped, (b) dropped, or (c) inserted into a boundary bin? Each has very different implications for the "BS designs" pathology.
3. Is the "first-surface fallback" behavior (ray bundle using surface 1 entirely when pupil calc fails) actually present in the code, or is the user misremembering? This is the load-bearing claim behind the whole hypothesis.
4. If rejection is added to the mutation path, does the QD loop handle "no valid offspring this generation" gracefully, or will it stall / divide-by-zero on emptiness statistics?

## For the paper's results/methods section
The narrative arc the user is implicitly building:
- **Symptom:** archives contained physically nonsensical designs that nonetheless had finite fitness and occupied legitimate-looking bins.
- **Diagnosis:** these designs originated from entrance-pupil computation failures where a fallback (full first surface as aperture) silently produced numerically valid but meaningless measures.
- **Fix at seeding:** reject unphysical seeds entirely rather than admitting them with sentinel fitness.
- **Proposed fix at mutation:** extend the same rejection rule to offspring, so the archive only ever contains physically meaningful individuals.
- **Impact to report:** cleaner archives, less wasted compute on unphysical lineages, more trustworthy QD coverage statistics.
