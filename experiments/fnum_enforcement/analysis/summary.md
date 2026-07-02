# Enforced vs soft f/# — matched A/B (seed42, 100k iters)

On-spec band f/# ∈ [3.6,4.4]; sharp = RMS < 20.0 µm; ε-net radius 0.3.

| system | constraint | refined N | f/# med | f/# p10–p90 | % in band | RMS med | RMS min (best) | n sharp(<20µm) | ε-distinct(<20µm) | ε-distinct(ungated) | on-spec&sharp | archive cells |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sixlet | none | 100 | 5.18 | 4.59–5.67 | 7% | 19.0 | 2.1 | 54 | 10 | 17 | 4 | 52003 |
| sixlet | enforced w=1 | 100 | 4.09 | 3.77–4.60 | 75% | 61.6 | 10.2 | 1 | 1 | 28 | 1 | 52635 |
| sixlet | enforced w=10 | 100 | 3.98 | 3.74–4.39 | 86% | 47.8 | 16.0 | 1 | 1 | 20 | 1 | 54252 |
| octuplet | none | 100 | 4.09 | 3.45–4.72 | 48% | 33.9 | 12.2 | 5 | 2 | 14 | 4 | 32150 |
| octuplet | enforced w=1 | 100 | 4.17 | 3.64–4.60 | 70% | 79.0 | 21.3 | 0 | 0 | 21 | 0 | 33801 |
| octuplet | enforced w=10 | 100 | 4.04 | 3.62–4.38 | 82% | 97.0 | 20.2 | 0 | 0 | 20 | 0 | 32454 |
