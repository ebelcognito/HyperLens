import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from optical_rl.qd.render_singlet_archive_designs import (
    ArchiveDesign,
    _find_latest_archive,
    resolve_theme,
    select_diverse_top,
)


class TestRenderSingletSelection(unittest.TestCase):
    def _design(self, cell, fitness, m1, m2, theta=None):
        if theta is None:
            theta = np.array([0.0, 0.0, 40.0], dtype=np.float64)
        return ArchiveDesign(
            cell=tuple(cell),
            fitness=float(fitness),
            measures=(float(m1), float(m2)),
            theta=np.asarray(theta, dtype=np.float64),
        )

    def test_resolve_theme_custom(self):
        theme = resolve_theme(
            use_default_theme=False,
            bg_color="#F8F5E9",
            lens_fill_color="#9DC08B",
            lens_outline_color="#3A7D44",
            ray_color="#DF6D14",
        )
        self.assertEqual(theme.background, "#F8F5E9")
        self.assertEqual(theme.lens_fill, "#9DC08B")
        self.assertEqual(theme.lens_outline, "#3A7D44")
        self.assertEqual(theme.ray_colors, ["#DF6D14"])

    def test_resolve_theme_default_ignores_custom(self):
        theme = resolve_theme(
            use_default_theme=True,
            bg_color="#111111",
            lens_fill_color="#222222",
            lens_outline_color="#333333",
            ray_color="#444444",
        )
        self.assertEqual(theme.background, "#FFFFFF")
        self.assertIsNone(theme.lens_fill)
        self.assertEqual(theme.lens_outline, "black")
        self.assertEqual(theme.ray_colors, ["#CC0000", "#006600", "#0066CC"])

    def test_selector_starts_with_best(self):
        designs = [
            self._design((0, 0), 1.0, 0.0, 0.0),
            self._design((0, 1), 0.8, 1.0, 0.0),
            self._design((1, 0), 0.7, 0.0, 1.0),
        ]
        selected = select_diverse_top(designs, count=3, top_frac=1.0, seed=0)
        self.assertEqual(selected[0].fitness, 1.0)

    def test_selector_unique_and_count(self):
        designs = [self._design((i, 0), 1.0 - i * 0.1, i, 0.0) for i in range(10)]
        selected = select_diverse_top(designs, count=5, top_frac=0.6, seed=1)
        self.assertEqual(len(selected), 5)
        self.assertEqual(len({id(x) for x in selected}), 5)

    def test_selector_spreads_measure_space(self):
        # Top fitness points cluster near m1 ~ 0; we still expect far points to appear.
        clustered = [
            self._design((i, 0), 10.0 - i * 0.01, 0.01 * i, 0.01 * i) for i in range(20)
        ]
        far = [
            self._design((30, 30), 9.7, 3.0, 3.0),
            self._design((31, 31), 9.6, -3.0, -3.0),
        ]
        selected = select_diverse_top(clustered + far, count=6, top_frac=1.0, seed=0)
        measures = np.array([d.measures for d in selected], dtype=np.float64)
        span = measures.max(axis=0) - measures.min(axis=0)
        self.assertGreater(span[0], 1.0)
        self.assertGreater(span[1], 1.0)

    def test_find_latest_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_old = root / "singlet_cmamega_20260101_000000"
            run_new = root / "singlet_cmamega_20260101_000001"
            run_old.mkdir(parents=True, exist_ok=True)
            run_new.mkdir(parents=True, exist_ok=True)
            old_archive = run_old / "archive.json"
            new_archive = run_new / "archive.json"
            old_archive.write_text("[]", encoding="utf-8")
            new_archive.write_text("[]", encoding="utf-8")
            os.utime(old_archive, (1_700_000_000, 1_700_000_000))
            os.utime(new_archive, (1_800_000_000, 1_800_000_000))
            latest = _find_latest_archive(root)
            self.assertEqual(latest, new_archive)


if __name__ == "__main__":
    unittest.main()
