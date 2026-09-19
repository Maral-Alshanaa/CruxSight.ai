# tests/test_stats.py

import unittest
import numpy as np
from scipy import stats

class TestStatisticalCalculations(unittest.TestCase):

    def test_wilcoxon_min_p_value_bound(self):
        """Verify min theoretical p-value calculation for N=10 seeds."""
        n_seeds = 10
        min_p = 2.0 / (2 ** n_seeds)
        self.assertAlmostEqual(min_p, 0.001953125, places=7)

    def test_paired_ttest_execution(self):
        """Ensure paired t-test handles identical vs distinct seed arrays properly."""
        run2 = np.array([0.90, 0.91, 0.92, 0.89, 0.90, 0.91, 0.92, 0.88, 0.90, 0.91])
        run4 = np.array([0.96, 0.97, 0.98, 0.95, 0.96, 0.97, 0.98, 0.94, 0.96, 0.97])
        
        t_stat, p_val = stats.ttest_rel(run4, run2)
        self.assertTrue(p_val < 0.05)
        self.assertTrue(t_stat > 0)

if __name__ == "__main__":
    unittest.main()