"""Deterministic portability and fail-closed validation regressions."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import stats

import validate
from semis_rel import Distribution, Ex1, MIS_Rel, Nataf, SeMIS, SuS, aCS, makedist
from semis_rel.log_operations import logcumsum


class NatafTests(unittest.TestCase):
    def test_frozen_wrappers_without_parameter_attributes(self):
        marginals = [Distribution('Normal', stats.norm(loc=2, scale=3)),
                     Distribution('Lognormal', stats.lognorm(s=.4, scale=np.exp(.2))),
                     Distribution('Generalized Extreme Value', stats.genextreme(c=-.1, loc=2, scale=3))]
        u = np.linspace(-3, 3, 24).reshape(3, 4, 2)
        actual = Nataf(marginals, np.eye(3)).U2X(u)
        expected = np.array([m.icdf(stats.norm.cdf(ui)) for m, ui in zip(marginals, u)])
        np.testing.assert_allclose(actual, expected, rtol=1e-14)
        self.assertEqual(actual.shape, u.shape)

    def test_explicit_parameters_retain_tail_transforms(self):
        marginals = [makedist('Normal', mu=2, sigma=3), makedist('Lognormal', mu=.2, sigma=.4)]
        u = np.array([[-10., 0., 10.], [-10., 0., 10.]])
        expected = np.vstack([2 + 3 * u[0], np.exp(.2 + .4 * u[1])])
        np.testing.assert_array_equal(Nataf(marginals, np.eye(2)).U2X(u), expected)

    def test_uniform_correlations_of_both_signs(self):
        uniform = makedist('Uniform', lower=0, upper=1)
        for rho in (-.5, .5):
            with self.subTest(rho=rho):
                model = Nataf([uniform, uniform], [[1, rho], [rho, 1]])
                # For uniform marginals rho_X = (6/pi) asin(rho_U/2).
                self.assertAlmostEqual(model.Rho_U[0, 1], 2 * np.sin(np.pi * rho / 6), delta=2e-4)
                np.testing.assert_allclose(model.RhoL_U @ model.RhoL_U.T, model.Rho_U)

    def test_zero_correlation_in_nonidentity_matrix(self):
        uniform = makedist('Uniform')
        rho = np.array([[1, .5, 0], [.5, 1, 0], [0, 0, 1.]])
        model = Nataf([uniform] * 3, rho)
        self.assertEqual(model.Rho_U[0, 2], 0)
        self.assertEqual(model.Rho_U[1, 2], 0)


class LogTests(unittest.TestCase):
    def test_signed_cumulative_sum_both_axes(self):
        values = np.array([[1., -3., 5.], [2., -5., 7.]])
        logs = np.log(values.astype(complex))
        for axis in (0, 1):
            with self.subTest(axis=axis):
                inp = logs if axis == 1 else logs.T
                expected = np.cumsum(values if axis == 1 else values.T, axis=axis)
                np.testing.assert_allclose(np.exp(logcumsum(inp, axis)), expected, atol=1e-13)

    @unittest.expectedFailure
    def test_known_inherited_all_negative_logsum_limitation(self):
        # logcumsum delegates to logsum, whose all-negative-slice limitation
        # also exists upstream; this fix only preserves complex output storage.
        inp = np.log(np.array([[-3., -5.]], dtype=complex))
        np.testing.assert_allclose(np.exp(logcumsum(inp, 1)), [[-3., -8.]], atol=1e-13)

    def test_positive_cumulative_sum_stays_real(self):
        values = np.array([[1., 3., 5.], [2., 5., 7.]])
        result = logcumsum(np.log(values), 1)
        self.assertFalse(np.iscomplexobj(result))
        np.testing.assert_allclose(np.exp(result), np.cumsum(values, axis=1))


class SuSTests(unittest.TestCase):
    def test_single_output_shape(self):
        result = SuS(Ex1()).EvlY(np.zeros((2, 3)))
        self.assertEqual(result.Y.shape, (2, 3))
        model = SuS(Ex1())
        with mock.patch.object(model.TF, 'EvlLSF', return_value=np.zeros((3, 1))):
            self.assertEqual(model.EvlY(np.zeros((2, 3))).Y.shape, (2, 3))
        with mock.patch.object(model.TF, 'EvlLSF', return_value=1.0):
            self.assertEqual(model.EvlY(np.zeros((2, 1))).Y.shape, (2, 1))

    def test_multiple_outputs_and_estimator(self):
        class Multi:
            Ndim = 2
            Nfun = 2
            Dist = Ex1().Dist
            def EvlLSF(self, x):
                return np.vstack([1.5 - x[0], 2 - x[1]])
        model = SuS(Multi())
        u = np.array([[1., 2., 3.], [4., 5., 6.]])
        result = model.EvlY(u)
        np.testing.assert_array_equal(result.Y[1:], Multi().EvlLSF(u))
        self.assertEqual(result.Y.shape, (3, 3))
        with mock.patch.object(model.TF, 'EvlLSF', return_value=np.array([1., 2.])):
            self.assertEqual(model.EvlY(np.zeros((2, 1))).Y.shape, (3, 1))
        np.random.seed(21)
        sampler = SeMIS(model, aCS())
        sampler.NumSam = 200
        sampler.Verbose = False
        output = sampler.RunIte()
        self.assertTrue(output['flg_cvg'])
        self.assertTrue(all(y.shape[0] == 5 for y in output['y']))
        pf = MIS_Rel(output).Pf_hat
        self.assertEqual(pf.shape, (2,))
        self.assertTrue(np.all(np.isfinite(pf)))

    def test_invalid_output_shape_is_rejected(self):
        model = SuS(Ex1())
        with mock.patch.object(model.TF, 'EvlLSF', return_value=np.zeros((2, 2))):
            with self.assertRaisesRegex(ValueError, 'Nfun, Nsam'):
                model.EvlY(np.zeros((2, 3)))


class ValidationTests(unittest.TestCase):
    def test_nonfinite_empty_and_shape_mismatch_fail(self):
        for a, b in [([1], [np.nan]), ([np.nan], [np.nan]), ([np.inf], [-np.inf]),
                     ([np.inf], [1]), ([1], [np.inf]), ([], []), ([1], [1, 1]),
                     ([1 + 1j], [1])]:
            with self.subTest(a=a, b=b):
                self.assertEqual(validate.rel(a, b), np.inf)
        self.assertEqual(validate.rel([np.inf, 2], [np.inf, 2]), 0)
        self.assertEqual(validate.rel([1, 2], [[1], [2]]), 0)

    def test_empty_reference_directory_fails(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(validate.replay(directory))

    def test_missing_pair_or_empty_reference_fails(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(validate, 'REPLAY_CASES', ('sus_Ex1_0_1',)):
            Path(directory, 'rec_sus_Ex1_0_1.mat').write_bytes(b'not loaded')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(validate.replay(directory))
                Path(directory, 'res_sus_Ex1_0_1.mat').touch()
                self.assertFalse(validate.replay(directory))

    def test_statistics_nonfinite_evaluation_counts_fail(self):
        result = ({"Ncal": [np.nan]}, validate.sr.PF_REF['Ex1'], np.nan, np.nan)
        with mock.patch.object(validate, 'run_once', return_value=result), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(validate.statistics(2, 0, ['Ex1']))

    def test_statistics_requires_replicates(self):
        with self.assertRaisesRegex(ValueError, 'at least two'):
            validate.statistics(1, 0, ['Ex1'])


if __name__ == '__main__':
    unittest.main()
