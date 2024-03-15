from qupulse_mfli.numpy_magic import average_within_window_assuming_linear_time_reduceat

import numpy as np


def test_average_within_window_assuming_linear_time_reduceat_1(benchmark):

	values = np.array([[0,]*10, [0, 0, 0, 1, 1, 1, 2, 2, 2, 9]])
	timeaxis = np.linspace(0, 1, 10)
	begins = np.array([0, 3, 6, 9])/9
	lengths = np.array([3, 3, 3, 1])/9

	averaged = benchmark(average_within_window_assuming_linear_time_reduceat, values, timeaxis, begins, lengths, True)

	np.testing.assert_allclose(averaged, np.array([[0, 0, 0, 0], [0, 1, 2, 9]]))

def test_average_within_window_assuming_linear_time_reduceat_2(benchmark):

	values = np.ones((3, 100_000))*1e-5
	timeaxis = np.linspace(0, 1, 100_000)

	begins = np.array([0])
	lengths = np.array([1])

	averaged = benchmark(average_within_window_assuming_linear_time_reduceat, values, timeaxis, begins, lengths, True)

	np.testing.assert_allclose(averaged, np.array([[1e-5], [1e-5], [1e-5]]))

def test_average_within_window_assuming_linear_time_reduceat_3(benchmark):

	values = np.ones((3, 100_000))*1e-5
	timeaxis = np.linspace(0, 1, 100_000)
	rng = np.random.default_rng(seed=11223344)

	begins = rng.uniform(0, 1, 100_000)
	lengths = rng.uniform(0.001, 0.1, 100_000)

	averaged = benchmark(average_within_window_assuming_linear_time_reduceat, values, timeaxis, begins, lengths, True)

	assert len(averaged == 100_000)


def get_interesting_testing_pairs():

	# pairs testing if the alignment is correct
	data = np.arange(10)
	timeaxis = np.arange(10)
	
	begins = np.array([0, 3, 6])
	lengths = np.array([3, 3, 3])
	expected = np.array([1, 4, 7])
	yield data, timeaxis, begins, lengths, expected

	begins = np.array([0, 0, 0])
	lengths = np.array([0, 1, 3])
	expected = np.array([0, 0, 1])
	yield data, timeaxis, begins, lengths, expected

	begins = np.array([9, 8, 9, 9])
	lengths = np.array([0, 2, 1, 3])
	expected = np.array([9, (8+9)/2, 9, np.nan])
	yield data, timeaxis, begins, lengths, expected

def test_various_interesting_ones():

	for d, t, b, l, e in get_interesting_testing_pairs():
		averaged = average_within_window_assuming_linear_time_reduceat(d, t, b, l, True)
		np.testing.assert_allclose(averaged, e.astype(float))