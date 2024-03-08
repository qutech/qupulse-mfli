""" This file contains optimizes post processing functions
"""
import matplotlib.pyplot as plt
import numpy as np

"""

-> https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.reduce_window.html

"""

def average_within_window_assuming_linear_time_reduceat(values:np.ndarray, timeaxis:np.ndarray, begins:np.ndarray, lengths:np.ndarray, check_linearity:bool=True) -> np.ndarray:

	assert len(timeaxis.shape) == 1

	dt = (timeaxis[-1]-timeaxis[0])/(len(timeaxis)-1)
	o = timeaxis[0]

	assert not check_linearity or np.allclose(np.diff(timeaxis), dt)

	begin_indeces = np.round((begins-o)/dt).astype(int)
	end_indeces = np.round((begins+lengths-o)/dt).astype(int)

	begin_index_below, end_index_below = (begin_indeces<0), (end_indeces<0)
	begin_above_below, end_above_below = (begin_indeces>=values.shape[-1]), (end_indeces>=values.shape[-1])
	out_of_range = begin_index_below|end_index_below|begin_above_below|end_above_below

	begin_indeces[begin_index_below] = 0
	begin_indeces[begin_above_below] = values.shape[-1]-1
	
	end_indeces[end_index_below] = 0
	end_indeces[end_above_below] = values.shape[-1]-1

	width = end_indeces - begin_indeces

	reduce_indeces = np.vstack([begin_indeces, end_indeces]).T.flatten()

	all_summed = np.add.reduceat(values, reduce_indeces, axis=-1)
	selected = all_summed[..., ::2]
	assert selected.shape[-1] == len(width)
	averaged = selected/np.maximum(1, width)

	# averaged[np.isinf(averaged)] = np.nan # this should not be necessary, as the inf through 1/width is taken care of by the next line. if infs are now within the returned data, than it should be due to infs in the raw data.
	averaged[:, out_of_range&(width<=0)] = np.nan

	return averaged

def test_average_within_window_assuming_linear_time_reduceat_1(benchmark):

	values = np.array([[0,]*10, [0, 0, 0, 1, 1, 1, 2, 2, 2, 9]])
	timeaxis = np.linspace(0, 1, 10)
	begins = np.array([0, 3, 6, 9])/9
	lengths = np.array([3, 3, 3, 1])/9

	averaged = benchmark(average_within_window_assuming_linear_time_reduceat, values, timeaxis, begins, lengths, True)

	assert np.allclose(averaged, np.array([[0, 0, 0, 0], [0, 1, 2, 9]]))


def test_average_within_window_assuming_linear_time_reduceat_2(benchmark):

	values = np.ones((3, 100_000))*1e-5
	timeaxis = np.linspace(0, 1, 100_000)

	begins = np.array([0])
	lengths = np.array([1])

	averaged = benchmark(average_within_window_assuming_linear_time_reduceat, values, timeaxis, begins, lengths, True)

	assert np.allclose(averaged, np.array([[1e-5], [1e-5], [1e-5]]))
