""" This file contains optimizes post processing functions
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax

"""

-> https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.reduce_window.html

"""

def average_within_window_assuming_linear_time_reduceat(values:np.ndarray, timeaxis:np.ndarray, begins:np.ndarray, lengths:np.ndarray, check_linearity:bool=True) -> np.ndarray:

	assert len(timeaxis.shape) == 1

	dt = (timeaxis[-1]-timeaxis[0])/(len(timeaxis)-1)
	o = timeaxis[0]

	assert not check_linearity or np.allclose(np.diff(timeaxis), dt)

	# from ns to bins: i_bin = np.round(t-o)/dt)

	print((begins-o)/dt)
	print((begins+lengths-o)/dt)

	begin_indeces = np.round((begins-o)/dt).astype(int)
	end_indeces = np.round((begins+lengths-o)/dt).astype(int)
	width = end_indeces - begin_indeces

	print(begin_indeces, end_indeces)
	end_indeces[end_indeces==values.shape[-1]] = values.shape[-1]-1

	reduce_indeces = np.vstack([begin_indeces, end_indeces]).T.flatten()

	all_summed = np.add.reduceat(values, reduce_indeces, axis=-1)
	selected = all_summed[..., ::2]
	assert selected.shape[-1] == len(width)
	averaged = selected/width

	return averaged

@jax.jit
def average_within_window_assuming_linear_time_jitted(values:np.ndarray, timeaxis:np.ndarray, begins:np.ndarray, lengths:np.ndarray) -> np.ndarray:

	dt = (timeaxis[-1]-timeaxis[0])/(len(timeaxis)-1)
	o = timeaxis[0]

	begin_indeces = jnp.round((begins-o)/dt).astype(int)
	end_indeces = jnp.round((begins+lengths-o)/dt).astype(int)
	width = end_indeces - begin_indeces

	mask = jnp.zeros((timeaxis.shape[0], begins.shape[0]))

	averaged = jnp.nanmean(values, axis=-1)
	reduce_indeces = jnp.vstack([begin_indeces, end_indeces]).T.flatten()
	all_summed = jnp.add.reduceat(values, reduce_indeces, axis=-1)
	selected = all_summed[..., ::2]

	averaged = selected/width

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

def test_average_within_window_assuming_linear_time_2_jitted(benchmark):

	values = np.ones((3, 100_000))*1e-5
	timeaxis = np.linspace(0, 1, 100_000)

	begins = np.array([0])
	lengths = np.array([1])

	averaged = benchmark(average_within_window_assuming_linear_time_jitted, jnp.array(values), jnp.array(timeaxis), jnp.array(begins), jnp.array(lengths))

	assert np.allclose(averaged, np.array([[1e-5], [1e-5], [1e-5]]))






