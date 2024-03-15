""" This file contains optimizes post processing functions
"""
import numpy as np

def average_within_window_assuming_linear_time_reduceat(values:np.ndarray, timeaxis:np.ndarray, begins:np.ndarray, lengths:np.ndarray, check_linearity:bool=True) -> np.ndarray:

	assert len(timeaxis.shape) == 1

	dt = (timeaxis[-1]-timeaxis[0])/(len(timeaxis)-1)
	o = timeaxis[0]

	assert not check_linearity or np.allclose(np.diff(timeaxis), dt)
	
	begin_indices = np.floor((begins-o)/dt).astype(int)
	end_indices = np.floor((begins+lengths-o)/dt).astype(int)

	begin_index_below, end_index_below = (begin_indices<0), (end_indices<0)
	begin_above_below, end_above_below = (begin_indices>=values.shape[-1]), (end_indices>values.shape[-1])
	out_of_range = begin_index_below|end_index_below|begin_above_below|end_above_below

	begin_indices[begin_index_below] = 0
	begin_indices[(begin_indices>values.shape[-1])] = values.shape[-1]
	
	end_indices[end_index_below] = 0
	end_indices[(end_indices>values.shape[-1])] = values.shape[-1]

	width = end_indices - begin_indices

	reduce_indices = np.vstack([begin_indices, end_indices]).T.flatten()

	# adding one last element to the values and indices, such that the last element also works.
	values_to_reduce = np.concatenate([values, np.full((*values.shape[:-1], 1), np.nan)], axis=-1)
	indices_to_reduce = np.concatenate([reduce_indices, [len(values)]])

	all_summed = np.add.reduceat(values_to_reduce, indices_to_reduce, axis=-1)[..., :-1]
	selected = all_summed[..., ::2]
	assert selected.shape[-1] == len(width)
	averaged = selected/np.maximum(1, width)

	# averaged[np.isinf(averaged)] = np.nan # this should not be necessary, as the inf through 1/width is taken care of by the next line. if infs are now within the returned data, than it should be due to infs in the raw data.
	averaged[..., out_of_range|(width<0)] = np.nan

	return averaged


