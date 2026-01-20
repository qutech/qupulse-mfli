""" this module contains a mocked MFLI api for testing purposes
"""

from typing import List, Dict, Union, Literal
import numpy as np
import time

class MockMFLI:
	def __init__(self, serial:str):

		# variables to facilitate the minimal working requirements
		self.subscribed_nodes = set()
		self.setted_values = {
			f"/{serial}/clockbase": 100,
			}
		self._serial = serial

		# variable to test the data playback functionality
		self._loaded_readout_values : Dict[str, np.ndarray] = {} # will contain the data for each channel that is to be replayed
		self._loaded_readout_time_axis : Union[np.ndarray, None] = None # the time axis of the pulse that is to be returned
		self._loaded_trigger : Literal[0, 1] = 0 # the trigger channel that is to be set to high when data is send
		self._loaded_readout_chunk_size : int = 100 # the number of elements for one playback chunk
		self._loaded_readout_chunk_size_jitter : Union[None, int] = 10 # the standard deviation of the returned chunk size
		self._playback_state : Literal['armed', 'running', 'completed', 'not_configured'] = 'not_configured'
		self._playback_position : int = 0 # the index in the pulse which might be returned with the next request.
		self._last_oll_time : Union[None, float] = None # the last point in time at which a poll event was performed
		self._playback_artificial_wait : float = 100e-3 # the time that that the poll function should wait before returning the next chunk of data.
		self._lowlevel_node_components = ["x", "y", "frequency", "phase", "dio", "trigger", "auxin0", "auxin1"] # the names of the nodes that the MFLI returns. This variable is not to be changed.

	@property
	def serial(self):
		return self._serial
	
	def subscribe(self, node_path):
		assert node_path != "*"
		self.subscribed_nodes.add(node_path)
		return True

	def unsubscribe(self, node_path):
		if node_path == "*":
			self.subscribed_nodes.clear()
		else:
			self.subscribed_nodes.discard(node_path)

	def set(self, path, value):
		self.setted_values[path] = value

	def getDouble(self, path):
		return self.setted_values.setdefault(path, 0.0)

	def configure_acquisition(self, 
		nodes:List[str], output_values:np.ndarray, time_axis:np.ndarray,
		trigger:Literal[0, 1]=0, prerun_chunks:Union[float, int, None]=2, chunk_size:int=100, chunk_size_jitter:int=10,
		):
		""" This function will configure the Mock to playback some data

		Parameter
		---------
		time_axis
			The time axis of the playback in units of s.

		"""

		assert output_values.shape[0] == len(nodes)
		assert output_values.shape[1] == len(time_axis)
		assert len(time_axis.shape) == 1

		assert all([c in self._lowlevel_node_components for c in nodes]), "some of the nodes that are to be played back are not known"

		# translating the time axis into clockbase units
		time_axis = time_axis*self.getDouble(f"/{self.serial}/clockbase")

		# filling in not specified outputs with 0.
		complete_data_field = np.zeros((len(self._lowlevel_node_components), len(time_axis)))
		for i, n in enumerate(self._lowlevel_node_components):
			if n in nodes:
				complete_data_field[i] = output_values[nodes.index(n)]

		# setting the chosen trigger input to high for the data that is to be played back
		assert trigger in [0, 1, None]
		if trigger is not None:
			complete_data_field[self._lowlevel_node_components.index("trigger")] = (0b0101<<trigger)

		# sampling the noise that is to be played back before the trigger event arises
		dt = max(np.min(np.diff(time_axis)), 1)
		noise_points = prerun_chunks*chunk_size
		noise_data = np.random.uniform(0, 1, size=(len(self._lowlevel_node_components), noise_points))
		for n in ['trigger', 'dio']:
			noise_data[self._lowlevel_node_components.index(n)] = 0
		noise_timeaxis = np.arange(noise_points)*dt

		# and generating again some noise s after the pulse
		post_noise_data = np.random.uniform(0, 1, size=(len(self._lowlevel_node_components), noise_points))
		post_noise_timeaxis = np.arange(noise_points)*dt

		# setting the digital dimensions to 0 for the fields containing noise
		for nd in [noise_data, post_noise_data]:
			for n in ['trigger', 'dio']:
				nd[self._lowlevel_node_components.index(n)] = 0
		
		# combining the data before the trigger signal arrives with the requested data
		shifted_time_axis = time_axis+noise_timeaxis[-1]+dt
		complete_time_axis = np.concatenate([noise_timeaxis, shifted_time_axis, post_noise_timeaxis+shifted_time_axis[-1]+dt], axis=-1)
		combined_data = np.concatenate([noise_data, complete_data_field, post_noise_data], axis=-1)

		self._loaded_readout_values = {n:d for n, d in zip(self._lowlevel_node_components, combined_data)}
		self._loaded_readout_time_axis = complete_time_axis
		self._loaded_trigger = trigger
		self._loaded_readout_chunk_size = chunk_size
		self._loaded_readout_chunk_size_jitter = chunk_size_jitter
		self._playback_state = 'armed'
		self._playback_position = 0

	def poll(self, recording_time_s:float, timeout_ms:float, flags:int, flat:bool):

		# checks if the last call to this function did not occur more than 8 seconds ago.
		time_of_call = time.time()
		if self._last_oll_time is not None:
			assert time_of_call-self._last_oll_time < 8, "The last poll commands were at least 8 seconds apart. This is too long. After 8 seconds, data will be dumped by the zhinst software."
		self._last_oll_time = time_of_call

		# checks if the input parameters are in reasonable ranges.
		assert recording_time_s >= 20e-3, "recording times below 20ms have shown to result in unstable behavior"
		assert timeout_ms < 8e3, "timeouts longer than 8s might result in data loss"
		assert flags == 0, "other flag configurations had not been intended previously"
		assert flat == True, "other flag configurations had not been intended previously"

		# checking if we have data to replay
		if self._playback_state == 'completed':
			return {}

		# selecting the next slice to play back
		next_chunk_size = max(1, int(self._loaded_readout_chunk_size + np.random.normal(0, self._loaded_readout_chunk_size_jitter)))
		head = self._playback_position
		section_end = min(head+next_chunk_size, len(self._loaded_readout_time_axis)+1)
		print(f"{(head, section_end)=}")
		selected_slice = slice(head, section_end)
		self._playback_position = section_end

		# package data
		base_node_name = f"/{self.serial}/demods/0/sample".lower()
		chunks = {base_node_name:{
			n:self._loaded_readout_values[n][selected_slice]
			for n in self._lowlevel_node_components
		}}
		chunks[base_node_name]["timestamp"] = self._loaded_readout_time_axis[selected_slice]
		chunks[base_node_name]["trigger"] = chunks[base_node_name]["trigger"].astype(int)
		chunks[base_node_name]["dio"] = chunks[base_node_name]["dio"].astype(int)

		if self._playback_position > len(self._loaded_readout_time_axis):
			self._playback_state = 'completed'

		if self._playback_state == 'completed':
			self._last_oll_time = None

		if self._playback_artificial_wait is not None:
			time.sleep(self._playback_artificial_wait)

		return chunks











