""" This file contains tests to test the polling thread and its averaging capabilities
"""

import numpy as np
import threading

from qupulse_mfli.mfli import polling_averaging_thread, MFLIPOLL
from mock_api import MockMFLI


def test_all_ones():
	""" this test if the MockMFLI class and the polling_averaging_thread can be used to obtain some data
	"""

	# creating the mock class into which we will add the data to play back later.
	mfli = MockMFLI("DEV12345")

	# preparing a pulse to play back
	data = np.ones(500)
	time_axis = np.linspace(0, 10, len(data)) # this is in units of s

	# arming the mfli playback
	mfli.configure_acquisition(
		nodes=["x"], output_values=data[None, :], time_axis=time_axis,
		trigger=0, prerun_duration=0.1, chunk_size=100, chunk_size_jitter=10,
		)

	# using the polling and averaging code, which is normally threaded, to retrieve the data and average it. 
	# Note that this is not running in its own thread here!
	running_flag = threading.Event()
	stop_flag = threading.Event()
	output_array = polling_averaging_thread(
		api_session=mfli, serial=mfli.serial, channel_mapping={"WG1": set("x")}, trigger=0,
		windows={"WG1": np.array([[0, time_axis[-1]]]).T*1e3}, output_array={"WG1": np.ones((1, 1))*np.nan}, 
		running_flag=running_flag, stop_flag=stop_flag, timeout=np.inf)

	print(output_array)

	assert not running_flag.is_set()
	assert output_array["WG1"].flatten().shape == (1,)
	assert output_array["WG1"].flatten()[0] == 1.0
	assert False



