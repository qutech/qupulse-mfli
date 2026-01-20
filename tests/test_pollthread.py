""" This file contains tests to test the polling thread and its averaging capabilities
"""

import numpy as np
import threading

from qupulse_mfli.mfli import polling_averaging_thread, MFLIPOLL
from mock_api import MockMFLI


def test_all_ones():
	""" this test if the MockMFLI class and the polling_averaging_thread can be used to obtain some data
	"""

	# seeding numpy
	np.random.seed(435786)

	# creating the mock class into which we will add the data to play back later.
	mfli = MockMFLI("DEV12345")

	# preparing a pulse to play back
	data = np.ones(500)
	data = np.zeros(500)
	data[-1] = 500.0
	time_axis = np.linspace(0, 10, len(data)) # this is in units of s

	# arming the mfli playback
	mfli.configure_acquisition(nodes=["x"], output_values=data[None, :], time_axis=time_axis)

	# using the polling and averaging code, which is normally threaded, to retrieve the data and average it. 
	# Note that this is not running in its own thread here!
	running_flag = threading.Event()
	stop_flag = threading.Event()
	output_array = polling_averaging_thread(
		api_session=mfli, serial=mfli.serial, 
		channel_mapping={"WG1": set("x")}, trigger=0,
		windows={"WG1": np.array([[0, time_axis[-1]+10/500]]).T*1e9}, # the measurement windows in units of ns
		output_array={"WG1": np.ones((1, 1))*np.nan}, 
		running_flag=running_flag, stop_flag=stop_flag, 
		timeout=np.inf, # timeout after which the acquisitions stops, even if not all samples were measured. This values is given in units of s
		)

	print(output_array)

	assert not running_flag.is_set()
	assert output_array["WG1"].flatten().shape == (1,)
	assert output_array["WG1"].flatten()[0] == 1.0

def test_random_averaging():
	""" this test if the MockMFLI class and the polling_averaging_thread can be used to obtain some data
	"""

	# seeding numpy
	np.random.seed(435786)

	# creating the mock class into which we will add the data to play back later.
	mfli = MockMFLI("DEV12345")

	# preparing a pulse to play back
	n_chunks = 500
	chunks = [np.random.uniform(0, 1, 10) for _ in range(n_chunks)]
	data = np.concatenate(chunks)
	time_axis = np.linspace(0, 1*n_chunks, len(data)) # this is in units of s

	# arming the mfli playback
	mfli.configure_acquisition(nodes=["x"], output_values=data[None, :], time_axis=time_axis)

	# using the polling and averaging code, which is normally threaded, to retrieve the data and average it. 
	# Note that this is not running in its own thread here!
	running_flag = threading.Event()
	stop_flag = threading.Event()
	output_array = polling_averaging_thread(
		api_session=mfli, serial=mfli.serial, 
		channel_mapping={"WG1": set("x")}, trigger=0,
		windows={"WG1": np.array([[i, 1+0.000001] for i in range(n_chunks)]).T*1e9}, # the measurement windows in units of ns
		output_array={"WG1": np.ones((1, n_chunks))*np.nan}, 
		running_flag=running_flag, stop_flag=stop_flag, 
		timeout=np.inf, # timeout after which the acquisitions stops, even if not all samples were measured. This values is given in units of s
		)

	print(output_array)

	assert not running_flag.is_set()
	assert output_array["WG1"].flatten().shape == (n_chunks,)
	assert np.allclose(output_array["WG1"].flatten(), [np.mean(c) for c in chunks])





