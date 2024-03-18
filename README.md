# MFLI Driver for qupulse

This is an early version of a Zurich Instruments MFLI driver for qupulse.

## Installation

```console
git clone https://github.com/qutech/qupulse-mfli
pip install -e ./qupulse-mfli
```

for a development install or

```console
pip install qupulse-mfli
```

for the published version.

## Usage

### Connecting a MFLI

```python
from qupulse_mfli.mfli import MFLIDAQ
my_lockin = MFLIDAQ.connect_to(device_serial="dev1234")
```

### Registering Measurement Channels

The driver needs to be made aware of which MFLI channels are to be recorded for which measurement window names. 

```python

from qupulse.hardware.setup import HardwareSetup, MeasurementMask

my_lockin_channels = {
    "R": ["demods/0/sample.R"],
    "X": ["demods/0/sample.X"],
    "Y": ["demods/0/sample.Y"],
    "A": ["auxins/0/sample.AuxIn0.avg"],
    "many": ["demods/0/sample.R", "auxins/0/sample.AuxIn0.avg", "demods/0/sample.X", "demods/0/sample.Y"]
}

hardware_setup = HardwareSetup()

for k, v in my_lockin_channels.items():
    # telling the driver which channels to record.
    my_lockin.register_measurement_channel(program_name=None, channel_path=v, window_name=k)
    # and notify qupulse to which device the channels relates
    hardware_setup.set_measurement(f"{k}", MeasurementMask(my_lockin, k))
    
```

### Configuring Post-processing

The recorded data is sliced to the measurement windows in the default configuration. Thus ```my_lockin.measure_program``` returns a list (number of measurements) of dicts (the qupulse channels), of dicts (the lockin channels), of lists (the observed trigger), of lists of xarray DataArrays (each DataArray containing the data sliced for one window) or numpy arrays (containing the data resulting from averaging over the windows). I.e. ```returned_data[<i_measurement>][<qupulse channel>][<lockin channel>][<i_triggerevent>]``` leads to ether the list of DataArrays or to a numpy array.
The post processing function that creates the xarary or numpy arrays can be chosen by registering the following post processing function. 

One can average all data points within one window with the following example:
Here ```None``` can be replaced by ```"my_program"```, which is the name of the qupulse program for which the post processing function is to be registered. ```None``` refers to the default.
```python
from qupulse_mfli.mfli import postprocessing_average_within_windows
my_lockin.register_operations(None, postprocessing_average_within_windows)
```

Or register the post processing function that is used as the defaults default: not averaging over points, just cropping them.
```python
from qupulse_mfli.mfli import postprocessing_crop_windows
my_lockin.register_operations(None, postprocessing_crop_windows)
```

### Trigger Settings

Overwriting the default trigger settings:
```python
my_lockin.register_trigger_settings(
    program_name=None,
    trigger_input=f"demods/0/sample.TrigIn1", # here TrigInN referrers to the printer label N
    edge="rising",
    trigger_count=1, # one trigger event per measurement
    level=.1,
    measurement_count=1, # one measurement per arm
    other_settings={"holdoff/time": 1e-3}
    )
```

### Retrieving Data

```python
data = my_lockin.measure_program(wait=False, return_raw=False)
```

### Uploading, Arming, Force Triggering, and Stopping

```python
# when registering a pulse, pulse is automatically "uploaded":
hardware_setup.register_program("my_programs_name", my_qupulse_program)
# the MFLI is configured and armed when calling the run_program method:
hardware_setup.run_program("my_programs_name")
# a trigger event can be forced via:
my_lockin.force_trigger()
# and the acquisition can be halted via: 
my_lockin.stop_acquisition()
```

## License

`qupulse-mfli` is distributed under the terms of the [GPLv3](https://spdx.org/licenses/GPL-3.0.html) license.

## Build & Publish

This project uses `hatch`. After you installed hatch `python -m pip install hatch`.You can push the version, build and publish (if you have the credentials) it via
```console
python -m hatch version M.N.L
python -m hatch build
python -m hatch publish
```
