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

## License

`qupulse-mfli` is distributed under the terms of the [GPLv3](https://spdx.org/licenses/GPL-3.0.html) license.

## Build & Publish

This project uses `hatch`. After you installed hatch `python -m pip install hatch`.You can push the version, build and publish (if you have the credentials) it via
```console
python -m hatch version M.N.L
python -m hatch build
python -m hatch publish
```
