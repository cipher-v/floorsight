#!/bin/bash
sudo -u floorsight /opt/floorsight/venv-perc/bin/pip install -q "matplotlib>=3.10" 2>&1 | tail -2
sudo -u floorsight /opt/floorsight/perc -c "import matplotlib; print('matplotlib', matplotlib.__version__)"
