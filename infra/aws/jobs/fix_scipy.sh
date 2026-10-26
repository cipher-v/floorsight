#!/bin/bash
# COOL's site-packages ships numpy 2.5.1, which shadows the venv's numpy (PYTHONPATH comes first); scipy 1.15.3 supports
# numpy < 2.5 only. Upgrade scipy in the COOL-based venvs to a release built for numpy 2.5 and show what is imported.
set -uo pipefail
for v in venv-perc venv; do
  sudo -u floorsight /opt/floorsight/$v/bin/pip install -q --upgrade "scipy>=1.16" 2>&1 | tail -2
done
for w in /opt/floorsight/perc /opt/floorsight/py; do
  sudo -u floorsight $w -c "import numpy, scipy, cv2; from scipy.optimize import linear_sum_assignment as l; import numpy as np; print('$w', 'numpy', numpy.__version__, numpy.__file__.split('site-packages')[0][-40:], 'scipy', scipy.__version__, 'cv2', cv2.__version__, l(np.eye(3)))" 2>&1 | tail -2
done
sudo -u floorsight /opt/floorsight/perc-stock -c "import numpy, scipy, cv2; print('stock numpy', numpy.__version__, 'scipy', scipy.__version__, 'cv2', cv2.__version__)"
