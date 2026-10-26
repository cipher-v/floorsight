#!/bin/bash
# Read-only probe: bootstrap progress, COOL layout, OpenCV bindings, CPU.
echo "== uptime"; uptime; nproc; lscpu | grep -E "Model name|Flags" | cut -c1-200
echo "== bootstrap tail"; tail -5 /var/log/floorsight-bootstrap.log 2>/dev/null; tail -3 /var/log/cloud-init-output.log
echo "== /opt/cool"; ls /opt/cool 2>&1; ls /opt/cool/* 2>&1 | head -40
echo "== cool readme"; ls /opt/cool/*.md /opt/cool/*/README* 2>/dev/null | head; find /opt/cool -maxdepth 3 -name "cv2*" 2>/dev/null | head
echo "== pythons"; ls /usr/bin/python3* ; python3 --version
for d in /opt/cool/venvs/*; do [ -x "$d/bin/python" ] && echo "venv $d" && "$d/bin/python" -c "import cv2,sys;print(sys.version.split()[0], cv2.__version__, cv2.__file__)" 2>&1 | tail -1; done
echo "== motd"; cat /etc/motd 2>/dev/null | head -30; ls /etc/update-motd.d/ | head
