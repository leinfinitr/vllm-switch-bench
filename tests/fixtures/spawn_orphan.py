#!/usr/bin/env python3
import os
import subprocess
import sys

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
print(child.pid, flush=True)
os._exit(0)
