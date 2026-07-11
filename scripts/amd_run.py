#!/usr/bin/env python3
"""Run a shell command on AMD Jupyter kernel, print output."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jupyter_exec import execute_py

cmd = sys.argv[1] if len(sys.argv) > 1 else "echo 'no command given'"
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 120

code = f'''import subprocess, json, shlex
cmd = {json.dumps(cmd)}
r = subprocess.run(cmd, capture_output=True, text=True, shell=True, cwd="/workspace/Kernel-Olympics", timeout={timeout})
out = r.stdout[-5000:] if len(r.stdout) > 5000 else r.stdout
err = r.stderr[-2000:] if len(r.stderr) > 2000 else r.stderr
print(json.dumps({{"out": out, "err": err, "rc": r.returncode}}))
'''

results = execute_py(code, timeout + 30)
for kind, data in results:
    if kind == "stdout" and data.strip():
        try:
            d = json.loads(data)
            sys.stdout.write(d.get("out", ""))
            if d.get("err"):
                sys.stderr.write(d["err"])
        except json.JSONDecodeError:
            sys.stdout.write(data)
    elif kind == "stderr" and data.strip():
        sys.stderr.write(data)
