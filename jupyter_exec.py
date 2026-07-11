#!/usr/bin/env python3
"""Execute Python code or shell command on remote Jupyter kernel via websocket."""

import json
import sys
import time
from websocket import create_connection

HOST = "radeon-global.anruicloud.com"
TOKEN = "amd-oneclick"
KERNEL_ID = "4933c541-ed9b-4efa-9fdf-9d0a2c4438e7"
INSTANCE = "hf-103-403fafc6"

def execute_py(code, timeout_sec=120):
    ws_url = f"wss://{HOST}/instances/{INSTANCE}/api/kernels/{KERNEL_ID}/channels?token={TOKEN}"
    ws = create_connection(ws_url, timeout=30)
    
    msg_id = "exec-1"
    execute_msg = {
        "header": {
            "msg_id": msg_id,
            "msg_type": "execute_request",
            "username": "",
            "session": "session-1",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": "5.3"
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True
        },
        "buffers": [],
        "channel": "shell"
    }
    
    ws.send(json.dumps(execute_msg))
    outputs = []
    deadline = time.time() + timeout_sec
    done = False
    
    while time.time() < deadline and not done:
        ws.settimeout(5)
        try:
            raw = ws.recv()
            msg = json.loads(raw)
            mt = msg.get("header", {}).get("msg_type", "")
            c = msg.get("content", {})
            
            if mt == "stream":
                outputs.append(("stdout", c.get("text", "")))
            elif mt == "error":
                tb = "\n".join(c.get("traceback", [c.get("evalue", "")]))
                outputs.append(("stderr", tb))
            elif mt == "execute_result":
                outputs.append(("result", c.get("data", {}).get("text/plain", "")))
            elif mt == "execute_reply":
                outputs.append(("status", c.get("status", "")))
                done = True
        except Exception:
            pass
    
    ws.close()
    return outputs

if __name__ == "__main__":
    code = sys.stdin.read()
    if not code:
        print("Usage: echo 'print(1+1)' | python3 jupyter_exec.py", file=sys.stderr)
        sys.exit(1)
    results = execute_py(code, timeout_sec=int(sys.argv[1]) if len(sys.argv) > 1 else 120)
    for kind, data in results:
        if kind == "stdout" and data.strip():
            print(data)
        elif kind == "stderr" and data.strip():
            print(data, file=sys.stderr)
        elif kind == "status":
            pass
