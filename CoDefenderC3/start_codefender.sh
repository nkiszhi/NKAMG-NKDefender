#!/bin/bash
# ── BinaryNinja 环境 ──────────────────────────────────────────
export PATH="/opt/binaryninja:$PATH"
export LD_LIBRARY_PATH="/opt/binaryninja:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/binaryninja/python:$PYTHONPATH"

# ── License 激活 ─────────────────────────────────────────────
# python3 -c "
# import binaryninja as bn, json, os
# lic = os.path.expanduser('~/.binaryninja/license.dat')
# if os.path.exists(lic):
#     with open(lic) as f:
#         bn.core_set_license(json.load(f)[0])
#     print('[startup] BinaryNinja license activated')
# " 2>&1
python3 -c "
import binaryninja as bn, os
lic = os.path.expanduser('~/.binaryninja/license.dat')
if os.path.exists(lic):
    with open(lic) as f:
        raw = f.read()            # 读成字符串，不 json.load
    bn.core_set_license(raw)      # 传字符串，匹配底层 BNSetLicense
    print('[startup] BinaryNinja license activated')
" 2>&1


# ── 启动 uvicorn ─────────────────────────────────────────────
exec python3 -m uvicorn engine.app:app --host 0.0.0.0 --port 8000
