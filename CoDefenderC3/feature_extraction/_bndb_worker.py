# -*- coding: utf-8 -*-
import sys, os, gc
# Windows: 禁止 crash 弹出 WER 对话框
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0002 | 0x8000)
    except Exception:
        pass
try:
    import binaryninja as bn
    try: bn.log.log_to_stderr(bn.LogLevel.ErrorLog)
    except: pass
    try: bn.disable_default_log()
    except: pass
except ImportError:
    print("FAIL\tNO_FILE\tno_binaryninja", flush=True)
    sys.exit(0)

# 从参数文件读取: pe_path\tbndb_path 每行一对
with open(sys.argv[1], "r", encoding="utf-8") as f:
    lines = f.read().strip().split("\n")
for line in lines:
    parts = line.strip().split("\t")
    if len(parts) != 2:
        continue
    pe_path, bndb_path = parts
    ok = False
    try:
        bv = bn.load(pe_path, options={
            "analysis.mode": "basic",
            "analysis.linearSweep.autorun": True,
            "analysis.limits.maxFunctionSize": 262144,
        })
        if bv is not None:
            bv.update_analysis_and_wait()
            bv.create_database(bndb_path)
            bv.file.close()
            del bv
            ok = True
            print(f"OK\t{pe_path}", flush=True)
        else:
            print(f"FAIL\t{pe_path}\tload_returned_None", flush=True)
    except Exception as e:
        print(f"FAIL\t{pe_path}\t{str(e)[:120]}", flush=True)
    # 失败时删除残留的 bndb 文件
    if not ok and os.path.exists(bndb_path):
        try: os.remove(bndb_path)
        except: pass
    gc.collect()
