# -*- coding: utf-8 -*-
import sys, os, gc, io, zipfile

# Windows: 禁止 segfault 弹出 '应用程序错误' 对话框
# 没有这个, 每次 BN crash 都弹窗, 阻塞 proc.wait(), 拖垮整个流程
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0002 | 0x8000)
    except Exception:
        pass

sys.path.insert(0, 'D:\\VM_Share\\CoDefenderC3_v8\\CoDefenderC3\\feature_extraction')
import numpy as np
from extract_feature import BinjaExtractor

def append_to_npz(npz_path, arrays):
    with zipfile.ZipFile(npz_path, 'a', zipfile.ZIP_STORED) as zf:
        existing = set(zf.namelist())
        for key, arr in arrays.items():
            fname = key + '.npy'
            if fname in existing:
                continue
            buf = io.BytesIO()
            np.save(buf, arr)
            zf.writestr(fname, buf.getvalue())

ext = BinjaExtractor()
ext._ensure_bn()  # Worker 进程中加载 BN (Master 不加载)
sys.stdout.write('READY\n')
sys.stdout.flush()

n_done = 0
for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line or line == 'EXIT':
        break
    parts = line.split('\t')
    if len(parts) != 3:
        sys.stdout.write('FAIL\t\tinvalid_input\n')
        sys.stdout.flush()
        continue
    pe_path, bndb_path, npz_path = parts
    try:
        bv = ext.bn.load(bndb_path)
        if bv is None:
            sys.stdout.write('FAIL\t' + pe_path + '\tbndb_load_None\n')
            sys.stdout.flush()
            continue
        feats = {}
        try:
            feats.update(ext._extract_disassembly(bv))
        except Exception:
            pass
        try:
            feats.update(ext._extract_graphs(bv))
        except Exception:
            pass
        try:
            bv.file.close()
        except Exception:
            pass
        del bv
        if feats:
            append_to_npz(npz_path, feats)
            del feats
            sys.stdout.write('OK\t' + pe_path + '\n')
        else:
            sys.stdout.write('FAIL\t' + pe_path + '\tempty_feats\n')
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write('FAIL\t' + pe_path + '\t' + str(e)[:120] + '\n')
        sys.stdout.flush()
    n_done += 1
    if n_done % 20 == 0:
        gc.collect()
