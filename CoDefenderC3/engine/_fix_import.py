"""Fix import line and verify all patches."""
import pathlib
import tempfile
import subprocess

p = pathlib.Path(r"d:\VM_Share\CoDefenderC3_v7\engine\app.py")
c = p.read_text(encoding="utf-8")

# Fix the doubled import line
bad = "from models.ensemble import MultiModelEnsemble, _input_dtypefrom models.ensemble import MultiModelEnsemble, _input_dtype"
good = "from models.ensemble import MultiModelEnsemble, _input_dtype\nfrom engine.result_formatter import format_scan_result\nfrom engine.inference import compute_identifiers"

if bad in c:
    c = c.replace(bad, good, 1)
    print("Fixed doubled import line")
else:
    print("Bad pattern not found (already fixed?)")

assert "from engine.result_formatter import format_scan_result" in c, "missing result_formatter import"
assert "from engine.inference import compute_identifiers" in c, "missing compute_identifiers import"
print("Import verification OK")

# Write to temp
tmp = pathlib.Path(tempfile.gettempdir()) / "app_fix.py"
tmp.write_text(c, encoding="utf-8")
print(f"Written to {tmp} ({len(c)} chars)")

# Copy back using PowerShell Copy-Item
r = subprocess.run(
    ["powershell", "-Command", f'Copy-Item -Path "{tmp}" -Destination "{p}" -Force'],
    capture_output=True, text=True,
)
print(f"Copy-Item exit: {r.returncode}")

# Final verify
v = p.read_text(encoding="utf-8")
for tag in [
    "from engine.result_formatter",
    "from engine.inference import compute_identifiers",
    "_SDD_ENGINE: Optional[Any]",
    "def _get_sdd_engine",
    "return format_scan_result(",
    "model_names=model_names",
]:
    print(f"  {tag}: {'OK' if tag in v else 'MISSING'}")
