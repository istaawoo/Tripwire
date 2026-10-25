"""Bundle the code + data into tripwire_code.zip for uploading to Colab.

    python make_colab_zip.py
"""
import zipfile
from pathlib import Path

INCLUDE = ["common", "monitor", "eval", "generate", "data", "requirements.txt", ".env.example"]
SKIP = {"__pycache__", ".ipynb_checkpoints"}

with zipfile.ZipFile("tripwire_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for item in INCLUDE:
        p = Path(item)
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        for f in files:
            if not SKIP & set(f.parts):
                z.write(f, f.as_posix())
    z.writestr("results/.gitkeep", "")
print("wrote tripwire_code.zip")
