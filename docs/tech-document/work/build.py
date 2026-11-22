"""Render the technical document (doc.html) to PDF and page pictures for checking.

    build.py OUT.pdf [PREVIEW_DIR]
"""
import subprocess, sys
from pathlib import Path
import weasyprint

here = Path(__file__).resolve().parent
out = Path(sys.argv[1])
weasyprint.HTML(here / 'doc.html').write_pdf(out)
if len(sys.argv) > 2:
    preview = Path(sys.argv[2])
    preview.mkdir(parents=True, exist_ok=True)
    for old in preview.glob('p-*.png'):
        old.unlink()
    subprocess.run(['pdftoppm', '-r', '80', '-png', str(out), str(preview / 'p')], check=True)
print(out, out.stat().st_size // 1024, 'KB')
