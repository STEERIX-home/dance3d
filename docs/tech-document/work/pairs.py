"""Lay page pictures side by side, two per sheet, for looking through a rendered document."""
import sys
from pathlib import Path
from PIL import Image

folder = Path(sys.argv[1])
pages = sorted(folder.glob('p-*.png'))
for old in folder.glob('pair_*.png'):
    old.unlink()
for i in range(0, len(pages), 2):
    group = [Image.open(p).convert('RGB') for p in pages[i:i + 2]]
    w, h = group[0].size
    sheet = Image.new('RGB', (w * 2 + 12, h), (120, 120, 120))
    for k, page in enumerate(group):
        sheet.paste(page, (k * (w + 12), 0))
    sheet.save(folder / f'pair_{i // 2 + 1:02d}.png')
print(len(pages), 'pages')
