#!/usr/bin/python3
"""Install the official Linux Blender build, verifying the published SHA-256."""
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

root = Path(os.environ['RGBMOCAP_INSTALL']) / 'cache/runtime'
version = '4.5.16'
name = f'blender-{version}-linux-x64.tar.xz'
base = 'https://download.blender.org/release/Blender4.5/'
root.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(base + f'blender-{version}.sha256') as response:
    checksums = response.read().decode()
expected = next(line.split()[0] for line in checksums.splitlines() if name in line)
archive = root / name
if not archive.exists():
    urllib.request.urlretrieve(base + name, archive)
digest = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
if digest != expected:
    raise RuntimeError('Blender SHA-256 mismatch')
print('Blender SHA-256 verified:', digest, flush=True)
with tarfile.open(archive) as source:
    source.extractall(root, filter='data')
print(root / f'blender-{version}-linux-x64/blender')
