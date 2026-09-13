"""Verify the downloaded data asset and stage inputs for arithmetic checks."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('archive',type=Path)
a=p.parse_args()
release=json.loads((ROOT/'DATA_ASSET.json').read_text())
assert hashlib.sha256(a.archive.read_bytes()).hexdigest()==release['sha256'], 'Data asset hash mismatch'
mapping=json.loads((ROOT/'verification/SOURCE_MANIFEST.json').read_text())['files']
with zipfile.ZipFile(a.archive) as z:
    for rel,info in mapping.items():
        safe=PurePosixPath(rel)
        assert not safe.is_absolute() and '..' not in safe.parts
        data=z.read(info['source_member'])
        assert hashlib.sha256(data).hexdigest()==info['sha256'],rel
        dest=ROOT/'verification'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
print(json.dumps({'status':'PASS','staged_files':len(mapping)}))
