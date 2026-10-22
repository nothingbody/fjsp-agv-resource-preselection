"""Read-only standard-library check of the downloaded release; no experiments run."""
import argparse
import collections
import hashlib
import json
import tarfile
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--assets', type=Path, help='Optional directory holding the three raw TAR assets')
    a = p.parse_args()
    for row in json.loads((a.root / 'MANIFEST.json').read_text()):
        assert sha(a.root / row['path']) == row['sha256'], row['path']
    count = 0
    if a.assets:
        for archive in sorted(a.assets.glob('full_records_*_v1.4.0.tar')):
            found = collections.defaultdict(list)
            with tarfile.open(archive) as t:
                for m in t:
                    parts = Path(m.name).parts
                    if m.isfile() and len(parts) == 4 and parts[0] == 'out' and parts[2] == 'runs' and parts[-1].endswith('.json.gz'):
                        found[parts[1]].append(parts[-1][:-8])
            for study, ids in found.items():
                tasks = json.loads((a.root / 'studies/tasks' / (study + '.json')).read_text())
                assert len(ids) == len(set(ids)) == len(tasks)
                assert set(ids) == {r['id'] for r in tasks}, study
                count += len(ids)
        assert count == 30624, count
    print(json.dumps({'compact_hashes': 'PASS', 'archived_run_sets': count if a.assets else 'not requested'}))


if __name__ == '__main__':
    main()
