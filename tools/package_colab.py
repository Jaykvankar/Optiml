"""Package only runnable source for Colab upload; never include raw data/secrets."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    out = ROOT / 'dist' / 'optiml_colab_source.zip'
    out.parent.mkdir(exist_ok=True)
    files = [ROOT / name for name in ['README.md', 'requirements.txt', 'requirements-catboost.txt']]
    files.extend(sorted((ROOT / 'optiml').glob('*.py')))
    files.extend(sorted((ROOT / 'tests').glob('*.py')))
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(out)

if __name__ == '__main__':
    main()
