import subprocess
from pathlib import Path

wav_dir = Path('wav/')
wav_files = wav_dir.glob('*.wav')

mp3_dir = Path('mp3/')

if not mp3_dir.exists():
    mp3_dir.mkdir(parents=True, exist_ok=True)

for wf in wav_files:
    name = wf.stem + '.mp3'

    mf = mp3_dir / name

    subprocess.run(['ffmpeg', '-i', str(wf), '-b:a', '320k', str(mf)])