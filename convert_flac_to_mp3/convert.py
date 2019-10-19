import os
import glob
from subprocess import call

flac_files = glob.glob("*.flac")

for fp in flac_files:
    new_name = os.path.splitext(fp)[0] + ".mp3"

    call(["ffmpeg", "-y", "-i", fp, "-q:a", "0",new_name])
    print("Converted file : ", new_name)

print("Finished converting all files !")