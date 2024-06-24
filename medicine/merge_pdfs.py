import os
import glob
import argparse

from PyPDF2 import PdfMerger


parser = argparse.ArgumentParser('Python pdf file merger')

parser.add_argument('-f', dest='filepaths', type=str, nargs='+', default=None, required=False, help='List of file paths')
parser.add_argument('-d', dest='directory', type=str, default=None, required=False, help='Directory of file paths')
parser.add_argument('-o', dest='output', type=str, default=None, required=False, help='Output file path')

args = parser.parse_args()

filepaths = args.filepaths
directory = args.directory
output_dir = args.output

if (filepaths is None or len(filepaths) == 0) and (directory is None):
    raise FileNotFoundError("No file provided!")

if filepaths is None:
    if directory[-1] in ("'", '"'):
        directory = directory[:-1]
    filepaths = sorted(list(glob.glob(os.path.join(directory, "*.pdf"))))

if output_dir is None:
    output_dir = os.getcwd()

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

merger = PdfMerger()

for pdf in filepaths:
    merger.append(pdf)


output_dir = os.path.abspath(output_dir)
output_filepath = os.path.join(output_dir, 'Som - Feb 2023 - Reports.pdf')
merger.write(output_filepath)
merger.close()

print(f"Results written to path : {output_filepath}")
