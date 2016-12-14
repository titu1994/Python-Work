from PIL import Image
import sys
import os
import glob

def process_gif(infile, outdir):
    try:
        im = Image.open(infile)
    except IOError:
        print("Cant load", infile)
        sys.exit(1)

    i = 0
    mypalette = im.getpalette()

    filepath, filename = os.path.split(infile)
    filterame, exts = os.path.splitext(filename)
    print("Processing: " + infile, filterame)

    outpath = os.path.join(filepath, outdir, filename)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    try:
        while 1:
            #im.putpalette(mypalette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            new_im.save(os.path.join(outpath, 'frame_' + str(i) + '.png'))

            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass # end of sequence

def process_gifs(indir, outdir):
    files = glob.glob(os.path.join(indir, '*.gif'))

    for file in files:
        process_gif(file, outdir)


if __name__ == "__main__":
    dir = r""

    process_gifs(dir, outdir='Images')