import urllib.request as request
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
from multiprocessing import Pool
import os

if __name__ == "__main__":
    gelbooru = "http://gelbooru.com/"
    base = "http://gelbooru.com/index.php?page=post&s=list&tags="
    tailer = "&pid="
    pid = 0

    search = input("Enter a search term : ")
    fn = search
    search = search.replace(" ", "+").lower()

    noPages = input("Enter no of pages : ")
    noPages = int(noPages)

    pool = Pool()
    results = []
    filecounter = 1

    if not os.path.exists("Images/" + fn):
        os.makedirs("Images/" + fn + "/")

    for pageno in range(noPages):
        pageURL = base + search + tailer + str(pid)

        page = request.urlopen(pageURL)
        bs = BeautifulSoup(page, "html5lib")

        spans = bs.find_all("span", {"class" : "thumb"})

        for span in spans:
            src = span.find("a")["href"]
            src = gelbooru + src

            imgSite = request.urlopen(src)
            bs2 = BeautifulSoup(imgSite, "html5lib")

            img = bs2.find("img", {"id" : "image"})
            if img is not None:
                imgsrc = img["src"]

                type = ".jpg"
                if ".png" in imgsrc:
                    type = ".png"

                results.append(pool.apply_async(urlretrieve, [imgsrc, "Images/" + fn + "/" + str(filecounter) + type]))
                filecounter += 1

                print("Image %d downloading." % (filecounter))
            else:
                print("%d is not an image file. Skipping." % (filecounter))

        print("Page %d downloading images." % (pageno))
        pid += 42

    pool.close()
    pool.join()
    print("Finished")





