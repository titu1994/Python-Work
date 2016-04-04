import urllib.request as request
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
from multiprocessing import Pool

if __name__ == "__main__":
    danbooru = "https://danbooru.donmai.us"
    base = "https://danbooru.donmai.us/posts?page="
    tags = "&tags=miqo%27te"

    pool = Pool()
    results = []
    filecounter = 1

    for pageno in range(1, 40):
        pageURL = base + str(pageno) + tags

        page = request.urlopen(pageURL)
        bs = BeautifulSoup(page, "html5lib")

        images = bs.find_all("img", {"itemprop" : "thumbnailUrl"})

        for img in images:
            src = img["src"]
            src = src.replace(r"preview/", r"sample/sample-")
            src = danbooru + src

            results.append(pool.apply_async(urlretrieve, [src, "Images/" + str(filecounter) + ".jpg"]))
            filecounter += 1

        print("Page %d downloading images." % (pageno))

    pool.close()
    pool.join()
    print("Finished")





