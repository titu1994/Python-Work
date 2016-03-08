import glob
import os
import json
import seaborn as sns
sns.set_style("whitegrid")

basepath = r"D:/Users/Yue/PycharmProjects/Python-Work/Parallel Quick Sort Results"

# Paths
somPath = r"/Som-Data/*.json"
ishaanPath = r"/Ishaan-Data/*.json"
javaPath = r"/Java-Data/*.json"

def analyze(longpath):
    path = basepath + longpath
    files = glob.glob(path)

    counter = 0

    y1 = []
    y2 = []

    for pth in files:
        counter += 1

        with open(pth, "r") as f:
            for line in f:
                js = json.loads(line)
                gainInMs = js["percentageGain"]
                if counter % 2 == 1: y1.append(gainInMs)
                else: y2.append(gainInMs)

        if counter % 2 == 0:
            x = [i for i in range(1, len(y1) + 1)]
            name = os.path.basename(pth).split(".")[0]

            plot = sns.plt.plot(x, y1, "b", x, y2, "y")
            sns.plt.xlabel("Dataset Number")
            sns.plt.ylabel("Percent Gain")

            sns.plt.savefig(name + " Image.png")

            sns.plt.clf()

            y1.clear()
            y2.clear()

def analyze350M():
    path = r"D:\Users\Yue\PycharmProjects\Python-Work\Parallel Quick Sort Results\Som-Data\Result 350000000 Type 1.json"

    counter = 0

    y1 = []

    with open(path, "r") as f:
        for line in f:
            js = json.loads(line)
            gainInMs = js["percentageGain"]
            y1.append(gainInMs)

        x = [i for i in range(1, len(y1) + 1)]
        name = os.path.basename(path).split(".")[0]

        plot = sns.plt.plot(x, y1, "b")
        sns.plt.xlabel("Dataset Number")
        sns.plt.ylabel("Percent Gain")
        sns.plt.ylim([0,400])

        sns.plt.savefig(name + " Image.png")


def analyzeJava(longpath):
    path = basepath + longpath
    files = glob.glob(path)

    counter = 0

    y1 = []
    y2 = []

    for pth in files:
        counter += 1

        with open(pth, "r") as f:
            for line in f:
                js = json.loads(line)
                gainInMs = js["arraySortTotalTime"]
                if counter % 2 == 1: y1.append(gainInMs)
                else: y2.append(gainInMs)

        if counter % 2 == 0:
            x = [i for i in range(1, len(y1) + 1)]
            name = os.path.basename(pth).split(".")[0]

            plot = sns.plt.plot(x, y1, "b", x, y2, "y")
            sns.plt.xlabel("Dataset Number")
            sns.plt.ylabel("Total Sorting Time (in Milliseconds)")
            sns.plt.ylim(0)
            sns.plt.savefig(name + " Image.png")

            sns.plt.clf()

            y1.clear()
            y2.clear()



if __name__ == "__main__":
    #analyze(somPath)
    #analyze(ishaanPath)
    #analyzeJava(javaPath)
    #analyze350M()
    pass