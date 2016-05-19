from pyspark import SparkContext

if __name__ == "__main__":
    parts = 8
    sc = SparkContext(master="local[" + str(parts) + "]", appName="Counter")

    text = sc.textFile(r"D:\Users\Yue\PycharmProjects\Python-Work\Spark\Data\readme.md")
    print("No of lines : ", text.count())

    print("First line : ", text.first())

    linesWithSpark = text.filter(lambda line: "Spark" in line)
    print("Lines which contain 'Spark' in them : ", linesWithSpark.count())

    lineWithMostWords = (
        text.map(lambda line: len(line.split()))
            .reduce(lambda a, b: a if a > b else b)
    )

    print("Line no with most no of words : ", lineWithMostWords)

    lineWithLeastWords = (
        text.map(lambda line: len(line.split()))
            .reduce(min)
    )

    print("Line no with least no of words : ", lineWithLeastWords)

    wordCount = (
        text.flatMap(lambda line: line.split(" ")) # returns a list of words for each sentence
            .map(lambda word: (word, 1)) # inits each word with its counter to 1
            .reduceByKey(lambda a, b: a + b) # reduces the value (count) by summing for same words
            .map(lambda x: (x[1], x[0]))
            .sortByKey(False)
    )

    print("Each Word Count : \n")
    wordcounts = wordCount.collect()

    for word, count in wordcounts:
        print(word, " - ", count)
