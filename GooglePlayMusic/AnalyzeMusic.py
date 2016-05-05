import pandas as pd
from sklearn.preprocessing import LabelEncoder
from GooglePlayMusic.DatabaseManager import GPMDBManager

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

if __name__ == "__main__":
    import sqlite3 as sql

    df = pd.read_csv("music.csv", header=0)

    df["SongTotal"] = df["songDurationMillis"] * df["songPlayCount"]

    thresholdCount = 25
    df["SongInBest"] = 0
    df.loc[(df.songPlayCount > thresholdCount), "SongInBest"] = 1

    print(df.info(), "\n", df.describe())

    timeInMillis = df.SongTotal.sum()
    print("Total Time listened to songs (in Minutes): ", timeInMillis / 1000 / 60)
    print("Total Time listened to songs (in Hours): ", timeInMillis / 1000 / 60 / 60)

    #con = sql.connect("GPM.db")

    #df.to_sql("MusicTable", con)
    #df.to_hdf(r"D:\Users\Yue\PycharmProjects\Python-Work\GooglePlayMusic\GPM.h5", "MusicData")
