from GooglePlayMusic.DatabaseManager import GPMDBManager

if __name__ == "__main__":
    db = GPMDBManager()

    select = """SELECT * FROM Songs WHERE (songPlayCount > 0) ORDER BY songPlayCount DESC"""
    cursor = db.conn.cursor()
    bestSongs = cursor.execute(select)

    print("Song Name, Album, Artist, Song Duration in Milliseconds, Play Count")
    for song in bestSongs:
        print(song)


