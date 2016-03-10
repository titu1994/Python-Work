from GooglePlayMusic.DatabaseManager import GPMDBManager

if __name__ == "__main__":
    db = GPMDBManager()

    select = """SELECT * FROM Songs WHERE (songPlayCount > 0) ORDER BY songPlayCount DESC"""
    cursor = db.conn.cursor()
    bestSongs = cursor.execute(select)

    print("Song Name, Album, Artist, Song Duration in Milliseconds, Play Count\n")
    for song in bestSongs:
        print(*[song[0], song[1], song[2], song[4]],sep=", ")