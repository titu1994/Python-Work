from GooglePlayMusic.DatabaseManager import GPMDBManager
import pandas as pd

if __name__ == "__main__":
    db = GPMDBManager()

    all_songs = """SELECT * FROM Songs WHERE (songPlayCount > 0) ORDER BY songPlayCount DESC"""
    all_songs_from_ragnarok = """SELECT * FROM Songs WHERE (songPlayCount > 0) AND (songAlbum = 'Ragnarok Online BGM') ORDER BY songPlayCount DESC"""

    top_k = 20
    top_k_best_songs = """SELECT * FROM Songs WHERE (songPlayCount > 0) ORDER BY songPlayCount DESC LIMIT """ + str(top_k)

    all_tsubasa_songs = """SELECT * FROM Songs WHERE (songPlayCount > 0) AND (songAlbum = 'Tsubasa Chronicles 2' OR songAlbum = 'Tsubasa Chronicles') ORDER BY songPlayCount DESC"""
    best_taku_iwasaki = """SELECT * FROM Songs WHERE (songPlayCount > 0) AND (songArtist = 'Taku Iwasaki') ORDER BY songPlayCount DESC"""

    group_by_artist = """SELECT songArtist, Count(songArtist) FROM Songs GROUP BY songArtist ORDER BY Count(songArtist) DESC"""

    """
    select = all_songs
    cursor = db.conn.cursor()
    bestSongs = cursor.execute(select)

    print("Song Name, Album, Artist, Song Duration in Milliseconds, Play Count\n")
    for song in bestSongs:
        print(*[song[0], song[1], song[2], song[4]], sep=", ")
    """

    """
    select = all_songs_from_ragnarok
    cursor = db.conn.cursor()
    bestSongs = cursor.execute(select)

    print("Song Name, Album, Artist, Song Duration in Milliseconds, Play Count\n")
    for song in bestSongs:
        print(*[song[0], song[1], song[2], song[4]], sep=", ")
    """

    """
    select = top_k_best_songs
    cursor = db.conn.cursor()
    bestSongs = cursor.execute(select)

    print("Song Name, Album, Artist, Song Duration in Milliseconds, Play Count\n")
    for song in bestSongs:
        print(*[song[0], song[1], song[2], song[4]], sep=", ")
    """

    """
    select = all_tsubasa_songs
    cursor = db.conn.cursor()
    bestSongs = cursor.execute(select)

    print("Song Name, Album, Artist, Song Duration in Milliseconds, Play Count\n")
    for song in bestSongs:
        print(*[song[0], song[1], song[2], song[4]], sep=", ")
    """

    """
    select = best_taku_iwasaki
    cursor = db.conn.cursor()
    bestSongs = cursor.execute(select)

    print("Song Name, Album, Artist, Song Duration in Milliseconds, Play Count\n")
    for song in bestSongs:
        print(*[song[0], song[1], song[2], song[4]], sep=", ")
    """

    select = group_by_artist
    cursor = db.conn.cursor()
    bestSongs = cursor.execute(select)

    print("Artist, No of Songs by Artist\n")
    for song in bestSongs:
        print(*[song[0], song[1]], sep=", ")
