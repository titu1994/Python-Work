import json
from tqdm import tqdm
from dataclasses import dataclass, field
import numpy as np
from pprint import pprint

import matplotlib.pyplot as plt

with open('dataset.json', 'r') as f:
    dataset = json.load(f)


@dataclass(order=True, repr=False)
class MusicAttributes:
    sort_index: int = field(init=False)
    title: str
    artist: str
    composer: str
    album: str
    albumArtist: str
    year: int
    durationMillis: int
    playCount: int
    totalPlaytime: int = field(init=False)

    def __post_init__(self):
        self.totalPlaytime = self.durationMillis * self.playCount
        self.sort_index = self.totalPlaytime

    def total_playtime(self, return_str=True):
        duration = self.durationMillis * self.playCount // 1000
        seconds = round(duration) % 60
        minutes = round(duration // 60) % 60
        hours = round(duration // 60 // 60)

        if return_str:
            duration_str = f"Total Playtime ={hours:4d}h:{minutes:1d}m:{seconds:2d}s"
            return duration_str
        else:
            return duration

    def __str__(self):
        duration = self.durationMillis / 1000.
        seconds = round(duration) % 60
        minutes = round(duration // 60)
        duration_str = f"{minutes:1d}m:{seconds:2d}s"

        result = f"[ {self.title} ] (Count: {self.playCount}) - Duration = {duration_str} --- " \
                 f"Artist = '{self.artist}' Album = '{self.album}'"

        return result


def parse_record(record: dict) -> MusicAttributes:
    attribute = MusicAttributes(
        title=record['title'],
        artist=record['artist'],
        composer=record.get('composer', ''),
        album=record['album'],
        albumArtist=record['albumArtist'],
        year=record.get('year', 0),
        durationMillis=int(record['durationMillis']),
        playCount=record.get('playCount', 0),
    )

    return attribute


records = []
for record in tqdm(dataset, total=len(dataset)):
    result = parse_record(record)

    if result is not None:
        records.append(result)


records = sorted(records, reverse=True)  # type: list(MusicAttributes)

# for ix, record in enumerate(records[:50]):
#     print(ix + 1, record, "|", record.total_playtime())

total_durations = [record.total_playtime(return_str=False) for record in records]

plt.plot(total_durations[:100])
plt.xlabel('Song id')
plt.ylabel('Total playtime in seconds')
plt.show()


