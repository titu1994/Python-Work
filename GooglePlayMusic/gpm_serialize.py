import numpy as np
from gmusicapi import Mobileclient
import json

client = Mobileclient()
# client.perform_oauth()

client.oauth_login(Mobileclient.FROM_MAC_ADDRESS)

songs = client.get_all_songs()

with open('dataset.json', 'w') as f:
    json.dump(songs, f, indent=4)

print("Serialized song data")