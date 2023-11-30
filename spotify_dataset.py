"""

File for loading in Spotify Million Songs Dataset

"""

import os
import pandas as pd
import numpy as np

class SpotifyDataset:

    def __init__(self, file_path):
        """Read in the necessary data from the specified filepath"""

        self.df = pd.read_csv(file_path)
    
    def __len__(self):
        """Returns size of the dataset"""

        return len(self.df)
    
    def get_lyrics(self, index):
        """Returns the lyrics at a specific index"""

        return self.df.loc[index, 'text']
    
    def get_title(self, index):
        """Returns the title at a specific index"""

        return self.df.loc[index, 'song']
    
    def get_index_from_title(self, title):
        """Returns the index corresponding to a given title"""

        index = self.df.index[self.df['song'] == title].tolist()

        if not index:
            print(f"Title '{title}' not found in the dataset.")
            return None

        return index[0]