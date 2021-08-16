"""
Match data from both cleaned Strip 82 and DR7 data sets
Cleaned data at clean_stripe82.csv and clean_dr7.csv
Generate a .csv file of matched data
"""
import pandas as pd
from astropy.coordinates import search_around_sky, SkyCoord
from astropy import units as u

# Reading SDSS Stripe 82 clean data
S82 = pd.read_csv('../../data/clean_stripe82.csv')

# Reading SDSS DR7 clean data
DR7 = pd.read_csv('../../data/clean_dr7.csv')

# Rename the ra, dec coordinates from the 2 different data sets
DR7 = DR7.rename(index=str, columns={'ra': 'ra_DR7', 'dec': 'dec_DR7'})
S82 = S82.rename(index=str, columns={'ra': 'ra_S82', 'dec': 'dec_S82'})

# Match data attributes in the 2 data sets using astropy's SkyCoord
COORD1 = SkyCoord(DR7['ra_DR7'], DR7['dec_DR7'], frame='icrs', unit='deg')
COORD2 = SkyCoord(S82['ra_S82'], S82['dec_S82'], frame='icrs', unit='deg')
IDX1, IDX2, OTHER1, OTHER2 = search_around_sky(COORD1, COORD2, seplimit=0.5 * u.arcsec)

# Generating columns for the matched
X_TRAIN = []
for i in range(len(IDX1)):
    result = DR7.iloc[IDX1[i]].append(S82.iloc[IDX2[i]])
    X_TRAIN.append(result)
X_TRAIN = pd.concat(X_TRAIN, axis=1)
X_TRAIN = X_TRAIN.T

# Remove the unnecessary part
X_TRAIN = X_TRAIN.loc[:, ~X_TRAIN.columns.str.contains('^Unnamed')]
X_TRAIN = X_TRAIN.drop(['BH_mass'], axis=1)
X_TRAIN = X_TRAIN.astype({'ID': int})

# Generate csv file of combined data sets
X_TRAIN.to_csv('../../data/clean_full_data.csv')
