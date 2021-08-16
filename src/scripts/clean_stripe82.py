"""
Read SDSS Stripe 82 data and clean it
Generate a .csv file of cleaned data (data in a desirable format)
Source: http://faculty.washington.edu/ivezic/macleod/qso_dr7/Southern.html
"""
import pandas as pd
from astropy.io import ascii

# Reading the raw data file
DATA = ascii.read('./DB_QSO_S82.dat')

# Extracting useful columns from the DATA file
ID = DATA.field('col1')
RA = DATA.field('col2')
DEC = DATA.field('col3')
Z = DATA.field('col7')
BH_MASS = DATA.field('col8')

# Converting the ID to an integer
ID = [int(i) for i in ID]

# Generating columns for the cleaned Stripe 82 data
X_TRAIN = pd.DataFrame(ID, columns=['ID'])
X_TRAIN['ra'] = RA
X_TRAIN['dec'] = DEC
X_TRAIN['z'] = Z
X_TRAIN['BH_mass'] = BH_MASS

# Generate csv file of cleaned Stripe 82 data
X_TRAIN.to_csv('../../data/clean_stripe82.csv')
