"""
Convert SDSS Stripe 82 light curves to images
as .npy files to be readable by neural network
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import resize

# Create full_train folder
ROOT_FOLDER = './full_train/'
if not os.path.exists(ROOT_FOLDER):
    os.mkdir(ROOT_FOLDER)

# Path to folder containing raw light curves
# Source: http://faculty.washington.edu/ivezic/macleod/qso_dr7/Southern.html (Light curve files)
PATH = './light_curves/stripe82/'
FILES = os.listdir(PATH)

# Read light curve files
for file_name in FILES:

    with open(PATH + file_name, 'r') as file:
        next(file)  # skip first row
        df = pd.DataFrame(l.rstrip().split() for l in file)

    # Convert str into numeric
    for i in range(14):
        df[i] = pd.to_numeric(df[i], errors='ignore')

    # Clean up anomalies for different pass-band
    df_u = df[df[1] > 1]
    df_g = df[df[4] > 1]
    df_r = df[df[7] > 1]
    df_i = df[df[10] > 1]
    df_z = df[df[13] > 1]

    # You may call plot_light_curve() here for a sanity check
    # plot_light_curve()

    # Gather the 5 band data and set mjd init = 0
    u_data = round(df_u[0]) - round(df_u[0].iloc[0])
    g_data = round(df_g[3]) - round(df_g[3].iloc[0])
    r_data = round(df_g[6]) - round(df_g[6].iloc[0])
    i_data = round(df_g[9]) - round(df_g[9].iloc[0])
    z_data = round(df_g[12]) - round(df_g[12].iloc[0])

    # Convert 5 band data into 5 (bands) x 3340 (days) image data
    Images = np.zeros((5, 3340))

    for i, day in enumerate(u_data):
        Images[0, int(day)] += df_u[1].iloc[i]

    for i, day in enumerate(g_data):
        Images[1, int(day)] += df_g[4].iloc[i]

    for i, day in enumerate(r_data):
        Images[2, int(day)] += df_r[7].iloc[i]

    for i, day in enumerate(i_data):
        Images[3, int(day)] += df_i[10].iloc[i]

    for i, day in enumerate(z_data):
        Images[4, int(day)] += df_z[13].iloc[i]

    # Reshape image into 167 x 100
    reshape_img = Images.reshape(167, 100)

    # Option 1: resize images to 224 x 224 (same as ImageNet)
    resize_img = reshape_img.copy()
    resize_img = resize(resize_img, (224, 224), anti_aliasing=True)

    # Save image as .npy format
    np.save(ROOT_FOLDER + 'lc_image_{}.npy'.format(file_name), resize_img)

    # You may call png_light_curve() here for a sanity check
    # png_light_curve()

    # Option 2: padding zeros to make the 167 x 100 image -> 224 x 224
    Padding_images = np.zeros((224, 224))
    Padding_images[:167, :100] = reshape_img


def plot_light_curve():
    """
    Helper function to plot the current light curve
    """
    plt.plot(df_u[0], df_u[1], label='u', ls='', marker='*')
    plt.plot(df_g[3], df_g[4], label='g', ls='', marker='*')
    plt.plot(df_r[6], df_r[7], label='r', ls='', marker='*')
    plt.plot(df_i[9], df_i[10], label='i', ls='', marker='*')
    plt.plot(df_z[12], df_z[13], label='z', ls='', marker='*')
    plt.title(file_name)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()


def png_light_curve():
    """
    Save image as .png file with 0-255 format
    """
    final_image = np.asarray((resize_img / 40) * 255., dtype=np.int32)
    plt.plot(final_image)
    plt.savefig(ROOT_FOLDER + 'lc_image_{}.png'.format(file_name))
