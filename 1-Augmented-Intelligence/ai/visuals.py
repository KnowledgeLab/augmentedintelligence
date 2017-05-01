import matplotlib.pyplot as plt #For graphics
import seaborn #Makes the graphics look nicer
import numpy as np #for arrays
import pandas #gives us DataFrames

def plotter(df):
    fig, ax = plt.subplots()
    pallet = seaborn.color_palette(palette='rainbow', n_colors= len(set(df['category'])))
    for i, cat in enumerate(set(df['category'])):
        a = np.stack(df[df['category'] == cat]['vect'])
        ax.scatter(a[:,0], a[:, 1], c = pallet[i], label = cat)
    ax.legend(loc = 'center right')
