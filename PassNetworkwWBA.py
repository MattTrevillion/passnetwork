# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:18:04 2021

@author: Matt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
import matplotlib
from matplotlib.colors import to_rgba

#Import EPV
epv = pd.read_csv("EPV_grid.csv", header=None)
epv = np.array(epv)
n_rows, n_cols = epv.shape
print(n_rows, n_cols)
plt.imshow(epv, cmap="inferno")

#Load Data
df = pd.read_csv("EventData.csv")

#By Team
df = df[df["teamId"]==30]

#Better Measure of Time
df["newsecond"]=60*df["minute"]+df["second"]
df.sort_values(by=['newsecond'])

#Passer and Receipient
df['passer'] = df['number']
df['recipient'] = df['passer'].shift(-1)

#Filtering Before First Substitution
subs = df.loc[(df['type/displayName']=="SubstitutionOff")]
subtimes = subs["newsecond"]
firstsub = subtimes.min()
df = df.loc[(df['newsecond']<firstsub)]

#Filter Successful Passes
df = df[(df["type/displayName"]=="Pass") & 
        (df["outcomeType/displayName"]=="Successful")]

#Bin Data
df['x1_bin'] = pd.cut(df['x'], bins=n_cols, labels=False)
df['x2_bin'] = pd.cut(df['endX'], bins=n_cols, labels=False)
df['y1_bin'] = pd.cut(df['y'], bins=n_rows, labels=False)
df['y2_bin'] = pd.cut(df['endY'], bins=n_rows, labels=False)

#Return Bin Values
df['start_zone_value'] = df[['x1_bin', 'y1_bin']].apply(lambda x: epv[x[1]][x[0]], axis=1)
df['end_zone_value'] = df[['x2_bin', 'y2_bin']].apply(lambda x: epv[x[1]][x[0]], axis=1)

#Calculate Difference
df['epv'] = df['end_zone_value'] - df['start_zone_value']

#Remove Bin Columns
df = df[[col for col in df.columns if 'bin' not in col]]

#Median Location
passer_avg = df.groupby('passer').agg({'x': ['median'], 'y': ['median','count'], 
                                       'epv': ['sum']})
passer_avg.columns = ['x', 'y', 'count','epv']

#Between Passer and Recipient
passes_between = df.groupby(['passer', 'recipient']).id.count().reset_index()
passes_between.rename({'id': 'pass_count'}, axis='columns', inplace=True)
passes_between = passes_between.merge(passer_avg, left_on='passer', right_index=True)
passes_between = passes_between.merge(passer_avg, left_on='recipient', right_index=True,
                                      suffixes=['', '_end'])

#Minimum No. of Passes
passes_between = passes_between.loc[(passes_between['pass_count']>4)]

#Make arrows less transparent if they have a higher count, totally optional of course
min_transparency = 0.1
color = np.array(to_rgba('#132257'))
color = np.tile(color, (len(passes_between), 1))
c_transparency = passes_between.pass_count / passes_between.pass_count.max()
c_transparency = (c_transparency * (1 - min_transparency)) + min_transparency
color[:, 3] = c_transparency

#Font
matplotlib.font_manager._rebuild()
plt.rcParams['font.family'] = 'Myriad Pro'
colour = '#132257'
plt.rcParams['text.color'] = colour

#Plot Pitch
fig, ax = plt.subplots()
fig.set_facecolor('#F8F8FF')
fig.patch.set_facecolor('#F8F8FF')
pitch = Pitch(pitch_type='opta', orientation='horizontal',
              pitch_color='#F8F8FF', line_color='#132257',
              constrained_layout=True, tight_layout=False,
              linewidth=0.5)
pitch.draw(ax=ax)
b = passer_avg.epv
a = plt.scatter(passer_avg.x, passer_avg.y, s=120, c=b,facecolor='none',lw=1,
                cmap='cool', alpha=1, zorder=2, vmin=0, vmax=0.4,
                marker='h')
plt.scatter(passer_avg.x, passer_avg.y, s=80, color='#F8F8FF',lw=1,
                alpha=1, zorder=3, marker='h')
pitch.arrows(passes_between.x, passes_between.y, passes_between.x_end, 
            passes_between.y_end, color=color, ax=ax, zorder=1, width=1.5)
legend1 = ax.legend(*a.legend_elements(num=6), frameon=False, ncol=7,
                    markerscale=2, columnspacing=9,
                    fontsize=0, bbox_to_anchor=(0.96,0.085))
ax.add_artist(legend1)
for text in legend1.texts:
    text.set_visible(False)
for index, row in passer_avg.iterrows():
    pitch.annotate(row.name, xy=(row.x, row.y), 
                   c='#132257', va='center', ha='center', size=7, ax=ax)
plt.text(1,2,"Positions = Median Location of Successful Passes\nArrows = Pass Direction\nTransparency = Frequency of Combination\nMinimum of 5 Passes ", color='#132257',
               fontsize=5, alpha=0.5, zorder=1)
plt.text(0,101,"Minutes 0-68 (First Substitution)", color='#132257',
               fontsize=6)
plt.text(90.5,101,"@trevillion_", color='#132257', fontsize=6)
plt.text(94.7,9,"+0.40", color='#132257', fontsize=5)
plt.text(76.7,9,"+0.00", color='#132257', fontsize=5)
plt.text(86.5,9,"EPV", color='#132257', fontsize=5)
plt.text(0,112,"Tottenham Hotspur PV Pass Network", 
             fontsize=12, color="#132257", fontweight = 'bold')
plt.text(0,106,"2-0 vs. West Bromwich Albion (H) | Premier League 2020-21", 
             fontsize=8, color="#132257", fontweight = 'bold')
plt.savefig('C:/Users/Matt/Documents/Football Data/matplotlib/Tottenham/WBA(H) 07.02.21/PassNetwork.png', 
            dpi=500, bbox_inches="tight",facecolor='#F8F8FF')