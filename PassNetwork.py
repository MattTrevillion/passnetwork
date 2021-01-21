# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 22:10:58 2021

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

#Scale Data
df['x']=df['x']*1.2
df['y']=df['y']*0.8
df['endX']=df['endX']*1.2
df['endY']=df['endY']*0.8

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
#passer_avg.index = passer_avg.index.astype(int)

#Between Passer and Recipient
passes_between = df.groupby(['passer', 'recipient']).id.count().reset_index()
passes_between.rename({'id': 'pass_count'}, axis='columns', inplace=True)
passes_between = passes_between.merge(passer_avg, left_on='passer', right_index=True)
passes_between = passes_between.merge(passer_avg, left_on='recipient', right_index=True,
                                      suffixes=['', '_end'])

#Minimum No. of Passes
passes_between = passes_between.loc[(passes_between['pass_count']>4)]

#Make arrows less transparent if they have a higher count, totally optional of course
min_transparency = 0.3
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

#WITHIN .PITCH THERE IS NO NEED TO FLIP X AND Y

#Plot Pitch
fig, ax = plt.subplots()
fig.set_facecolor('#F8F8FF')
fig.patch.set_facecolor('#F8F8FF')
pitch = Pitch(pitch_type='statsbomb', orientation='vertical',
              pitch_color='#F8F8FF', line_color='#132257',
              constrained_layout=True, tight_layout=False,
              linewidth=0.5)
pitch.draw(ax=ax)
b = passer_avg.epv
a = plt.scatter(passer_avg.y, passer_avg.x, s=100,c=b,facecolor='none',lw=1,
                cmap="cool", alpha=1, zorder=2, vmin=0 , vmax=1, marker='h')
#c = plt.scatter(passer_avg.y, passer_avg.x, s=60,c='#F8F8FF',
#                alpha=1, zorder=3, marker='h')
pitch.arrows(passes_between.x, passes_between.y, passes_between.x_end, 
            passes_between.y_end, color=color, ax=ax, zorder=1, width=1.5)
cbar = plt.colorbar(a, orientation="horizontal",shrink=0.3, pad=0,
             ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
cbar.set_label('Expected Possession Value (EPV)', color='#132257', size=6)
cbar.outline.set_edgecolor('#132257')
cbar.ax.xaxis.set_tick_params(color='#132257')
cbar.ax.xaxis.set_tick_params(labelcolor='#132257')
cbar.ax.tick_params(labelsize=5)
#cbar.ax.xaxis.set_label_position('top')
plt.gca().invert_xaxis()
for index, row in passer_avg.iterrows():
    pitch.annotate(row.name, xy=(row.x, row.y), 
                   c='#132257', va='center', ha='center', size=5, ax=ax)
plt.text(79,2,"Positions = Median Location of Successful Passes\nArrows = Pass Direction\nTransparency = Frequency of Combination\nMinimum of 5 Passes ", color='#132257',
               fontsize=5, alpha=0.5, zorder=1)
plt.text(80,122,"Minutes 0-75 (First Substitution)", color='#132257',
               fontsize=5)
plt.text(18,122,"@trevillion_", color='#132257', fontsize=5)
ax.set_title("Tottenham Hotspur PV Pass Network\n1-1 vs. Fulham (H)", 
             fontsize=8, color="#132257", fontweight = 'bold', y=1.01)
#plt.show()

plt.savefig('C:/Users/Matt/Documents/Football Data/matplotlib/Tottenham/FUL(H) 13.01.21/PassNetworkPV.png', 
            dpi=500, bbox_inches="tight",facecolor='#F8F8FF')