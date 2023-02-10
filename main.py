import pandas as pd
import numpy as np

import prophet

df = pd.read_csv(r'data\placer.csv', delimiter=',')
num_rows = len(df)

# rename columns
df['recipe'] = df['RECIPENAME']
df['start'] = pd.to_datetime(df['STARTTIME'])
df['end'] = pd.to_datetime(df['ENDTIME'])
df['moduleno'] = df['MODULENO'].astype('int32')
df['comp'] = df['NUMCOMP'].astype('int32')
df['block'] = df['NUMBLOCKS'].astype('int32')
df['error'] = df['NUMERRORS'].astype('int32')
df.drop(['PCBID', 'RECIPENAME','SIDE','STARTTIME','ENDTIME','MODULENO','NUMCOMP','NUMBLOCKS','NUMERRORS'], axis=1, inplace=True)


df_furnace = pd.read_csv(r'data\furnace_train_set.csv', delimiter=';')
df_furnace["time"] = pd.to_datetime(df_furnace["time"])
df_furnace["energy"].fillna(0, inplace=True)
df_furnace["energy"] = df_furnace["energy"]/max(df_furnace["energy"])
df_furnace["nitrogen"] = df_furnace["nitrogen_Nm3h"]/max(df_furnace["nitrogen_Nm3h"])
df_furnace.drop(['nitrogen_Nm3h'], axis=1, inplace=True)
df_furnace.set_index('time', inplace=True)

energy = np.zeros((num_rows, 1))
for i, row in enumerate(df.start):
    energy[i] = df_furnace.truncate(row).iloc[0].energy
    if i % 1000 == 0:
        print(f"{i/num_rows*100:.1f}%")

df['energy'] = energy
df.to_pickle(r'C:\Users\jastr\Desktop\AI Challenge Days 2023\data\furnace_train_prepped.pkl')
# metrics
#nan_ratio_energy = df.energy.isnull().sum() / df.energy.isnull().count()
#nan_ratio_nitrogen_Nm3h = df.nitrogen_Nm3h.isnull().sum() / df.nitrogen_Nm3h.isnull().count()



df.describe()

