# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:21:04 2019

@author: oscar
"""
import pandas as pd

df = pd.DataFrame({'numbers': [1, 2, 3], 'colors': ['red', 'white', 'blue']})

print(df)
df.drop([0], inplace=True)
df.drop([2], inplace=True)
print(df)