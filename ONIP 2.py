# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:37:07 2024

@author: routi
"""




z = []
nom = []
with open('data.csv', 'r') as f:
reader = csv.reader(f , delimiter=',')
for row in reader:
z = np.append(z,float(row[0]))