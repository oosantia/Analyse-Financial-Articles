# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:17:05 2015

@author: Omar
"""

f = open('newconbineddata.txt', 'r',)
w = open('data3.txt', 'w' , encoding="utf-8")
for line in f:
    print(line)
    if line != '\n':
        w.write(line)
f.close()
w.close()