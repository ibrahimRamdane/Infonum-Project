import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from trend_classifier import Segmenter
import numpy as np
from scipy.ndimage import gaussian_filter1d

def start_stop(datetime,data):
    i=0
    stop=data['stop'][i]
    start=data['start'][i]
    while datetime>stop:
            i=i+1
            stop=data['stop'][i]
            start=data['start'][i]
    return start,stop,i

def mid(datetime,dataa):
      return (start_stop(datetime,dataa)[0]+start_stop(datetime,dataa)[1])/2

def max_start_stop(dataa):
    list_time=list(dataa['stop']-dataa['start'])
    max=list_time[0]
    for i in range(1,len(dataa['stop'])):
        if max<list_time[i]:
             max=list_time[i]
    return max

def slope(datetime,dataa):
    i=start_stop(datetime,dataa)[2]
    return dataa['slope'][i]

def metric4(datetime,tabl):
    
    tabl['start'][1:]=tabl['start'][1:]
    tabl['stop'][:-1]=tabl['stop'][:-1]
    met=slope(datetime,tabl)*((start_stop(datetime,tabl)[1]-datetime)**2/(4*((start_stop(datetime,tabl)[1]-start_stop(datetime,tabl)[0]))*max_start_stop(tabl)))
    return met