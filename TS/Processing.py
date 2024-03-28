import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from trend_classifier import Segmenter
import numpy as np
import datetime
from datetime import timedelta
from scipy.ndimage import gaussian_filter1d
import numpy
import pandas
import matplotlib.colors as mcolors

from TS.Metrics import metric4

stocks1 =['AAPL','MSFT']
stocks2='AAPL'

def data_yf(stocks,start_date,end_date):
    data=yf.download(stocks, start=start_date, end=end_date)
    return data


def plot_data_segmented(data,precision):
    seg = Segmenter([i for i in range (len(data.index))],list(data['Close']), n=precision)
    seg.calculate_segments()
    seg.plot_segments()
    return

def plot_data_segmented_several(data,precision,stocks):
    for x in stocks:    
        seg = Segmenter([i for i in range (len(data['Close']))],list(data['Close'][x]), n=precision)
        seg.calculate_segments()
        seg.plot_segments()
    return


def tab_date(tab_seg,data):   
    j=0
    table=tab_seg
    for i in range(len(data)):
        if i==table['start'][j]:
            table['start'][j]=data.index[i]
        if i==table['stop'][j]-1:
            table['stop'][j]=data.index[i]
            j=j+1
    table=table.reset_index()
    L=[]
    P=[]
    for i in range(len(data.index)):
        if data.index[i] in list(table['start']):
            L.append(data['Close'][i]) 
        if data.index[i] in list(table['stop']):
            P.append(data['Close'][i]) 
    table['Close_start']=L
    table['Close_stop']=P
    return table

# precision= int(len(data['Close'].index)/8)

def metric_date_tab_several(data,stocks,precision):
    L=[]
    tab_list=[]
    for x in stocks:
        T=[]
        P=[]
        #Colonne metric
        seg = Segmenter([i for i in range (len(data['Close'][x].index))],list(data['Close'][x]), n=precision)
        seg.calculate_segments()
        tab=seg.segments.to_dataframe()  
        tab['metric']=[0]*len(tab.index)

        for i in range(len(tab.index)):
            tab['metric'][i]=metric4(tab['start'][i],tab)

        #Changement date en TimeStamp
            
        j=0
        for i in range(len(data['Close'][x])):
            if i==tab['start'][j]:
                tab['start'][j]=data['Close'][x].index[i]
            if i==tab['stop'][j]- 1:
                tab['stop'][j]=data['Close'][x].index[i]
                j=j+1

        tab=tab.reset_index()

        #Ajout Close_start et Close_end

        for i in range(len(data.index)):
            if data.index[i] in list(tab['start']):
                T.append(data['Close'][x][i]) 
            if data.index[i] in list(tab['stop']):
                P.append(data['Close'][x][i])

        tab['Close_start']=T
        tab['Close_stop']=P
        tab=tab.reset_index()
        tab_list.append(tab)
        L.append(tab)

        
    return L

def metric_date_tab(data,precision):
    seg = Segmenter([i for i in range (len(data['Close'].index))],data['Close'], n=precision)
    seg.calculate_segments()
    tab=seg.segments.to_dataframe()
    tab['metric']=[0]*len(tab.index)
    for i in range(len(tab.index)):
        tab['metric'][i]=metric4(tab['start'][i],tab)
    return tab_date(tab,data)


def plot_segm(data,precision):
    table=metric_date_tab(data,precision)
    for i in range (len (table)):
        if table['slope'][i]>=0.04:
            start=table['start'][i]
            stop=table['stop'][i]
            close_start=table['Close_start'][i]
            close_stop=table['Close_stop'][i]
            plt.plot([start,stop],[close_start,close_stop],'g--', linewidth=3)
        elif table['slope'][i]<=-0.04:
            start=table['start'][i]
            stop=table['stop'][i]
            close_start=table['Close_start'][i]
            close_stop=table['Close_stop'][i]
            plt.plot([start,stop],[close_start,close_stop],'r--', linewidth=3)
        elif (table['slope'][i]<0.04) and (table['slope'][i]>-0.04):
            start=table['start'][i]
            stop=table['stop'][i]
            close_start=table['Close_start'][i]
            close_stop=table['Close_stop'][i]
            plt.plot([start,stop],[close_start,close_stop],'m--', linewidth=3)
    return

def plot_graph(data,stocks):
    plt.plot(data.index,data['Close'], color="#C0C0C0",label=stocks[0])
    plt.xlabel('Date')
    plt.ylabel('Prix action')
    plt.gcf().autofmt_xdate()
    plt.legend()
    return

def plot_segm_several(data,stocks,precision):
    k=0
    palette=[]
    number = len(stocks)
    cmap = plt.get_cmap('twilight')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    for i in range(number):
        palette.append(mcolors.rgb2hex(colors[i]))
    for x in stocks:
        table=metric_date_tab_several(data,stocks,precision)[k]
        for i in range (len (table)):
            if table['slope'][i]>=0.04:
                start=table['start'][i]
                stop=table['stop'][i]
                close_start=table['Close_start'][i]
                close_stop=table['Close_stop'][i]
                plt.plot([start,stop],[close_start,close_stop],'g--', linewidth=3)
            elif table['slope'][i]<=-0.04:
                start=table['start'][i]
                stop=table['stop'][i]
                close_start=table['Close_start'][i]
                close_stop=table['Close_stop'][i]
                plt.plot([start,stop],[close_start,close_stop],'r--', linewidth=3)
            elif (table['slope'][i]<0.04) and (table['slope'][i]>-0.04):
                start=table['start'][i]
                stop=table['stop'][i]
                close_start=table['Close_start'][i]
                close_stop=table['Close_stop'][i]
                plt.plot([start,stop],[close_start,close_stop],'m--', linewidth=3)    
        
        plt.plot(data['Close'][x].index,data['Close'][x],label=str(x), color=palette[k])
        plt.xlabel('Date')
        plt.ylabel('Prix action')
        plt.gcf().autofmt_xdate()
        plt.legend()
        k=k+1
    return



def tab_fin_several(data,stocks,precision):
    L=[]
    i=0
    for x in stocks:
        dict={'metric':[]} 
        dict['start']=metric_date_tab_several(data,stocks,precision)[i]['start']
        dict['metric'].append(metric_date_tab_several(data,stocks,precision)[i]['metric'][0])
        for j in range(1,len(dict['start'])):
            dict['metric'].append(metric_date_tab_several(data,stocks,precision)[i]['metric'][j]-metric_date_tab_several(data,stocks,precision)[i]['metric'][j-1])
        df=pd.DataFrame(dict)
        L.append(df)
        i=i+1
    return L

def tab_fin(data,precision):
    dict={'metric':[]} 
    dict['start']=metric_date_tab(data,precision)['start']
    dict['metric'].append(metric_date_tab(data,precision)['metric'][0])
    for i in range(1,len(dict['start'])):
        dict['metric'].append(metric_date_tab(data,precision)['metric'][i]-metric_date_tab(data,precision)['metric'][i-1])
    df=pd.DataFrame(dict)
    return df

def tab_var_sig_several(data,stocks,precision):
    Final_list=[]
    j=0
    for x in stocks:    
        best_score=sorted(list(tab_fin_several(data,stocks,precision)[j]['metric']),reverse=True, key=abs)[:4]
        tab_final=tab_fin_several(data,stocks,precision)[j][abs(tab_fin_several(data,stocks,precision)[j]['metric'])>=abs(best_score[-1])]
        tab_final=tab_final.reset_index()
        L=[]
        for i in range(len(data.index)):
            if data.index[i] in list(tab_final['start']):
                L.append(data['Close'][x][i]) 
        tab_final['Close']=L
        Final_list.append(tab_final)
        j=j+1
    return Final_list

def tab_var_sig(data,precision): 
    Final_list=[]
    best_score=sorted(list(tab_fin(data,precision)['metric']),reverse=True, key=abs)[:4]
    tab_final=tab_fin(data,precision)[abs(tab_fin(data,precision)['metric'])>=abs(best_score[-1])]
    tab_final=tab_final.reset_index()
    L=[]
    for i in range(len(data.index)):
        if data.index[i] in list(tab_final['start']):
            L.append(data['Close'][i]) 
    tab_final['Close']=L
    Final_list.append(tab_final)
    return Final_list

def cren_date(data,precision):
    tab=tab_var_sig(data,precision)[0].reset_index()
    tab['creneau']=[0]*len(tab['start'])
    for i in range(len(tab['start'])):
        date=tab['start'][i]
        L=[]
        for j in range(10):
            L.append((date- datetime.timedelta(10-j)).date())
        for j in range(6):
            L.append((date+datetime.timedelta(j)).date()) 
        tab['creneau'][i]=L
        
    return [tab]

def cren_date_several(data,stocks,precision):
    P=[]
    for k in range(len(stocks)):
        tab=tab_var_sig_several(data,stocks,precision)[k].reset_index()
        tab['creneau']=[0]*len(tab['start'])
        for i in range(len(tab['start'])):
            date=tab['start'][i]
            L=[]
            for j in range(10):
                L.append(date- datetime.timedelta(10-j))
            for j in range(6):
                L.append(date+datetime.timedelta(j)) 
            tab['creneau'][i]=L
        P.append(tab)
    return P
