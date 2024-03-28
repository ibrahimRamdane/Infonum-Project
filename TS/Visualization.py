from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
from trend_classifier import Segmenter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime

from TS.Metrics import metric4
from TS.Processing import *

# stocks= input('list of companies : ')
# precision= input('précision segment : ')
# data=data_yf(stocks)

# dict={}
# L=[]
# for i in range(len(data[2].index)):
#     L.append(metric4(i,data[2]))

# dict['metric']=L
# dict['centered_metric']=L-np.mean(dict['metric'])
# plt.plot(L)
# plt.plot(list(dict['centered_metric']))
# y_smoothed = gaussian_filter1d(L, sigma=3)
# y_centered=y_smoothed-np.mean(y_smoothed)

# plt.plot(y_centered)
# plot_data_segmented(data,precision)

# dt = datetime(year=2020,month=1,day=1,)

# timestamp = int(dt.timestamp())
# print(timestamp)
def adding_image(im,date):
    ax = plt.subplot(111)
    im = OffsetImage(im, zoom=.05)
    ab = AnnotationBbox(im, (date, 120), xycoords='data', box_alignment=(1.1,-0.1))
    ax.add_artist(ab)