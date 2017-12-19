# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:17:05 2016

@author: mb
"""


import matplotlib.pyplot as plt
import numpy as np


# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

#
'''drill = np.array([1030.4056799999996, 633.4658728459998, 1247.3726350659999, 369.00662127599998, 1355.5090700859992, 113.96680270499998, 2278.1888899999985, 1695.305396666])
runs = np.array([1718.0226499999994, 2052.806530544,    1086.8455363809999, 1782.328888882,     343.45487529000002, 1466.5102947389996, 1065.7334899999998, 717.75092969000013])
tots = drill+runs'''

#RUNS VS DRILLS
mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)
tots = np.array([548,482,441,157,436,70,562,340]).astype(float)
drill = np.array([426,329,387,94,413,25,519,312])
labels = ['E-1', 'E-2', 'I-1', 'I-2', 'I-3','I-4','N-1','N-2']
drills = 100*(drill/tots)
runs = 100-drills
ind = np.arange(8)    # the x locations for the groups
width = 0.8      # the width of the bars: can also be len(x) sequence
p1 = ax.bar(ind, drills, width, color='brown',label='Drill/Work',alpha=0.5)
p2 = ax.bar(ind, runs, width, color='blue',
             bottom=drills,label='Run',alpha=0.5)
ax.set_ylabel('Percentage of number of Drill vs Runs segments')
ax.set_xlabel('Subject')
ax.set_title('Ratio of Drills to Runs segments')
ax.legend()
ax.set_xticks(ind + width/2.)
#ax.set_yticks(np.arange(0, 81, 10))
ax.set_xticklabels((labels))
ax.legend(loc='lower center', bbox_to_anchor=(1, 0.5))

#COMPRESSION SIZE
'''mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)
y_pos = np.arange(len(labels))
compression = [16958,15131,13511,5098,13291,2438,5318,3428] 
ax.bar(y_pos, compression, width, alpha=0.5)
ax.set_xticks(ind + width/2.)
ax.set_xticklabels((labels))
ax.set_ylabel('Size of Compressed files in Bytes')
ax.set_title('Compression of Chaffin Graph')'''


#sone_val = np.array([0,13,26,36,52,75,105])
#midi_val = np.array([0,50,65,77.5,90,110,127])

#sone_val = np.array([0, 7.5, 19.5, 31,   44,   63.5,  90,105])
#midi_val = np.array([0, 25,  57.5, 72.5, 82.5, 100, 118.5, 127])

#plt.figure()
#plt.plot(sone_val,midi_val)

'''multiply = midi_val/sone_val
multiply[0]=100000'''

a = 129.521228779
b = 0.0209107533354
c = 136.802176681
#test = -a*np.exp(b*(-sone_val))+c
#plt.plot(sone_val,test)

#avg_loud = (perf_avg_sone*10)
#avg_lous = -a*np.exp(b*(-avg_loud))+c

#plt.plot(avg_lous)

#plt.plot(midi_val)
#plt.plot(sone_val)
#labelss = ['MazurkaDS','E-1', 'E-2', 'I-1', 'I-2', 'I-3','I-4','N-1','N-2']
#tempo_data = [perf_avg_tempo,meanT[0],meanT[7],meanT[1],meanT[2],meanT[3],meanT[4],meanT[5],meanT[6]]
#loud_data = [perf_avg_sone,meanL[0],meanL[7],meanL[1],meanL[2],meanL[3],meanL[4],meanL[5],meanL[6]]


labelss = ['E-1', 'E-2', 'I-1', 'I-2', 'I-3','I-4','N-1','N-2']
tempo_data = [meanT[0],meanT[7],meanT[1],meanT[2],meanT[3],meanT[4],meanT[5],meanT[6]]
loud_data = [meanL[0],meanL[7],meanL[1],meanL[2],meanL[3],meanL[4],meanL[5],meanL[6]]


#midi_perfus = [perf_avg_sone,segments0m[:,2]*127,segments7m[:,2]*127,segments1m[:,2]*127,segments2m[:,2]*127,segments3m[:,2]*127,segments4m[:,2]*127,segments5m[:,2]*127,segments6m[:,2]*127]



for i in range(0,len(tempo_data)):
    #print np.around(tempo_data[i].mean(),decimals=2),np.around(tempo_data[i].std(),decimals=2),np.around(tempo_data[i].min(),decimals=2),np.around(tempo_data[i].max(),decimals=2)
    print np.around(loud_data[i].mean(),decimals=2),np.around(loud_data[i].std(),decimals=2),np.around(loud_data[i].min(),decimals=2),np.around(loud_data[i].max(),decimals=2)
    
    
    
np.around(loud_data[i].mean(),decimals=2)



f, (ax1, ax2) = plt.subplots(1, 2)

ax1.boxplot(tempo_data,labels=labelss)
ax1.set_title('Tempo Distribution')
ax1.set_ylabel('Tempo in BPM')

ax2.boxplot(loud_data,labels=labelss)
ax2.set_title('Loudness Distribution')
ax2.set_ylabel('MIDI Velocity')
f.suptitle('Distribution of Expressive Parameters')
f.tight_layout()



tempo = (0.47,0.29,0.08,0.3,-0.26,-0.15,0.12,0)
loudness = (0.75,0.69,0.62,0.66,0.53,0.51,0.24,0.21)

ind = np.arange(len(tempo))  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, tempo, width, color='r')#, yerr=tempo)


rects2 = ax.bar(ind + width, loudness, width, color='y')#, yerr=loudness)

# add some text for labels, title and axes ticks
ax.set_ylabel('Correlation with Mazurka Dataset')
ax.set_title('Correlation with Mazurka Dataset')
ax.set_xticks(ind + width)
ax.set_xticklabels(('E-1', 'E-2', 'I-1', 'I-2', 'I-3','I-4','N-1','N-2'))

ax.legend((rects1[0], rects2[0]), ('Tempo', 'Loudness'))


'''def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)'''

plt.show()


'''from scipy import interpolate
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return -a * np.exp(b * (-x)) + c

x = sone_val
y = midi_val

#multiply = y/x
#multiply[0]=6

#y=multiply

#x = np.arange(len(multiply))

f = interpolate.interp1d(x, y)
#plt.figure()
xnew = np.linspace(0, max(x), 1051)
ynew = f(xnew)
#ynew = ynew + np.random.normal(0,1,len(xnew))   # use interpolation function returned by `interp1d`
#plt.plot(x, y, 'o', xnew, ynew, '.')
#plt.show()
#yn = y + 0.2*np.random.normal(size=len(x))

ave_func = np.zeros((10000,3))
for i in range (0,10000):
    ynews = ynew + 2*np.random.normal(0,1,len(xnew))
    #print min(ynews)
    popt, pcov = curve_fit(func, xnew, ynews)
    ave_func[i]=popt

print ave_func[:,0].mean(),ave_func[:,1].mean(),ave_func[:,2].mean()
print ave_func[:,0].std(),ave_func[:,1].std(),ave_func[:,2].std()



#plt.figure()
#plt.plot(xnew, ynew, 'k.', label="Original Noised Data")
#plt.plot(xnew, func(xnew, *popt), 'r-', label="Fitted Curve")
#plt.legend()
#plt.show()'''

spread = np.random.rand(50) * 100