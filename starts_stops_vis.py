import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib.patches as patches
import os
from scipy import stats
import time
from scipy import interpolate
from scipy import signal
#import bayesian_changepoint_detection.offline_changepoint_detection as offcd
#import bayesian_changepoint_detection.online_changepoint_detection as oncd
from functools import partial
import matplotlib.cm as cm

import sklearn as sklearn
#from sklearn import metrics
from sklearn.cluster import KMeans

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import matplotlib.cm as cm
import pywt
#import powerlaw

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

mode = pywt.Modes.smooth

def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.

    S = An + Dn + Dn-1 + ... + D1
    """
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    for i in range(5):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    fig = plt.figure()
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))

def clusterTing(beat_segments):
    y = np.arange(len(beat_segments))
    x = (beat_segments[:,0].astype(float)+beat_segments[:,1])/2
    
    X = np.zeros((len(beat_segments),2))
    
    X[:,0]=x
    X[:,1]=y
    
    bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=50)
    
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    print("number of estimated clusters : %d" % n_clusters_)
    
    #import matplotlib.pyplot as plt
    
    
    plt.figure()
    plt.clf()
    
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def clusterAgg(beat_segments):
    y = np.arange(len(beat_segments))
    x = (beat_segments[:,0].astype(float)+beat_segments[:,1])/2
    
    X = np.zeros((len(beat_segments),2))
    
    X[:,0]=x
    X[:,1]=y
    
    # Create a graph capturing local connectivity. Larger number of neighbors
    # will give more homogeneous clusters to the cost of computation
    # time. A very large number of neighbors gives more evenly distributed
    # cluster sizes, but may not impose the local manifold structure of
    # the data
    knn_graph = kneighbors_graph(X, 5, include_self=False)
    
    for connectivity in (None, knn_graph):
        for n_clusters in (30, 3):
            plt.figure(figsize=(10, 4))
            for index, linkage in enumerate(('average', 'complete', 'ward')):
                plt.subplot(1, 3, index + 1)
                model = AgglomerativeClustering(linkage=linkage,
                                                connectivity=connectivity,
                                                n_clusters=n_clusters)
                t0 = time.time()
                model.fit(X)
                elapsed_time = time.time() - t0
                plt.scatter(X[:, 0], X[:, 1], c=model.labels_,
                            cmap=plt.cm.spectral)
                plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time),
                          fontdict=dict(verticalalignment='top'))
                #plt.axis('equal')
                #plt.axis('off')
    
                plt.subplots_adjust(bottom=0, top=.89, wspace=0,
                                    left=0, right=1)
                plt.suptitle('n_cluster=%i, connectivity=%r' %
                             (n_clusters, connectivity is not None), size=17)

subsec_bound = [1,12,60,107,129,180,204,228,252,276,323,347,372,385,396]
subsec_color = np.array(['#487f7b','#90d0ff','#90d0ff','#f790ff','#90d0ff','#ff0012','#ff0012','#ff0012','#ff0012','#90d0ff','#fff790','#fff790','#7f484c','#487f7b'])

def drawRect(x1,x2,y,h,axis,fc):
    axis.add_patch(
        patches.Rectangle(
            (x1, y),   # (x,y)
            x2-x1,          # width
            h+2,          # height
            facecolor = fc,
            alpha = 0.1,
            linestyle = None,
        )
    )

path_ting = "/Volumes/MAC_BACKUP/Organised_Practices"

#from glob import glob
def getAvg():
    pathT = path_ting + '/other_perf/kat_tempo'
    pathS = path_ting + '/other_perf/kat_sone'
    
    tempo = np.zeros(395)
    tempo_all = genfromtxt(path_ting + '/other_perf/tempo_all.csv',delimiter=',')
    tempo_index = []
    
    sone = np.zeros(395)
    
    #GET THE TEMPOS AND SONES FROM PERFS
    f, ax = plt.subplots(2)
    for i in range(0,len(os.listdir(pathT))):
        ts = np.zeros(396)
        ts[:-1] = tempo_all[:,i]
        ts[-1] = ts[-2]+ts[-2]-ts[-3]
        delta_ts=ts[1:]-ts[:-1]
        temposs = 60/(delta_ts)
        if max(temposs)<320:
            print i
            temposs = 60/(delta_ts)
            #temposs = stats.zscore(temposs)
            tempo_index.append(i)
            tempo = np.vstack((tempo,temposs))
            ax[0].plot(temposs,alpha=0.2,color='k')
            sone_s = genfromtxt(pathS+'/'+os.listdir(pathS)[i],delimiter=',')
            sone_i = np.zeros(395)
            for j in range(len(ts)-1):
                index = np.array((sone_s[:,0] >= ts[j]) & (sone_s[:,0] < ts[j+1]))
                sone_i[j] = np.mean(np.array(sone_s[index])[:,1])*10
            #sone_i = stats.zscore(sone_i)
            sone = np.vstack((sone,sone_i))
            ax[1].plot(sone_i,alpha=0.2,color='k')
    tempo = tempo[1:]
    sone = sone[1:]
    
    perf_avg_tempo = np.zeros(395)
    perf_avg_sone = np.zeros(395)
    for i in range(0,395):
        perf_avg_tempo[i] = np.mean(tempo[:,i])
        perf_avg_sone[i] = np.mean(sone[:,i])
        
    ax[0].plot(perf_avg_tempo,lw=2,color = 'b',label='Average Tempo')
    ax[1].plot(perf_avg_sone,lw=2,color = 'b')
    ax[0].set_title('Average Expressive Parameters for Mazurka Dataset')
    ax[0].set_ylabel('Tempo in BPM')
    ax[1].set_ylabel('MIDI Velocity')
    ax[1].set_xlabel('Length in Beats')
    ax[0].legend()
    return perf_avg_tempo, perf_avg_sone
    
#GET THE SONES FROM PERSF
'''sone = np.zeros(395)
for i in range(0,len(tempo_index)):
    sone_s = genfromtxt(pathS+'/'+os.listdir(pathS)[tempo_index[i]],delimiter=',')
    for i in range(0,len(tempo[i])):
        index = (segment_tempo >= segment_times[i,0]) & (segment_tempo <= (segment_times[i,0]+segment_times[i,1]))'''

def oneArray(interesti,j,segmentos):
    beat_segments = segmentos[interesti[j]:interesti[j+1]]
    segs = np.arange(np.array(len(beat_segments)).astype(float))/(len(beat_segments)-1)
    mid = stats.zscore((beat_segments[:,0].astype(float)+beat_segments[:,1])/2)
    fmid = interpolate.interp1d(segs, mid)
    segsnew = np.linspace(0,1,100)
    midnew = fmid(segsnew)
    return midnew

def fishbone(start_hist,stop_hist,title,last_beat_played):
    fig, ax1 = plt.subplots(figsize=(12,4))
    for i in range(0,len(subsec_color)):
        drawRect(subsec_bound[i],subsec_bound[i+1],min(stop_hist)-10,(max(start_hist)+10+abs(min(stop_hist)-10)),ax1,subsec_color[i])
        
    ax1.stem(start_hist,'g',markerfmt=' ')
    ax1.stem(stop_hist,'r',markerfmt=' ')
    ax1.set_ylim(min(stop_hist)-5,max(start_hist)+5)
    ax1.set_xlim(0,last_beat_played)
    ax1.set_title('Frequency of Starts and Stops for ' + title)
    ax1.set_xlabel('Length in Beats')
    ax1.set_ylabel('Frequency')
    fig.tight_layout()
    #fig.savefig('/Volumes/Untitled/Organised_Practices/visuals/'+str(j)+'/0_fish_map.pdf')
    #plt.close()

#perf_avg_tempo, perf_avg_sone = getAvg()

avg_bpm = []
yos = []

#IMPORT SEGMENTS FROM PRACTICE
segments0 = genfromtxt(path_ting + '/Experts/Nika/nika_segments.csv',delimiter=',',dtype=int)
segments0t = genfromtxt(path_ting + '/Experts/Nika/nika_times.csv',delimiter=',')
segments0m = genfromtxt(path_ting + '/Experts/Nika/nika_midi.csv',delimiter=',',usecols=(0,1,3))
segments0i = [-1,33.5,104.5,165.5,213.5,234.5,248.5,265.5,300.5,316.5,341.5,375.5,407.5,463.5,513.5]
interesti0 = np.array(segments0i)+0.5
interesti0[0]=0
interesti0 = np.hstack((interesti0,len(segments0)))

segments1 = genfromtxt(path_ting + '/Intermediate/Janis/janis_segments.csv',delimiter=',',dtype=int)
segments1t = genfromtxt(path_ting + '/Intermediate/Janis/janis_times.csv',delimiter=',')
segments1m = genfromtxt(path_ting + '/Intermediate/Janis/janis_midi.csv',delimiter=',',usecols=(0,1,3))
segments1i = [-1,52.5,167.5,211.5,245.5,294.5,354.5,387.5]
interesti1 = np.array(segments1i)+0.5
interesti1[0]=0
interesti1 = np.hstack((interesti1,len(segments1)))

segments2 = genfromtxt(path_ting + '/Intermediate/ShCh/shch_segments.csv',delimiter=',',dtype=int)
segments2t = genfromtxt(path_ting + '/Intermediate/ShCh/shch_times.csv',delimiter=',')
segments2m = genfromtxt(path_ting + '/Intermediate/ShCh/shch_midi.csv',delimiter=',',usecols=(0,1,3))
segments2i = [-1,63.5,83.5,115.5,131.5]
interesti2 = np.array(segments2i)+0.5
interesti2[0]=0
interesti2 = np.hstack((interesti2,len(segments2)))

segments3 = genfromtxt(path_ting + '/Intermediate/Stratos/stratos_segments.csv',delimiter=',',usecols=(3,4,5,6),dtype=int)
segments3t = genfromtxt(path_ting + '/Intermediate/Stratos/stratos_times.csv',delimiter=',')
segments3m = genfromtxt(path_ting + '/Intermediate/Stratos/stratos_midi.csv',delimiter=',',usecols=(0,1,3))
segments3i = [-1,17.5,87.5,152.5]
interesti3 = np.array(segments3i)+0.5
interesti3[0]=0
interesti3 = np.hstack((interesti3,len(segments3)))

segments4 = genfromtxt(path_ting + '/Intermediate/Katerina/katerina_segments.csv',delimiter=',',usecols=(3,4,5,6),dtype=int)
segments4t = genfromtxt(path_ting + '/Intermediate/Katerina/katerina_times.csv',delimiter=',')
segments4m = genfromtxt(path_ting + '/Intermediate/Katerina/katerina_midi.csv',delimiter=',',usecols=(0,1,3))
segments4i = [-1,5.5,21.5,25.5,38.5,57.5]
interesti4 = np.array(segments4i)+0.5
interesti4[0]=0
interesti4 = np.hstack((interesti4,len(segments4)))

segments5 = genfromtxt(path_ting + '/Novice/Giulio/giulio_segments.csv',delimiter=',',dtype=int)
segments5t = genfromtxt(path_ting + '/Novice/Giulio/giulio_times.csv',delimiter=',')
segments5m = genfromtxt(path_ting + '/Novice/Giulio/giulio_midi.csv',delimiter=',',usecols=(0,1,3))
segments5i = [-1,57.5,124.5,148.5,292.5,348.5,504.5,531.5,546.5]
interesti5 = np.array(segments5i)+0.5
interesti5[0]=0
interesti5 = np.hstack((interesti5,len(segments5)))

segments6 = genfromtxt(path_ting + '/Novice/Manos/manos_segments.csv',delimiter=',',usecols=(3,4,5,6),dtype=int)
segments6t = genfromtxt(path_ting + '/Novice/Manos/manos_times.csv',delimiter=',')
segments6m = genfromtxt(path_ting + '/Novice/Manos/manos_midi.csv',delimiter=',',usecols=(0,1,3))
segments6i = [-1,15.5,23.5,35.5,43.5,53.5,68.5,81.5,96.5,103.5,127.5,173.5,203.5,216.5,240.5,268.5,294.5,300.5,323.5]
interesti6 = np.array(segments6i)+0.5
interesti6[0]=0
interesti6 = np.hstack((interesti6,len(segments5)))

segments7 = genfromtxt(path_ting + '/Experts/Petr/petr_segments.csv',delimiter=',',usecols=(3,4,5,6),dtype=int)
segments7t = genfromtxt(path_ting + '/Experts/Petr/petr_segments.csv',delimiter=',',usecols=(0,2))
segments7m = genfromtxt(path_ting + '/Experts/Petr/petr_midi.csv',delimiter=',',usecols=(0,1,3))
segments7i = [-1,71.5,103.5,208.5,219.5,293.5,311.5,324.5,392.5,414.5,434.5]
interesti7 = np.array(segments7i)+0.5
interesti7[0]=0
interesti7 = np.hstack((interesti7,len(segments7)))


hello = [segments0,segments1,segments2,segments3,segments4,segments5,segments6,segments7]
times = [segments0t,segments1t,segments2t,segments3t,segments4t,segments5t,segments6t,segments7t]
interesting = [segments0i,segments1i,segments2i,segments3i,segments4i,segments5i,segments6i,segments7i]
interesti = [interesti0,interesti1,interesti2,interesti3,interesti4,interesti5,interesti6,interesti7]


meanT = []
meanL = []



#avg_tempos = []
loud = [segments0m,segments1m,segments2m,segments3m,segments4m,segments5m,segments6m,segments7m]
titles = ['E-1','I-1','I-2','I-3','I-4','N-1','N-2','E-2']

phrases =  genfromtxt(path_ting + '/segments/phrases_new.csv', delimiter=',',usecols=(1,2,3,4),dtype=int)
beat_phrases = np.zeros((len(phrases),3)).astype(int)
input_matrices = []

big_cluster = np.zeros((77,100))

times_drill = []
times_smooth = []

#for i in range(0,len(phrases)):
for i in range(0,len(phrases)):
    beat_start = (phrases[i,0]-1)*3 + (phrases[i,1]-1)
    beat_end = (phrases[i,2]-1)*3 + (phrases[i,3])
    beat_len = beat_end - beat_start
    beat_phrases[i] = [beat_start,beat_end,beat_len]

run_len = np.round(np.mean(beat_phrases[:,2])*2).astype(int)
yo = 0
bo = 0
fiff, axurr = plt.subplots(8,sharex=True)
for j in range(0,8):
    print j
    #os.makedirs(path_ting + '/visuals/'+str(j))
    segments = hello[j]
    #print len(segments)
    beat_segments = np.zeros((len(segments),3)).astype(int)
    
    times_d = 0
    times_s = 0
    #GET SEGMENT STARTS AND STOPS IN BEATS
    start_hist = np.zeros(396)
    stop_hist = np.zeros(397)
    
    runs = np.array([0,0,0])
    drill = np.array([0,0,0])
    rc = 0
    dc = 0
    pic = np.zeros( ( len(beat_segments),np.max((segments[:,2]-1)*3 + (segments[:,3]))))
    for i in range(0,len(segments)):
        beat_start = (segments[i,0]-1)*3 + (segments[i,1]-1)
        start_hist[beat_start] += 1
        beat_end = (segments[i,2]-1)*3 + (segments[i,3])
        stop_hist[beat_end] -= 1
        beat_len = beat_end - beat_start
        beat_segments[i] = [beat_start,beat_end,beat_len]
        if (beat_len)>=run_len:
            runs = np.vstack([runs,beat_segments[i]])
            rc+=1
            pic[i,beat_start:beat_end].fill(2)
            times_s += times[j][i,1]
            #print 'r'
        else:
            drill = np.vstack([drill,beat_segments[i]])
            dc+=1
            pic[i,beat_start:beat_end].fill(1)
            times_d += times[j][i,1]
            #print 'd'
    runs = runs[1:]
    drill = drill[1:]
    last_beat_played = max(beat_segments[:,1])
    input_matrices.append(pic)
    
    #powerlaw stuff
    #results = powerlaw.Fit(beat_segments[:,2]) 
    #powerlaw.plot_pdf(beat_segments[:,2])
    #print results.power_law.alpha 
    #print results.power_law.xmin 
    #R, p = results.distribution_compare('power_law', 'lognormal')
    
    #FISHBONE
    '''fig, ax2 = plt.subplots(figsize=(12,4))
    for i in range(0,len(subsec_color)):
        drawRect(subsec_bound[i],subsec_bound[i+1],min(stop_hist)-10,(max(start_hist)+10+abs(min(stop_hist)-10)),ax2,subsec_color[i])
        
    ax2.stem(start_hist,'g',markerfmt=' ')
    ax2.stem(stop_hist,'r',markerfmt=' ')
    ax2.set_ylim(min(stop_hist)-5,max(start_hist)+5)
    ax2.set_xlim(0,last_beat_played)
    ax2.set_title('Frequency of Starts and Stops for ' + titles[j])
    ax2.set_xlabel('Length in Beats')
    ax2.set_ylabel('Frequency')
    fig.savefig('/Volumes/Untitled/Organised_Practices/figures/'+titles[j]+'_fish.pdf')
    fig.tight_layout()'''
    
    #VISUALISING CHAFFIN MAP
    fig, ax = plt.subplots(figsize=(6,8))
    for i in range(0,len(subsec_color)):
        drawRect(subsec_bound[i],subsec_bound[i+1],-1,len(segments),ax,subsec_color[i])
    for i in range(0,len(beat_segments)):
        x = np.linspace(beat_segments[i,0],beat_segments[i,1],beat_segments[i,2]+1).astype(int)
        y = np.full(len(x),i,dtype=int)
        if (beat_segments[i,2])>=run_len:
            ax.plot(x,y,color='blue',lw=2,alpha=0.7)

        else:
            ax.plot(x,y,color='brown',lw=2,alpha=0.7)

    ax.set_ylim(-1,len(segments)+1)
    #ax.set_xlim(0,last_beat_played)
    ax.set_title('Practice Session Map for '+titles[j])
    ax.set_ylabel('Segment Number')
    ax.set_xlabel('Length in Beats')
    '''interest = interesting[j]
    for i in range(0,len(interest)):
        plt.axhline(y=interest[i],linestyle='--',color='k',alpha=0.5)
        ax.text(50, interest[i]+1, str(i+1), fontsize=15)
        #ax.text(50, interest[i]+1, str(bo)+' '+str(predict[bo]), fontsize=15)
        bo+=1
    interest.append(len(beat_segments))
    interest[0]=0
    ax.set_yticks(np.array(interest).astype(int))'''
    fig.tight_layout()
    
    #clusterTing(beat_segments)
    #clusterAgg(beat_segments)
    #plt.close()
    #fig.savefig('/Volumes/Untitled/Organised_Practices/figures/'+titles[j]+'_chaff_map.pdf')
    #plt.grid(True)'''
    
    #REPETITION ARCS

    '''for i in range(0,len(subsec_color)):
        drawRect(subsec_bound[i],subsec_bound[i+1],-1,len(segments),axurr[j],subsec_color[i])
    maximum = 0
    if j == 0:
        bum = j
    if j == 7:
        bum = 1
    if (j < 7) and (j>0):
        bum = j+1'''
    maximum = 0 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(0,len(beat_segments)):
        start_beat = beat_segments[i,0]
        end_beat = beat_segments[i,1]
        seg_len = beat_segments[i,2]
        x = np.linspace(start_beat,end_beat,seg_len*10+1)
        r = float(seg_len)/2
        if r > maximum:
            maximum = r
        mid_x = float(start_beat + float(seg_len)/2)
        #print(mid_x)
        y = np.sqrt(r**2 - (x-mid_x)**2)
        z = np.full((1,len(y)),i, dtype=int)
        #print(y)
        if seg_len >= run_len:
            colour = ('blue')
        else:
            colour = ('brown')
        ax.plot(x,y,i,zdir='z')#,color=colour,alpha=0.6)

    '''axurr[bum].set_ylim(0,maximum+5)
    #axurr[bum].set_xlim(0,last_beat_played)
    axurr[bum].yaxis.set_ticklabels([]) 
    axurr[bum].set_ylabel(titles[bum])'''
    
    #f.savefig('/Volumes/Untitled/Organised_Practices/figures/'+titles[j]+'_rep.pdf')'''
    #plt.close()
    
    all_start_hist = []
    all_stop_hist = []
    for i in range(0,len(beat_segments)):
        all_start_hist.append((subsec_bound - beat_segments[i,0])[np.argmin(abs(subsec_bound - beat_segments[i,0]))])
        all_stop_hist.append((subsec_bound - beat_segments[i,1])[np.argmin(abs(subsec_bound - beat_segments[i,1]))])
        
    '''plt.figure()
    plt.hist(all_start_hist,histtype='stepfilled',bins=50,label='Starts',color='g')
    plt.hist(all_stop_hist,histtype='stepfilled',bins=50,label='Stops',alpha=0.5,color='r')
    plt.legend()
    plt.xlabel('Distance from Section Boundary')
    plt.ylabel('Frequency')
    plt.suptitle('All Starts/Stops for '+titles[j])
    plt.savefig('/Volumes/Untitled/Organised_Practices/visuals/'+str(j)+'/3_all_seg.pdf')
    plt.close()
    '''
    run_start_hist = []
    run_stop_hist = []
    for i in range(0,len(runs)):
        run_start_hist.append((subsec_bound - runs[i,0])[np.argmin(abs(subsec_bound - runs[i,0]))])
        run_stop_hist.append((subsec_bound - runs[i,1])[np.argmin(abs(subsec_bound - runs[i,1]))])
    run_start_hist = np.array(run_start_hist)
    run_stop_hist = np.array(run_stop_hist)
    
    '''plt.figure()
    plt.hist(run_start_hist,histtype='stepfilled',bins=50,label='Starts',color='g')
    plt.hist(run_stop_hist,histtype='stepfilled',bins=50,label='Stops',alpha=0.5,color='r')
    plt.legend()
    plt.xlabel('Distance from Section Boundary')
    plt.ylabel('Frequency')
    plt.suptitle('Run Starts/Stops for '+ titles[j])
    plt.savefig('/Volumes/Untitled/Organised_Practices/visuals/'+str(j)+'/4_run_seg.pdf')
    plt.close()
    '''
    drill_start_hist = []
    drill_stop_hist = []
    for i in range(0,len(drill)):
        drill_start_hist.append((subsec_bound - drill[i,0])[np.argmin(abs(subsec_bound - drill[i,0]))])
        drill_stop_hist.append((subsec_bound - drill[i,1])[np.argmin(abs(subsec_bound - drill[i,1]))])
    drill_start_hist = np.array(drill_start_hist)
    drill_stop_hist = np.array(drill_stop_hist)
        
    '''plt.figure()
    plt.hist(drill_start_hist,histtype='stepfilled',bins=50,label='Starts',color='g')
    plt.hist(drill_stop_hist,histtype='stepfilled',bins=50,label='Stops',alpha=0.5,color='r')
    plt.legend()
    plt.xlabel('Distance from Section Boundary')
    plt.ylabel('Frequency')
    plt.suptitle('Drill Starts/Stops for '+ titles[j])
    plt.savefig('/Volumes/Untitled/Organised_Practices/visuals/'+str(j)+'/5_drill_seg.pdf')
    plt.close()
    '''
    drill_start_sec = np.around(len(np.where(drill_start_hist==0)[0])/np.float(len(drill)),decimals=2)
    drill_stop_sec = np.around(len(np.where(drill_stop_hist==0)[0])/np.float(len(drill)),decimals=2)
    drill_avg_len = np.around(np.mean(drill[:,2].astype(float)),decimals=2)  
    run_start_sec = np.around(len(np.where(run_start_hist==0)[0])/np.float(len(runs)),decimals=2)
    run_stop_sec = np.around(len(np.where(run_stop_hist==0)[0])/np.float(len(runs)),decimals=2)
    run_avg_len = np.around(np.mean(runs[:,2].astype(float)),decimals=2)  
    
    time_played = np.around(np.sum(times[j][-1])/60.,decimals=2)
    
    #print time_played,'&',len(segments),'&',np.around(len(segments)/time_played,decimals=2),'&',len(drill),'&',drill_avg_len,'&',drill_start_sec,'&',drill_stop_sec,'&',len(runs),'&',run_avg_len,'&',run_start_sec,'&',run_stop_sec
    
    avg_tempos = np.zeros((len(beat_segments),last_beat_played))
    avg_sones = np.zeros((len(beat_segments),last_beat_played))  
    
    loudness = loud[j]    
    
    segment_times = np.array(times[j])
    seg_tempo = np.zeros(len(segments))
    seg_sone = np.zeros(len(segments))
    
    for i in range(0,len(beat_segments)):
        index = (loudness[:,0] >= segment_times[i,0]) & (loudness[:,0] <= (segment_times[i,0]+segment_times[i,1]))
        avg_loud = np.mean(loudness[index,2])
        avg_tempo = beat_segments[i,2]/segment_times[i,1]*60
        if beat_segments[i,2]==1:
            seg_tempo[i]=0
            seg_sone[i]=0
        else:
            seg_tempo[i]=avg_tempo
            seg_sone[i]=avg_loud
        avg_tempos[i,beat_segments[i,0]:beat_segments[i,1]].fill(avg_tempo)
        avg_sones[i,beat_segments[i,0]:beat_segments[i,1]].fill(avg_loud)
        
    '''plt.figure()
    plt.plot(np.linspace(0,len(segments)-1,len(segments)),seg_tempo)
    plt.suptitle(str(titles[j]))'''

    means_n = np.zeros(last_beat_played-1)
    means_s = np.zeros(last_beat_played-1)
    for i in range(0,last_beat_played-1):
        beat = i+1
        non_zero_index = np.flatnonzero(avg_tempos[:,beat])
        beats = avg_tempos[:,beat]
        sones = avg_sones[:,beat]
        data_t = beats[non_zero_index]
        data_s = sones[non_zero_index]
        mean = data_t.mean()
        std = data_t.std()
        means_n[i]=mean
        means_s[i]=data_s.mean()
    #means_ns = stats.zscore(means_n)
    #means_ss = stats.zscore(means_s)
    means_ns = means_n
    means_ss = means_s
    interest = np.array(interesti[j]).astype(int)
    for k in range(len(interest)-1):
        big_cluster[yo,:]=oneArray(interest,k,beat_segments)
        #plt.figure()
        #plt.plot(big_cluster[yo,:])
        yo+=1
    
    '''f, ax = plt.subplots(2,figsize=(12,6),sharex=True)
    x = np.linspace(1,last_beat_played-1,last_beat_played-1)
    ax[0].plot(x,perf_avg_tempo[:(last_beat_played-1)],label='Dataset')
    ax[0].plot(x,means_ns,label='Subject')
    ax[0].set_xlim(1,last_beat_played)
    ax[0].set_ylabel('Normalised Tempo')
    ax[0].legend()
    ax[1].plot(x,perf_avg_sone[:(last_beat_played-1)])
    ax[1].plot(x,means_ss)
    ax[1].set_xlabel('Length in Beats')
    ax[1].set_ylabel('Normalised Loudness')
    ax[0].set_title('Expressivity comparison for '+str(titles[j]))
    
    for i in range(0,len(subsec_color)):
        drawRect(subsec_bound[i],subsec_bound[i+1],-5,10,ax[0],subsec_color[i])
        drawRect(subsec_bound[i],subsec_bound[i+1],-5,10,ax[1],subsec_color[i])
    ax[0].set_ylim(-4,4)
    ax[1].set_ylim(-4,4)
    #f.savefig('/Volumes/Untitled/Organised_Practices/figures/'+titles[j]+'_express.pdf')'''
    
    '''plt.figure()
    plt.plot(np.linspace(1,last_beat_played-1,last_beat_played-1),means_n)
    plt.suptitle(str(titles[j]))
    midi_perf = loud[j]
    plt.figure()
    plt.hist(np.around(midi_perf[:,2]*127),bins=127,histtype='stepfilled')
    plt.suptitle('Total Loudness Hist for '+ (str(titles[j])))
    
    plt.figure()
    plt.hist(beat_segments[:,2],bins=max(beat_segments[:,2]))'''
    '''f, axarr = plt.subplots(6,sharex=True)
    for i in range(0,3):
        mean_starts = np.convolve(beat_segments[:,0], np.ones((i*10+1,))/(i*10+1), mode='same')
        mean_ends = np.convolve(beat_segments[:,1], np.ones((i*10+1,))/(i*10+1), mode='same')
        mean_lens = np.convolve(beat_segments[:,2], np.ones((i*10+1,))/(i*10+1), mode='same')
        axarr[i*2].plot(mean_starts)
        axarr[i*2].plot(mean_ends)
        axarr[i*2].plot(mean_lens)
        for k in range(0,len(interest)):
            axarr[i*2+1].axvline(interest[k])
        axarr[i*2+1].plot(np.diff(mean_starts))
        axarr[i*2+1].plot(np.diff(mean_ends))
        axarr[i*2+1].axhline(0)
        #axarr[i*2+1].plot(np.diff(mean_lens))
        
        axarr[i].set_title(str(j)+' '+str(i*5+1))
        
    widths = np.arange(1, 30)
    f, axarr = plt.subplots(2,sharex=True)
    cwtmatr_start = signal.cwt(beat_segments[:,0], signal.ricker, widths)
    axarr[0].imshow(cwtmatr_start, extent=[-1, 1, 1, 201], aspect='auto',
               vmax=abs(cwtmatr_start).max(), vmin=-abs(cwtmatr_start).max())
    cwtmatr_stop = signal.cwt(beat_segments[:,1], signal.ricker, widths)
    axarr[1].imshow(cwtmatr_stop, extent=[-1, 1, 1, 201], aspect='auto',
               vmax=abs(cwtmatr_stop).max(), vmin=-abs(cwtmatr_stop).max())
    axarr[0].set_title(str(j))
    plt.figure()
    for i in range(0,len(cwtmatr_start[:,2])):
        plt.plot(cwtmatr_start[i,:],alpha=0.2,color='g')
        plt.plot(cwtmatr_stop[i,:],alpha=0.2,color='r')
    plt.suptitle(str(j))
    
    plot_signal_decomp(beat_segments[:,0], 'haar', "Starts")
    
    Q, P, Pcp = offcd.offline_changepoint_detection(beat_segments[:,0], partial(offcd.const_prior, l=(len(beat_segments[:,0])+1)), offcd.gaussian_obs_log_likelihood, truncate=-40)

    fig, ax = plt.subplots()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(beat_segments[:,0])
    for k in range(0,len(interest)):
        ax.axvline(interest[k])
    ax = fig.add_subplot(2, 1, 2, sharex=ax)
    ax.plot(np.exp(Pcp).sum(0))
    for k in range(0,len(interest)):
        ax.axvline(interest[k])
    
    R, maxes = oncd.online_changepoint_detection(beat_segments[:,0], partial(oncd.constant_hazard, 250), oncd.StudentT(0.1, .01, 1, 0))
    fig, ax = plt.subplots(figsize=[18, 16])
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(beat_segments[:,0])
    for k in range(0,len(interest)):
        ax.axvline(interest[k])
    ax = fig.add_subplot(3, 1, 2, sharex=ax)
    sparsity = 5  # only plot every fifth data for faster display
    ax.pcolor(np.array(range(0, len(R[:,0]), sparsity)), 
              np.array(range(0, len(R[:,0]), sparsity)), 
              -np.log(R[0:-1:sparsity, 0:-1:sparsity]), 
              cmap=cm.Greys, vmin=0, vmax=30)
    ax = fig.add_subplot(3, 1, 3, sharex=ax)
    Nw=10;
    ax.plot(R[Nw,Nw:-1])
    for k in range(0,len(interest)):
        ax.axvline(interest[k])
    
    yos.append(beat_segments)
    #plot_signal_decomp(beat_segments[:,1], 'haar', "Stops")
    #plot_signal_decomp(beat_segments[:,2], 'haar', "Lengths")'''
    yos.append(beat_segments)
    meanT.append(means_n)
    meanL.append(means_s*127)
    #means_n = means_n[1:]
    #print j, np.sum(beat_segments[:,2])
    #np.sum(segment_times[:,1])/np.sum(beat_segments[:,2])
    print titles[j]
    #print 'Time spent at the piano ', int(divmod(np.sum(segment_times[-1]),60)[0]),'m',int(divmod(np.sum(segment_times[-1]),60)[1]),'s'
    #print 'Time played ',int(divmod(np.sum(segment_times[:,1]),60)[0]),'m',int(divmod(np.sum(segment_times[:,1]),60)[1]),'s'
    #print 'Beats Played ', np.sum(beat_segments[:,2])    
    #print 'Avg BPM ',np.around(avg_bpm[j],decimals=3)
    #print np.corrcoef(perf_avg_tempo[:(last_beat_played-1)],means_ns)[0,1],np.corrcoef(perf_avg_sone[:(last_beat_played-1)],means_ss)[0,1]
    print 'Tempo',np.around(stats.pearsonr(perf_avg_tempo[:(last_beat_played-1)], means_ns)[0],decimals=3)
    print 'Loudn',np.around(stats.pearsonr(perf_avg_sone[:(last_beat_played-1)], means_ss)[0],decimals=3)
    
    '''times_drill.append(times_d)
    times_smooth.append(times_s)
    print beat_segments[:,2].mean()
    plt.figure()
    plt.hist(beat_segments[:,2],bins=50)
    plt.suptitle(titles[j])'''
#print yo
#AVERAGE TEMPO CALCULATION
axurr[0].set_title('Overview of All Practice Sessions')
axurr[7].set_xlabel('Length in Beats')
fiff.tight_layout()
'''segments = genfromtxt('/Volumes/Untitled/Organised_Practices/Experts/Nika/nika_segments.csv',delimiter=',',dtype=int)
segment_times = genfromtxt('/Volumes/Untitled/Organised_Practices/Experts/Nika/nika_times.csv',delimiter=',')
segment_tempo = genfromtxt('/Volumes/Untitled/Organised_Practices/Experts/Nika/nika_tempo.csv',usecols=(0),delimiter=',')
midi_perf = genfromtxt('/Volumes/Untitled/Organised_Practices/Experts/Nika/nika_midi.csv',delimiter=',',usecols=(0,1,3))
'''
#plt.hist(beat_segments[:,2])

#for i in range(0,len(beat_segments)):
#    index = (segment_tempo >= segment_times[i,0]) & (segment_tempo <= (segment_times[i,0]+segment_times[i,1]))

#midi_perf = np.delete(midi_perf,-1,1)
#midi_perf = np.delete(midi_perf,-2,1)

'''segments = genfromtxt('/Volumes/Untitled/Organised_Practices/Novice/Giulio/giulio_segments.csv',delimiter=',',dtype=int)
segment_times = genfromtxt('/Volumes/Untitled/Organised_Practices/Novice/Giulio/giulio_times.csv',delimiter=',')
segment_tempo = genfromtxt('/Volumes/Untitled/Organised_Practices/Novice/Giulio/giulio_tempo.csv',usecols=(0),delimiter=',')
midi_perf = genfromtxt('/Volumes/Untitled/Organised_Practices/Novice/Giulio/giulio_midi.csv',delimiter=',',usecols=(0,1,3))

segment_tempo[0]=0.000000001

last_beat_played = max(beat_segments[:,1])

large_tempos = np.zeros((len(beat_segments),last_beat_played))
large_sones = np.zeros((len(beat_segments),last_beat_played))

avg_tempos = np.zeros((len(beat_segments),last_beat_played))
avg_sones = np.zeros((len(beat_segments),last_beat_played))

#for i in range(20,21):   
prev_mean = 0 
for i in range(0,len(beat_segments)):
    index = (segment_tempo >= segment_times[i,0]) & (segment_tempo <= (segment_times[i,0]+segment_times[i,1]))
    beats = segment_tempo[index]
    tempo = 60/(beats[1:]-beats[:-1])
    large_tempos[i,beat_segments[i,0]:beat_segments[i,1]]=tempo
    beat_sones = np.zeros(len(tempo))
    for j in range(0,len(tempo)):
        index = (midi_perf[:,0] >= beats[j]) & (midi_perf[:,0] < beats[j+1])
        curr_mean = midi_perf[index,2].mean()
        if len(np.where(index==True)[0])==0:
            curr_mean = prev_mean           
        beat_sones[j] = curr_mean
        prev_mean = curr_mean
    large_sones[i,beat_segments[i,0]:beat_segments[i,1]] = beat_sones
    
    if (beat_segments[i,2]-len(tempo))!=0:
        print i, beat_segments[i,2]-len(tempo), beat_segments[i,2], len(tempo)
        plt.figure()
        plt.plot(tempo)
        plt.suptitle(i)
        time_m = int(segment_times[i,0]/60)
        time_s = int((segment_times[i,0]/60-int(segment_times[i,0]/60))*60)
        print str(time_m)+'.'+str(time_s),segments[i]
        break
    avg_tempo = beat_segments[i,2]/segment_times[i,1]*60
    avg_tempos[i,beat_segments[i,0]:beat_segments[i,1]].fill(avg_tempo)
    
    avg_sone = beat_sones.mean()
    avg_sones[i,beat_segments[i,0]:beat_segments[i,1]].fill(avg_sone)

meanst = np.zeros(last_beat_played)
meanss = np.zeros(last_beat_played)

for i in range(0,last_beat_played):
    beat = i
    non_zero_index = np.flatnonzero(large_tempos[:,beat])
    beats = large_tempos[:,beat]
    beatss = large_sones[:,beat]
    datat = beats[non_zero_index]
    datas = beatss[non_zero_index]
    meanst[i]=datat.mean()
    meanss[i]=datas.mean()
f, ax = plt.subplots(2)
ax[0].plot(np.linspace(0,last_beat_played-1,last_beat_played),meanst)
ax[1].plot(np.linspace(0,last_beat_played-1,last_beat_played),meanss)

means_nt = np.zeros(last_beat_played)
means_ns = np.zeros(last_beat_played)

for i in range(0,last_beat_played):
    beat = i
    non_zero_index = np.flatnonzero(avg_tempos[:,beat])
    beats = avg_tempos[:,beat]
    beatss = avg_sones[:,beat]
    datat = beats[non_zero_index]
    datas = beatss[non_zero_index]
    means_nt[i]=datat.mean()
    means_ns[i]=datas.mean()
    
ax[0].plot(np.linspace(0,last_beat_played-1,last_beat_played),means_nt)
ax[1].plot(np.linspace(0,last_beat_played-1,last_beat_played),means_ns)

meanst = stats.zscore(meanst[1:])
means_nt = stats.zscore(means_nt[1:])

meanss = stats.zscore(meanss[1:])
means_ns = stats.zscore(means_ns[1:])

print np.corrcoef(meanst,means_nt)
print np.corrcoef(meanss,means_ns)'''

'''interesti = interesti[6]
segmentos = beat_segments

for j in range(0,len(interesti)-1):
    f, ax = plt.subplots(2,figsize=(6,4))
    for i in range(0,len(subsec_color)):
        drawRect(subsec_bound[i],subsec_bound[i+1],-1,len(segments),ax[0],subsec_color[i])
        drawRect(subsec_bound[i],subsec_bound[i+1],-1,len(segments),ax[1],subsec_color[i])
    maximum = 0
    beat_segments = segmentos[interesti[j]:interesti[j+1]]
    for i in range(0,len(beat_segments)):
        start_beat = beat_segments[i,0]
        end_beat = beat_segments[i,1]
        seg_len = beat_segments[i,2]
        x = np.linspace(start_beat,end_beat,seg_len*10+1)
        r = float(seg_len)/2
        xc = np.linspace(beat_segments[i,0],beat_segments[i,1],beat_segments[i,2]+1).astype(int)
        yc = np.full(len(xc),i,dtype=int)
        if r >= maximum:
            maximum = r
        mid_x = float(start_beat + float(seg_len)/2)
        #print(mid_x)
        y = np.sqrt(r**2 - (x-mid_x)**2)
        #print(y)
        if seg_len >= run_len:
            colour = ('blue')
            ax[0].plot(xc,yc,color='blue',lw=2,alpha=0.7)
        else:
            colour = ('brown')
            ax[0].plot(xc,yc,color='brown',lw=2,alpha=0.7)
        ax[1].plot(x,y,color=colour,alpha=0.6)
    ax[1].set_ylim(0,maximum+5)
    ax[0].set_ylim(-1,i+1)
    #ax.set_xlim(0,last_beat_played)
    ax[1].set_xlabel('Length in Beats')
    ax[0].get_yaxis().set_visible(False)
    ax[0].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    f.tight_layout()
    #ax[0].set_title('Repetition Arcs for '+str(j+1))
    f.savefig('/Volumes/Untitled/Organised_Practices/Novice/Manos/n2_sess/'+str(j+1)+'.pdf')
    plt.close()'''
    
'''sup = np.zeros((2,100))
sup[1,:]=stats.zscore(np.arange(100))

cluster_this = big_cluster

number_clusters = 2
kmeans = sklearn.cluster.KMeans(n_clusters=number_clusters, init=sup, n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1)
fit = sklearn.cluster.KMeans.fit(kmeans,cluster_this)
fit_predict = sklearn.cluster.KMeans.fit_predict(kmeans,cluster_this)
fit_transform = sklearn.cluster.KMeans.fit_transform(kmeans,cluster_this)
get_params = sklearn.cluster.KMeans.get_params(kmeans)
predict = sklearn.cluster.KMeans.predict(kmeans,cluster_this)
score = sklearn.cluster.KMeans.score(kmeans,cluster_this)
transform = sklearn.cluster.KMeans.transform(kmeans,cluster_this)

cl_centers = kmeans.cluster_centers_
    #dictio = []

#import matplotlib.pyplot as plt

fig, axarr = plt.subplots(2)
fig.suptitle('Practice Pattern')

#plt.show()

for i in range(0,number_clusters):
    cluster = np.where(predict==i)
    print len(cluster[0])
    for j in range(len(cluster[0])):
        axarr[i].plot(cluster_this[cluster[0][j]],alpha=0.2)
    axarr[i].plot(cl_centers[i],color='r',lw=3)
    
X = np.array([[1, 2], [1, 4], [1, 0],
               [4, 2], [4, 4], [4, 0]])'''
'''import powerlaw 
data = beat_segments[:,2] 
results = powerlaw.Fit(data) 
powerlaw.plot_pdf(data)
print results.power_law.alpha 
print results.power_law.xmin 
R, p = results.distribution_compare('power_law', 'lognormal')'''