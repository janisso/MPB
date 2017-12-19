# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 09:26:11 2016

@author: mb
"""
from scipy.interpolate import interp1d
import scipy.ndimage
#import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from operator import itemgetter
from itertools import groupby
from scipy import stats
from collections import Counter
from matplotlib.widgets import SpanSelector
import matplotlib.collections as collections
import sys
import matplotlib.patches as mpatches

import numpy as np
import copy
import os
import music21


#phrases = genfromtxt('/Users/mb/Desktop/Yamaha/new_era/phrases.csv',delimiter=',',dtype=str)
phrases = genfromtxt('/Volumes/MAC_BACKUP/Yamaha/practices/new_anot_study/perf_alphabet/phrases.csv',delimiter=',',dtype=str)
phrasess = np.zeros((len(phrases),4)).astype(int)

plump = music21.converter.parse('/Volumes/MAC_BACKUP/Yamaha/NIME_study/stimuli/nika/mazurka_17_4.xml')

for i in range(0,len(phrases)):
    startBar = int(phrases[i,1].split('.')[0])
    startBeat = int(phrases[i,1].split('.')[1])
    endBar = int(phrases[i,2].split('.')[0])
    endBeat = int(phrases[i,2].split('.')[1])
    phrasess[i]=startBar,startBeat,endBar,endBeat
    
beat_phrasess = np.zeros((len(phrasess),2))
beat_phrasess[:,0]=(phrasess[:,0]-1)*3+(phrasess[:,1]-1)
beat_phrasess[:,1]=(phrasess[:,2]-1)*3+(phrasess[:,3]-1)

'''def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))'''

pp = []

'''def onselect(vmin,vmax):
    print np.round(vmin)+1,np.round(vmax)-1
    b = np.round(vmin)+1
    e = np.round(vmax)-1
    ax7.axhspan(b,e,alpha=2)
    #dc.append([vmin,vmax])'''

class Spans(object):
    def __init__(self):
        self.shift_is_held = False
        self.super_is_held = False
        self.last_span = []
        self.pp = []
        self.z_c = 0
        
    def on_key_press(self,event):
       if event.key == 'shift':
           self.shift_is_held = True
       if event.key == 'super+z':
           self.z_c+=1
           self.last_span[-1].remove()
           self.last_span = self.last_span[:-1]
           self.pp = self.pp[:-1]
           f1.canvas.draw()
    
    def on_key_release(self,event):
       if event.key == 'shift':
           self.shift_is_held = False
    
    def onselect(self,vmin, vmax):
        if self.shift_is_held==True:
            self.last_span.append(ax7.axhspan(np.floor(vmin)+1,np.floor(vmax),color='b',alpha=0.2))
            self.pp.append([int(np.floor(vmin)+1),int(np.floor(vmax)),1])
            print 'Drill-Smooth', np.round(vmin)+1, np.round(vmax)
            f1.canvas.draw()
            self.z_c = 0
            #self.updatePP()
        else:
            self.last_span.append(ax7.axhspan(np.floor(vmin)+1,np.floor(vmax),color='k',alpha=0.2))
            self.pp.append([int(np.floor(vmin)+1),int(np.floor(vmax)),0])
            print 'Drill-Correct', np.round(vmin)+1, np.round(vmax)
            f1.canvas.draw()
            self.z_c = 0
            #self.updatePP()
            
    def updatePP(self):
        for i in range(0,len(self.pp)):
            if self.pp[:,2]==0:
                self.last_span = ax7.axhspan(self.pp[i,0],self.pp[i,1],color='k',alpha=0.2)
            else:
                self.last_span = ax7.axhspan(self.pp[i,0],self.pp[i,1],color='b',alpha=0.2)
                
def workStart(array,work_start,run_len):
    work_count = 0
    for i in range(work_start,len(array)):
        if array[i,2]<run_len:
            #print i
            work_start = i
            work_count = i
            break
    return work_start,work_count

def interpol(x,y,nrp):
    new_x = np.linspace(x[0],x[-1],nrp)
    new_y_interp = interp1d(x, y, kind='zero')
    new_y_interp = new_y_interp(new_x)
    return new_x, new_y_interp
    
def countWork(array,work_start,work_count):
     
    for i in range(work_start+1,len(array)):
        if (
            array[i,0] > (array[work_start,0]-bound) and
            array[i,1] < (array[work_start,1]+bound) and 
            array[i,2] < (array[work_start,2]+bound)
            ):
            work_count+=1
            if work_count!=i:
                next_work_start = i
                #print i,work_count
                break
    return next_work_start,work_count
    
def movingAverage(x,w,thresh):
    buff = np.zeros(w)
    output = np.zeros(len(x))
    std = np.zeros(len(x))
    var = np.zeros(len(x))
    idx = []
    for i in range(len(x)):
        buff[i % w] = x[i]
        output[i]=buff.mean()
        std[i]=buff.std()
        var[i]=buff.var()
        if (x[i] <= (np.percentile(buff,25)+thresh)) and (x[i]>=(np.percentile(buff,25)-thresh)):
            idx.append(i)
    return output,std,np.array(idx),var

def getWork(data):
    ranges = []
    for k, g in groupby(enumerate(data), lambda (i,x):i-x):
        group = map(itemgetter(1), g)
        ranges.append((group[0], group[-1]))
    return ranges

def cusum(x,work_on_this,ax,c):
    o = []
    means = []
    stds = []
    arr = x
    cumSumMean = arr[:10].mean()
    cumSumStd = arr[:10].std()
    for q in range(0,len(x12)):
        if (arr[q]>cumSumMean+4) or (arr[q]< cumSumMean-4) :
            o = []
            bound.append(q-1)
            #for w in range(0,1):
            #    axarr[w].axvline(q-1)
        o.append(arr[q])
        cumSumMean = np.array(o).mean()
        cumSumStd = np.array(o).std()
        means.append(cumSumMean)
        stds.append(cumSumStd)
    #axarr[0].plot(arr)
    #axarr[0].plot(x12A)
    #axarr[1].plot(means)
    #axarr[2].plot(np.diff(means))
    bound.append(len(arr))
    for i in range(1,len(bound)):
        if bound[i]-bound[i-1] >= 2:
            y = arr[bound[i-1]+1:bound[i]]
            x = np.arange(bound[i-1]+1,bound[i-1]+len(y)+1)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            #print slope
            #if slope <= runss and slope >= -runss:
            '''y = mad_based_outlier(y,thresh=1.5)
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)'''
            #if slope <= np.diff(arr).mean()*2 and slope >= -np.diff(arr).mean()*2:
            #    for w in range(0,2):
            ax.axhspan(bound[i-1]+0.75,bound[i]+0.25,facecolor=c, alpha=0.2)
            #        #ax.plot(x,x*slope+intercept,'k',lw=2,alpha=0.5)
            #        #ax.text((bound[i-1]+bound[i])/2,10,str(np.around(slope,decimals=2)))


def delBelowAbove(part,delAbove,delBelow):
    i = 0
    bigOffset = 0
    thisOffset = 0
    for element in part.recurse():
        if isinstance(element, music21.stream.Measure)==True:
            bigOffset = element.offset
            #print element.offset
        if isinstance(element, music21.stream.Voice):# continu
            #print 'Im voice',bigOffset,element.offset
            if isinstance(element, music21.note.Note):# continue
                 thisOffset = bigOffset+element.offset
                 if (thisOffset < delBelow) or (thisOffset >= delAbove):
                     element.activeSite.remove(element)
                 if isinstance(element, music21.chord.Chord)==True:
                    thisOffset = bigOffset+element.offset
                    if (thisOffset < delBelow) or (thisOffset >= delAbove):
                        element.activeSite.remove(element)
        if isinstance(element, music21.note.Note)==True:
            thisOffset = bigOffset+element.offset
            if (thisOffset < delBelow) or (thisOffset >= delAbove):
                element.activeSite.remove(element)
        if isinstance(element, music21.chord.Chord)==True:
            thisOffset = bigOffset+element.offset
            if (thisOffset < delBelow) or (thisOffset >= delAbove):
                element.activeSite.remove(element)
        i+=1
    return part
					

def segmentScore(score,phrase):
    
    phrases_bar_start = int(phrase[0])
    phrases_bar_end = int(phrase[2])

    measure_count = map(int, list(np.linspace(phrases_bar_start,phrases_bar_end,(phrases_bar_end-phrases_bar_start+1))))
    print measure_count
    rh = music21.stream.Stream()
    lh = music21.stream.Stream()
    
    for i in range(0,len(measure_count)):
        smallRh = score.parts[0].measure(measure_count[i])
        smallLh = score.parts[1].measure(measure_count[i])
        rh.append(smallRh)
        lh.append(smallLh)

    ts = music21.meter.TimeSignature('3/4')
    
    rh.insert(0, ts)
    #lh.insert(0, ts)
    c1 = music21.clef.BassClef()
    lh.insert(0, c1)
    
    delAbove = (phrases_bar_end-phrases_bar_start)*3+(int(phrase[3])-1)+1
    delBelow = int(phrase[1])-1
    
    #print 'delAbove = ',delAbove,'delBelow = ',delBelow    
    
    #delBelowAbove(rh,delAbove,delBelow)
    #delBelowAbove(lh,delAbove,delBelow)
    
    sNew = music21.stream.Score()    
    sNew.insert(rh)
    sNew.insert(lh)
    sNew.insert(0,ts)
    sNew.insert(0, music21.metadata.Metadata())
    sNew.metadata.title = 'Mazurka Op. 17 No. 4 --- ' + str(phrase[0]) + ' ' + str(phrase[1]) + '-' +str(phrase[2]) + ' ' + str(phrase[3])
    sNew.metadata.composer = 'F.Chopin'
    return sNew#, rh,lh


ds = []

bigStarts = []
bigEnds = []

e1 = genfromtxt(path_ting + '/Experts/nika/e1_clusters.csv',delimiter=',')
e2 = genfromtxt(path_ting + '/Experts/petr/e2_clusters.csv',delimiter=',')
i1 = genfromtxt(path_ting + '/Intermediate/janis/i1_clusters.csv',delimiter=',')
i2 = genfromtxt(path_ting + '/Intermediate/shch/i2_clusters.csv',delimiter=',')
i3 = genfromtxt(path_ting + '/Intermediate/stratos/i3_clusters.csv',delimiter=',')
i4 = genfromtxt(path_ting + '/Intermediate/katerina/i4_clusters.csv',delimiter=',')
n1 = genfromtxt(path_ting + '/Novice/giulio/n1_clusters.csv',delimiter=',')
n2 = genfromtxt(path_ting + '/Novice/manos/n2_clusters.csv',delimiter=',')

pp = [e1,i1,i2,i3,i4,n1,n2,e2]
cs = []

drill_counter = 0
run_counter = 0

buffSize = 4
thresh = 15
for m in range(5,6):
    for u in range(0,8):
        run_len = 10
        work_on_this = copy.deepcopy(yos[u])
        thresh = work_on_this[:,2].mean()
        runss = []
        #fig, ax = plt.subplots(figsize=(6,8))
        # Two subplots, unpack the axes array immediately
        f1, ax7 = plt.subplots()#figsize=(7,8))
        '''f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
        ax1.plot(x, y)
        ax1.set_title('Sharing Y axis')
        ax2.scatter(x, y)'''
        for i in range(0,len(subsec_color)):
            '''drawRect(subsec_bound[i],subsec_bound[i+1],-1,len(work_on_this),ax1,subsec_color[i])
            drawRect(subsec_bound[i],subsec_bound[i+1],-1,len(work_on_this),ax2,subsec_color[i])
            drawRect(subsec_bound[i],subsec_bound[i+1],-1,len(work_on_this),ax3,subsec_color[i])'''
            drawRect(subsec_bound[i],subsec_bound[i+1],-1,len(work_on_this),ax7,subsec_color[i])
            #drawRect(subsec_bound[i],subsec_bound[i+1],-1,len(work_on_this),ax9,subsec_color[i])
        for i in range(0,len(work_on_this)):
            x = np.linspace(work_on_this[i,0],work_on_this[i,1],work_on_this[i,2]+1).astype(int)
            y = np.full(len(x),i,dtype=int)
            if (work_on_this[i,2])>=run_len:
                '''ax1.plot(x,y,color='blue',lw=2,alpha=0.7)
                ax2.plot(x,y,color='blue',lw=2,alpha=0.7)
                ax3.plot(x,y,color='blue',lw=2,alpha=0.7)'''
                if run_counter == 0:
                    ax7.plot(x,y,color='blue',lw=2,alpha=0.7,label='Run')
                    runss.append(work_on_this[i,2])
                    run_counter+=1
                else:
                    ax7.plot(x,y,color='blue',lw=2,alpha=0.7)
                    runss.append(work_on_this[i,2])
        
            else:
                '''ax1.plot(x,y,color='brown',lw=2,alpha=0.7)
                ax2.plot(x,y,color='brown',lw=2,alpha=0.7)
                ax3.plot(x,y,color='brown',lw=2,alpha=0.7)'''
                if drill_counter == 0:
                    ax7.plot(x,y,color='brown',lw=2,alpha=0.7,label='Drill')
                    drill_counter += 1
                else:
                    ax7.plot(x,y,color='brown',lw=2,alpha=0.7)
        ax7.set_title('Practice Session Map')
        ax7.set_xlabel('Score Time in Beats')
        ax7.set_ylabel('Practice Segment Number')
        ax7.set_ylim(0,len(work_on_this))
        ax7.set_xlim(0,396)
        box = ax7.get_position()
        ax7.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax7.legend(loc='upper left', bbox_to_anchor=(1, 1))
        #f1.tight_layout()
        
        '''ax1.set_ylim(-1,len(work_on_this)+1)
        ax2.set_ylim(-1,len(work_on_this)+1)
        ax3.set_ylim(-1,len(work_on_this)+1)
        sp = genfromtxt('/Users/mb/Desktop/Yamaha/new_era/journal/manual_analysis/i1.csv',delimiter=',',dtype=int)+1

        #ICLUDE START
        cont = []
        prev_arr = work_on_this[0,0]
        for i in range(1,len(work_on_this)):
            curr_arr = work_on_this[i,0]
            if curr_arr == prev_arr:
                cont.append(i)
            prev_arr = curr_arr
        
        ranges = getWork(cont)
        for i in range(len(ranges)):
            ax1.axhspan(ranges[i][0]-1.25,ranges[i][1]+0.25,alpha=0.2,color='green')
            #ax7.axhspan(ranges[i][0]-1.25,ranges[i][1]+0.25,alpha=0.2,color='green')
        
        #ICLUDE end
        cont = []
        prev_arr = work_on_this[0,1]
        for i in range(1,len(work_on_this)):
            curr_arr = work_on_this[i,1]
            if curr_arr == prev_arr:
                cont.append(i)
            prev_arr = curr_arr
        
        ranges = getWork(cont)
        for i in range(len(ranges)):
            ax2.axhspan(ranges[i][0]-1.25,ranges[i][1]+0.25,alpha=0.2,color='red')
            #ax7.axhspan(ranges[i][0]-1.25,ranges[i][1]+0.25,alpha=0.2,color='red')'''
        
        #ICLUDE THE BEAT
        cont = []
        prev_arr = np.arange(work_on_this[0,0],work_on_this[0,1])
        for i in range(1,len(work_on_this)):
            curr_arr = np.arange(work_on_this[i,0],work_on_this[i,1])
            same_vals = np.intersect1d(prev_arr,curr_arr)
            if len(same_vals)==0:
                prev_arr = curr_arr
            else:
                prev_arr = same_vals
                '''for hum in range(0,1):
                    prev_arr = np.insert(prev_arr,0,prev_arr[0]-1)
                    prev_arr =np.insert(prev_arr,len(prev_arr),prev_arr[-1]+1)'''
                cont.append(i)
        
        ranges = getWork(cont)
        '''for i in range(len(ranges)):
            ax3.axhspan(ranges[i][0]-1.25,ranges[i][1]+0.25,alpha=0.2,color='blue')
            #ax7.axhspan(ranges[i][0]-1.25,ranges[i][1]+0.25,alpha=0.2,color='blue')
            #ax7.text(5,(ranges[i][0]-1.25+ranges[i][1]+0.25)/2,str(i+1))
        ax1.set_ylabel('Practice Segment')
        ax1.set_xlabel('Length in Beats')
        ax2.set_xlabel('Length in Beats')
        ax3.set_xlabel('Length in Beats')
        #        ax1.set_ylim(250,280)
        #        ax1.set_xlim(175,280)
        #        ax2.set_xlim(175,280)
        #        ax3.set_xlim(175,280)
        f.suptitle('Rule Based Segmentation Comparison')
        f.tight_layout()'''
        
        '''f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
        for i in range(0,len(subsec_color)):
            drawRect(subsec_bound[i],subsec_bound[i+1],-1,len(work_on_this),ax1,subsec_color[i])
            drawRect(subsec_bound[i],subsec_bound[i+1],-1,len(work_on_this),ax2,subsec_color[i])
            drawRect(subsec_bound[i],subsec_bound[i+1],-1,len(work_on_this),ax3,subsec_color[i])
        for i in range(0,len(work_on_this)):
            x = np.linspace(work_on_this[i,0],work_on_this[i,1],work_on_this[i,2]+1).astype(int)
            y = np.full(len(x),i,dtype=int)
            if (work_on_this[i,2])>=run_len:
                ax1.plot(x,y,color='blue',lw=2,alpha=0.7)
                ax2.plot(x,y,color='blue',lw=2,alpha=0.7)
                ax3.plot(x,y,color='blue',lw=2,alpha=0.7)
                runss.append(work_on_this[i,2])
            else:
                ax1.plot(x,y,color='brown',lw=2,alpha=0.7)
                ax2.plot(x,y,color='brown',lw=2,alpha=0.7make run)
                ax3.plot(x,y,color='brown',lw=2,alpha=0.7)
        
        #        ax1.set_xlim(175,280)
        #        ax2.set_xlim(175,280)
        #        ax3.set_xlim(175,280)
        
        x1 = work_on_this[:,0]
        x2 = work_on_this[:,1]
        x12 = (x1+x2)/2
        #cusum(x1,work_on_this,ax1,'green')
        #cusum(x2,work_on_this,ax2,'red')
        #cusum(x12,work_on_this,ax3,'blue')
        ax1.set_ylabel('Practice Segment')
        ax1.set_xlabel('Length in Beats')
        ax2.set_xlabel('Length in Beats')
        ax3.set_xlabel('Length in Beats')
        #ax1.set_ylim(250,280)
        f.suptitle('Stats Based Segmentation Comparison')
        f.tight_layout()
      
        #idx = x12std <= 4
        y = np.arange(len(work_on_this))
        #pim = work_on_this[work_ind,2].std()
        x1 = work_on_this[:,0]
        #x1A,x1std,x1idx,x1var = movingAverage(x1,buffSize,int(pim))
        x2 = work_on_this[:,1]
        #x2A,x2std,x2idx,x2var = movingAverage(x2,buffSize,int(pim))
        x12 = (x1+x2)/2'''
        #x12A,x12std,x12idx,x12var = movingAverage(x12,buffSize,int(pim))
        ranges = copy.deepcopy(pp[u])
        #for go in range(0,len(phrasess)):
        #    li = (phrasess[go,0]-1)*3 + (phrasess[go,1]-1)
        #    ax7.axvline(li)
        css = np.array([])
        for i in range(0,len(ranges)):
            check_these = work_on_this[ranges[i,0]:ranges[i,1]+1]
            #check_these[:,1]=check_these[:,1]
            x = np.arange(ranges[i,0],ranges[i,1]+1)
            css = np.hstack((css,x))
            #if len(check_these)>2:
            #plt.figure()
            #plt.plot(check_these[:,0],'g')
            #plt.plot(check_these[:,1],'r')
            #plt.plot(check_these[:,2]+check_these[:,0],'b')
            #plt.axhline(np.median(check_these[:,0:2]))
            b = Counter(check_these[:-1,0]).most_common(1)[0][0]
            bar = b/3+1
            beat = b%3+1
            if ranges[i,2]==0:
                co = 'brown'
            else:
                co = 'blue'
            ax7.axhspan(ranges[i,0]-0.25,ranges[i,1]+0.25,alpha=0.2,color=co)
            #ax9.axhspan(ranges[i,0]-0.25,ranges[i,1]+0.25,alpha=0.2,color=co)
            ax7.plot(np.full(len(x),b),x,'r',lw=2)
            #ax7.text(np.median(check_these[:,0:2]),x.mean(), str(bar) + ' ' + str(beat))
            #ax7.text(np.median(check_these[:,0:2]),x.mean(),str(bar)+' '+str(beat))
            #print np.median(check_these[:,2]),check_these[-1,2]
            #                plt.figure()
            #                for j in range(0,len(check_these)):
            #                    x = np.linspace(check_these[j,0],check_these[j,1],check_these[j,2]+1).astype(int)
            #                    y = np.full(len(x),j,dtype=int)
            #                    if (check_these[j,2])>=run_len:
            #                        plt.plot(x,y,color='blue',lw=2,alpha=0.7)                
            #                    else:
            #                        plt.plot(x,y,color='brown',lw=2,alpha=0.7)
            #                    plt.xlim(check_these[:,0].min()-0.1,check_these[:,1].max()+0.1)
            #                    plt.ylim(-0.1,len(check_these)-0.9)
            #                    plt.suptitle(str(check_these[:,0].min()/3+1),str(check_these[:,0].min()%3+1))
            '''phrase_begin_idx = np.where((check_these[:,0].min() >= beat_phrasess[:,0]) & (check_these[:,0].min() <= beat_phrasess[:,1]))[0][0]
            phrase_end_idx = np.where((check_these[:,1].max() >= beat_phrasess[:,0]) & (check_these[:,1].max() <= beat_phrasess[:,1]))[0][0]
            phrase = np.zeros(4).astype(int)
            beat_phrase = np.zeros(2).astype(int)
            phrase[:2]=phrasess[phrase_begin_idx][:2]
            phrase[2:]=phrasess[phrase_end_idx][2:]
            beat_phrase[0]=beat_phrasess[phrase_begin_idx][0]
            beat_phrase[1]=beat_phrasess[phrase_end_idx][1]
            #print phrase
            score = copy.deepcopy(plump)
            sNew = segmentScore(score,phrase)
            #sNew.show()
            fPath =  '/Users/mb/Desktop/Yamaha/new_era/journal/manual_analysis/work/scores/'+str(u)+'/cluster_'+str(i)
            if not os.path.exists(fPath):
                os.makedirs(fPath)
            xmlPath = fPath+'/cluster_'+str(i)+'.xml'
            gmnPath = fPath+'/cluster_'+str(i)+'.gmn'
            sNew.write('xml', xmlPath)
            os.system('/Users/mb/xml2guido '+ xmlPath+ ' 5> ' + gmnPath)
            file = open(fPath+'/cluster_'+str(i)+'.txt','w')
            file.write('Practice Segments\n')
            file.write(str(check_these)+'\n')
            file.write('Figural Groups\n')
            file.write(str(beat_phrase)) 
            file.close()'''
        brown_patch = mpatches.Patch(color = 'brown',alpha = 0.2, label = 'Drill-Correct')
        blue_patch = mpatches.Patch(color = 'blue',alpha = 0.2, label = 'Drill-Smooth')
        #ax7.plot([-1,-2,-3,-4,-5,-6],[-1,-2,-3,-4,-5,-6],color='brown',alpha=0.2,label='Drill Correct')
        #ax7.plot([-1,-2,-3,-4,-5,-6],[-1,-2,-3,-4,-5,-6],color='blue',alpha=0.2,label='Drill Smooth')
        ax7.set_ylim(0,len(work_on_this)+1)
        ax7.set_xlim(0,397)
        ax7.set_xlabel('Beats in Score Time')
        ax7.set_ylabel('Practice Segment Number')
        #ax7.legend(handles = [brown_patch,blue_patch])
        #f1.tight_layout()
        #ax7.set_title('Drill-Correct and Drill-Smooth Clusters for '+ titles[u])
        ax7.set_title('Memorisation Strategy for '+titles[u])
        cs.append(css)
        '''bbb = Spans()
        f1.canvas.mpl_connect('key_press_event', bbb.on_key_press)
        f1.canvas.mpl_connect('key_release_event', bbb.on_key_release)
        span = SpanSelector(ax7, bbb.onselect, 'vertical')
        #np.savetxt('/Users/mb/Desktop/Yamaha/new_era/journal/Organised_Practices/Experts/petr/e2_clusters.csv',save_this,fmt='%i',delimiter=',')'''

#-------------DC vs DS------------
'''e1 = genfromtxt('/Users/mb/Desktop/Yamaha/new_era/journal/Organised_Practices/Experts/nika/e1_clusters.csv',delimiter=',')
e2 = genfromtxt('/Users/mb/Desktop/Yamaha/new_era/journal/Organised_Practices/Experts/petr/e2_clusters.csv',delimiter=',')
i1 = genfromtxt('/Users/mb/Desktop/Yamaha/new_era/journal/Organised_Practices/Intermediate/janis/i1_clusters.csv',delimiter=',')
i2 = genfromtxt('/Users/mb/Desktop/Yamaha/new_era/journal/Organised_Practices/Intermediate/shch/i2_clusters.csv',delimiter=',')
i3 = genfromtxt('/Users/mb/Desktop/Yamaha/new_era/journal/Organised_Practices/Intermediate/stratos/i3_clusters.csv',delimiter=',')
i4 = genfromtxt('/Users/mb/Desktop/Yamaha/new_era/journal/Organised_Practices/Intermediate/katerina/i4_clusters.csv',delimiter=',')
n1 = genfromtxt('/Users/mb/Desktop/Yamaha/new_era/journal/Organised_Practices/Novice/giulio/n1_clusters.csv',delimiter=',')
n2 = genfromtxt('/Users/mb/Desktop/Yamaha/new_era/journal/Organised_Practices/Novice/manos/n2_clusters.csv',delimiter=',')

pp = [e1,e2,i1,i2,i3,i4,n1,n2]

np.where(e1[:,2]==0)
tots = np.zeros(8)
dc = np.zeros(8)

for i in range(len(pp)):
    tots[i]=len(pp[i])
    dc[i]=len(np.where(pp[i][:,2]==0)[0])

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)

labels = ['E-1', 'E-2', 'I-1', 'I-2', 'I-3','I-4','N-1','N-2']
#drills = 100*(dc/tots)
runs = tots-dc
ind = np.arange(8)    # the x locations for the groups
width = 0.8      # the width of the bars: can also be len(x) sequence
p1 = ax.bar(ind, dc, width, color='brown',label='DC',alpha=0.5)
p2 = ax.bar(ind, runs, width, color='blue',
             bottom=dc,label='DS',alpha=0.5)
ax.set_ylabel('Percentage of number of DC vs DS clusters')
ax.set_xlabel('Subject')
ax.set_title('Ratio of DC to DS')
ax.legend()
ax.set_xticks(ind + width/2.)
#ax.set_yticks(np.arange(0, 81, 10))
ax.set_xticklabels((labels))
ax.legend(loc='lower center', bbox_to_anchor=(1, 0.5))
mpl_fig.canvas.draw()'''

#-------------Segments in Clusters------------
#np.where(e1[:,2]==0)
tots = np.array([len(yos[0]),len(yos[7]),len(yos[1]),len(yos[2]),len(yos[3]),len(yos[4]),len(yos[5]),len(yos[6])]).astype(float)
dc = np.array([len(cs[0]),len(cs[7]),len(cs[1]),len(cs[2]),len(cs[3]),len(cs[4]),len(cs[5]),len(cs[6])])

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)

labels = ['E-1', 'E-2', 'I-1', 'I-2', 'I-3','I-4','N-1','N-2']
drills = dc
runs = tots-drills
ind = np.arange(8)    # the x locations for the groups
width = 0.8      # the width of the bars: can also be len(x) sequence
p1 = ax.bar(ind, drills, width, color='brown',label='Work',alpha=0.5)
p2 = ax.bar(ind, runs, width, color='blue',
             bottom=drills,label='Other',alpha=0.5)
ax.set_ylabel('Percentage of Segments in Clusters')
ax.set_xlabel('Subject')
ax.set_title('Ratio of Work to Other')
ax.legend()
ax.set_xticks(ind + width/2.)
#ax.set_yticks(np.arange(0, 81, 10))
ax.set_xticklabels((labels))
ax.legend(loc='lower center', bbox_to_anchor=(1, 0.5))
mpl_fig.canvas.draw()