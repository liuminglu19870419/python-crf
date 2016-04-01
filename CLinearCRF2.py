# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:31:31 2014

@author: Huang,Zheng

Linear CRF in Python

License (BSD)
==============
Copyright (c) 2013, Huang,Zheng.  huang-zheng@sjtu.edu.cn
All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import codecs
import ctypes
import datetime
from multiprocessing import Process, Queue
import multiprocessing
import numpy
import os
import re
from scipy.misc import logsumexp
import sys


_gradient = None  #  global variable used to store the gradient calculated in Liklihood function.

def logdotexp_vec_mat(loga,logM):
        return logsumexp(loga+logM,axis=1)
        
def logdotexp_mat_vec(logM,logb):
    return logsumexp(logM+logb[:,numpy.newaxis],axis=0)

def validTemplateLine(strt):
    ifvalid=True
    if strt.count("[")!=strt.count("]"):
        ifvalid=False
    if "UuBb".find(strt[0])==-1:
        ifvalid=False
    if ifvalid==False:
        print "error in template file:", strt
    return ifvalid

def readData(dataFile):
    """
    read the train data from data file
    """
    texts = []
    labels = []
    text = []
    label=[]
    obydic=dict()
    file = codecs.open(dataFile, 'r')  #  default encoding.
    obyid=0
    linecnt=0
    spacecnt=0
    for line in file:
        #print line
        line = line.strip()
        if len(line) == 0:
            if len(text)>0:
                texts.append(text)
                labels.append(label)
            text = []
            label = []
        else:
            linecnt+=1
            if linecnt % 10000 == 0 :
                print "read ",linecnt , " lines."
            chunk = line.split()
            if spacecnt==0:
                spacecnt=len(chunk)
            else:
                if len(chunk)!=spacecnt:
                    print "Error in input data:",line
            text.append(chunk[0:-1])
            ylabel=chunk[-1]
            if obydic.has_key(ylabel)==False:
                obydic[ylabel]=obyid
                label.append(obyid)
                obyid+=1
            else:
               label.append(obydic[ylabel])
              
    if len(text)>0:  # sometimes, there is no empty line at the end of file.
        texts.append(text)
        labels.append(label)
    
    #print texts,labels
    texts=texts
    oys=labels
    seqnum=len(oys)
    seqlens=[len(x) for x in texts]
    K = len(obydic)
    y2label = dict([(obydic[key],key) for key in obydic.keys()])
    print "number of labels:", K
    return texts,seqlens,oys,seqnum,K,obydic,y2label
   
def readTemplate(tmpFile):
    """
    read the template from tmpFile
    """
    tlist=[]  # list of list(each template line)  
    file = codecs.open(tmpFile, 'r')  #  default encoding.
    #repat=r'\[\d+,\d+\]'
    repat=r'\[-?\d+,-?\d+\]'    #-?[0-9]*
    for line in file:
        #print line    
        line=line.strip()
        if len(line)==0:
            continue
        if line[0]=="#":  # not comment line
            continue
        fl=line.find("#")
        if fl!=-1:  # remove the comments in the same line.
            line=line[0:fl]
        if validTemplateLine(line)==False:
            continue
        fl=line.find(":")
        if fl!=-1:  # just a symbol
            eachlist=[line[0:fl]]
        else:
            eachlist=[line[0]]
            
        for a in list(re.finditer(repat, line)):
            locstr=line[a.start()+1:a.end()-1]
            loc = locstr.split(",")
            eachlist.append(loc) 
            #print a.start(),a.end()
        tlist.append(eachlist)
    print "Valid Template Line Number:",len(tlist)
    return tlist


def expandOBX(texts,seqid,locid,tp):  # expend the observation at locid for sequence(seqid)
    strt=tp[0]
    for li in tp[1::]:
        row=locid+int(li[0]); col=int(li[1])
        if row>=0 and row<len(texts[seqid]):
            if col>=0 and col<len(texts[seqid][row]):
                strt+= ":" + texts[seqid][row][col]
    #print strt
    return strt       

    
def processFeatures(tplist,texts,seqnum,K,fd=1):
    """
    计算feature个数,删除出现次数较少的feature,并生成feature的排序
    """
    uobxs =  dict(); bobxs=dict() ; 
    '''add feature reduction here'''
    for ti,tp in enumerate(tplist):  # for each template line
        for sid in range(seqnum):  # for each traning sequence.
            for lid in range(len(texts[sid])):
                obx=expandOBX(texts,sid,lid,tp)
                if obx[0]=="B":
                    if bobxs.has_key(obx)==False:
                        bobxs[obx]=1
                    else:
                        tval= bobxs[obx]
                        bobxs[obx]= tval+1
                        
                if obx[0]=="U":
                    if uobxs.has_key(obx)==False:
                        uobxs[obx]=1
                    else:
                        tval= uobxs[obx]
                        uobxs[obx]= tval+1
                        
    if fd>=2:  # need to remove ub frequency less than fd
        uobxnew = { k : v for k,v in uobxs.iteritems() if v >= fd }
        bobxnew = { k : v for k,v in bobxs.iteritems() if v >= fd }
        del uobxs; del bobxs;
        uobxs,bobxs=uobxnew,bobxnew
    
    ufnum, bfnum = 0 , 0                    
    #猜测可能是用来计算特征id
    for obx in bobxs.keys():
        bobxs[obx]=bfnum
        bfnum+=K*K
    for obx in uobxs.keys():
        uobxs[obx]=ufnum
        ufnum+=K
    return uobxs,bobxs,ufnum,bfnum

def calObservexOn(tplist,texts,uobxs,bobxs,seqnum):
    '''speed up the feature calculation
      calculate the on feature functions
      验证每一个训练数据符合那些feature
       ''' 
    uon=[]; bon=[]
    for sid in range(seqnum):  # for each traning sequence.
        sequon=[];seqbon=[]
        for lid in range(len(texts[sid])):
            luon=[];lbon=[]
            for ti,tp in enumerate(tplist):  # for each template line
                obx=expandOBX(texts,sid,lid,tp)
                #skey = str(ti)+":"+str(sid)+":"+str(lid)
                #obx=oball[skey]
                if tp[0][0]=="B":
                    fid=bobxs.get(obx)
                    #print fid
                    if fid!=None:
                        lbon.append(fid)
                if tp[0][0]=="U":
                    fid=uobxs.get(obx)
                    if fid!=None:
                        luon.append(fid)
            sequon.append(luon);seqbon.append(lbon)
        uon.append(sequon);bon.append(seqbon)
    return uon,bon

def calObservexOnLoc(uon,bon,seqnum,mp):
    '''speed up the feature calculation (Mulitprocessing)
      calculate the on feature list and location ''' 
    ulen=0 ; loclen=0
    for a in uon:
        loclen+=len(a)
        for b in a:
            ulen+=len(b)
    #ulen = sum[(len(b)) for b in a for a in uon]
    if sys.platform=="win32" and mp==1:   # windows system need shared memory to do multiprocessing
        uonarr=multiprocessing.Array('i', ulen)
        uonseqsta=multiprocessing.Array('i', seqnum)
        uonlocsta=multiprocessing.Array('i', loclen)
        uonlocend=multiprocessing.Array('i', loclen)     
    else:
        uonarr=numpy.zeros((ulen),dtype=numpy.int)
        uonseqsta=numpy.zeros((seqnum),dtype=numpy.int)
        uonlocsta=numpy.zeros((loclen),dtype=numpy.int)
        uonlocend=numpy.zeros((loclen),dtype=numpy.int)
        
    uid=0 ; seqi = 0 ; loci=0
    for seq in uon:  # for each traning sequence.
        uonseqsta[seqi]=loci
        for loco in seq:
            uonlocsta[loci]=uid
            for aon in loco:
                uonarr[uid]=aon
                uid+=1
            uonlocend[loci]=uid
            loci+=1
        seqi+=1
        
    blen=0 ; loclen=0
    for a in bon:
        loclen+=len(a)
        for b in a:
            blen+=len(b)
    #ulen = sum[(len(b)) for b in a for a in uon]
            
    if sys.platform=="win32" and mp==1:   # windows system need shared memory to do multiprocessing
        bonarr=multiprocessing.Array('i', ulen)
        bonseqsta=multiprocessing.Array('i', seqnum)
        bonlocsta=multiprocessing.Array('i', loclen)
        bonlocend=multiprocessing.Array('i', loclen)     
    else:
        bonarr=numpy.zeros((ulen),dtype=numpy.int)
        bonseqsta=numpy.zeros((seqnum),dtype=numpy.int)
        bonlocsta=numpy.zeros((loclen),dtype=numpy.int)
        bonlocend=numpy.zeros((loclen),dtype=numpy.int)
        
    bid=0 ; seqi = 0 ; loci=0
    for seq in bon:  # for each traning sequence.
        bonseqsta[seqi]=loci
        for loco in seq:
            bonlocsta[loci]=bid
            for aon in loco:
                bonarr[bid]=aon
                bid+=1
            bonlocend[loci]=bid
            loci+=1
        seqi+=1
    #print len(uonlocsta)
    return uonarr,uonseqsta,uonlocsta,uonlocend, bonarr,bonseqsta,bonlocsta,bonlocend

def calFSS(texts,oys,uon,bon,ufnum,bfnum,seqnum,K,y0):
    fss=numpy.zeros((ufnum+bfnum))
    fssb=fss[0:bfnum]
    fssu=fss[bfnum:]
    for i in range(seqnum):
        for li in range(len(texts[i])):
            for ao in uon[i][li]:
                fssu[ao+oys[i][li]]+=1.0
            for ao in bon[i][li]:
                if li==0:  # the first , yt-1=y0
                    fssb[ao+oys[i][li]*K+y0]+=1.0
                else:
                    fssb[ao+oys[i][li]*K+oys[i][li-1]]+=1.0
    return fss

def random_param(ufnum,bfnum):
    #theta=numpy.random.randn(ufnum+bfnum)
    theta=numpy.ones(ufnum+bfnum)
    return theta

def regularity(theta,type=0,sigma=1.0):
    if type == 0:
        regularity = 0
    elif type == 1:
        regularity = numpy.sum(numpy.abs(theta)) / sigma
    else:
        v = sigma ** 2
        v2 = v * 2
        regularity = numpy.sum(numpy.dot(theta,theta) )/ v2
    return regularity

def regularity_deriv(theta,type=0,sigma=1.0):
    if type == 0:
        regularity_deriv = 0
    elif type == 1:
        regularity_deriv = numpy.sign(theta) / sigma
    else:
        v = sigma ** 2
        regularity_deriv = theta / v
    return regularity_deriv

def logMarray(seqlen,auon,abon,K, thetau,thetab):
    ''' logMlist (n, K, K ) --> (sequence length, Yt, Yt-1)'''
    mlist=[]
    for li in range(seqlen):
        fv = numpy.zeros((K,K))
        for ao in auon[li]:
            fv+=thetau[ao:ao+K][:,numpy.newaxis]
        for ao in abon[li]:
            fv+=thetab[ao:ao+K*K].reshape((K,K))
        mlist.append(fv)
    
    for i in range(0,K):  # set the energe function for ~y(0) to be -inf.
        mlist[0][i][1:]= - float("inf")
    #print "mlist:",mlist
    return mlist      

def logM_sa(seqlen,seqid,uonarr,uonseqsta,uonlocsta,uonlocend,
                                bonarr,bonseqsta,bonlocsta,bonlocend,K, thetau,thetab):
    ''' logMlist (n, K, K ) --> (sequence length, Yt, Yt-1)'''
    mlist=[]
    iloc = uonseqsta[seqid]
    ilocb = bonseqsta[seqid]
    for li in range(seqlen):
        fv = numpy.zeros((K,K))
        for i in range(uonlocsta[iloc+li],uonlocend[iloc+li]):
            ao=uonarr[i]
            fv+=thetau[ao:ao+K][:,numpy.newaxis]
        for i in range(bonlocsta[ilocb+li],bonlocend[ilocb+li]):
            ao=bonarr[i]
            fv+=thetab[ao:ao+K*K].reshape((K,K))
        mlist.append(fv)
    
    for i in range(0,K):  # set the energe function for ~y(0) to be -inf.
        mlist[0][i][1:]= - float("inf")
    #print "mlist:",mlist
    return mlist      

def logAlphas(Mlist):
    logalpha = Mlist[0][:,0] # alpha(1)
    logalphas = [logalpha]
    for logM in Mlist[1:]:
        logalpha = logdotexp_vec_mat(logalpha, logM)
        logalphas.append(logalpha)
    #print "logalphas:",logalphas
    return logalphas
    
def logBetas(Mlist):
    logbeta = numpy.zeros_like(Mlist[-1][:, 0])
    logbetas = [logbeta]
    for logM in Mlist[-1:0:-1]:
        logbeta = logdotexp_mat_vec(logM, logbeta)
        logbetas.append(logbeta)
    #print "logbeta:",logbetas[::-1]
    return logbetas[::-1]

def likelihood_sa(seqlens,fss,uonarr,uonseqsta,uonlocsta,uonlocend,
                                bonarr,bonseqsta,bonlocsta,bonlocend,theta,seqnum,K,ufnum,bfnum,regtype,sigma):
    global _gradient
    grad = numpy.array(fss,copy=True)  # data distribuition
    gradb=grad[0:bfnum]
    gradu=grad[bfnum:]
    thetab=theta[0:bfnum]
    thetau=theta[bfnum:]
    likelihood = numpy.dot(fss,theta)
    for si in range(seqnum):
        logMlist = logM_sa(seqlens[si],si,uonarr,uonseqsta,uonlocsta,uonlocend,
                                bonarr,bonseqsta,bonlocsta,bonlocend,K,thetau,thetab)
        logalphas = logAlphas(logMlist)
        logbetas = logBetas(logMlist)
        logZ = logsumexp(logalphas[-1])
        likelihood -= logZ
        expect = numpy.zeros((K,K))
        for i in range(len(logMlist)):
            if i == 0:
                expect = numpy.exp(logMlist[0] + logbetas[i][:,numpy.newaxis] - logZ)
            elif i < len(logMlist) :
                expect = numpy.exp(logMlist[i] + logalphas[i-1][numpy.newaxis,: ] + logbetas[i][:,numpy.newaxis] - logZ)
            #print "expect t:",i, "expect: ", expect
            p_yi=numpy.sum(expect,axis=1)
            # minus the parameter distribuition
            iloc = uonseqsta[si]
            for it in range(uonlocsta[iloc+i],uonlocend[iloc+i]):            
                ao=uonarr[it]
                gradu[ao:ao+K] -= p_yi
                
            iloc = bonseqsta[si]    
            for it in range(bonlocsta[iloc+i],bonlocend[iloc+i]):            
                ao=bonarr[it]
                gradb[ao:ao+K*K] -= expect.reshape((K*K))
    grad -= regularity_deriv(theta,regtype,sigma)
    _gradient = grad
    return likelihood - regularity(theta,regtype,sigma)

def likelihood(seqlens,fss,uon,bon,theta,seqnum,K,ufnum,bfnum,regtype,sigma):
    global _gradient
    grad = numpy.array(fss,copy=True)  # data distribuition
    likelihood = numpy.dot(fss,theta)
    gradb=grad[0:bfnum]
    gradu=grad[bfnum:]
    thetab=theta[0:bfnum]
    thetau=theta[bfnum:]
    #likelihood = numpy.dot(fss,theta)
    for si in range(seqnum):
        #logMlist = logMarray(seqlens[si],si,uon,bon,K,thetau,thetab)
        logMlist = logMarray(seqlens[si],uon[si],bon[si],K,thetau,thetab)
        logalphas = logAlphas(logMlist)
        logbetas = logBetas(logMlist)
        logZ = logsumexp(logalphas[-1])
        likelihood -= logZ
        expect = numpy.zeros((K,K))
        for i in range(len(logMlist)):
            if i == 0:
                expect = numpy.exp(logMlist[0] + logbetas[i][:,numpy.newaxis] - logZ)
            elif i < len(logMlist) :
                expect = numpy.exp(logMlist[i] + logalphas[i-1][numpy.newaxis,: ] + logbetas[i][:,numpy.newaxis] - logZ)
            #print "expect t:",i, "expect: ", expect
            p_yi=numpy.sum(expect,axis=1)
            # minus the parameter distribuition
            for ao in uon[si][i]:
                gradu[ao:ao+K] -= p_yi
            for ao in bon[si][i]:
                gradb[ao:ao+K*K] -= expect.reshape((K*K))
    grad -= regularity_deriv(theta,regtype,sigma)
    _gradient = grad
    return likelihood - regularity(theta,regtype,sigma)

def gradient_likelihood(theta):    # this is a dummy function
    global _gradient
    return _gradient

def saveModel(bfnum,ufnum,tlist,obydic,uobxs,bobxs,theta,modelfile):
    import cPickle as pickle
    with open(modelfile, 'wb') as f:
        pickle.dump([bfnum,ufnum,tlist,obydic,uobxs,bobxs,theta], f)

def train(datafile,tpltfile,modelfile,mp=1,regtype=2,sigma=1.0,fd=1.0):
    import time 
    import cPickle as pickle

    start_time = time.time()
    if not os.path.isfile(tpltfile):
        print "Can't find the template file!"
        return -1
    tplist=readTemplate(tpltfile)    #read the template data , format by crf++
    #print tplist
    if not os.path.isfile(datafile):
        print "Data file doesn't exist!"
        return -1
    texts,seqlens,oys,seqnum,K,obydic,y2label=readData(datafile) #read train data
    #print seqlens 
    
    uobxs,bobxs,ufnum,bfnum=processFeatures(tplist,texts,seqnum,K, fd=fd) #编辑训练数据特征模板
    fnum=ufnum+bfnum #总特征数目
    print "Linear CRF in Python.. ver 0.1 "
    print "B features:",bfnum,"U features:",ufnum, "total num:",fnum
    print "training sequence number:",seqnum
    print "start to calculate ON feature.  elapsed time:", time.time() - start_time, "seconds. \n "
    if fnum==0:
        print "No Parameters to Learn. "
        return
    uon,bon = calObservexOn(tplist,texts,uobxs,bobxs,seqnum)
    with open("ubobx", 'wb') as f:
        pickle.dump([uobxs,bobxs], f)
    del uobxs
    del bobxs
    
    print "start to calculate data distribuition. elapsed time:", time.time() - start_time, "seconds. \n "    
    y0=0
    fss=calFSS(texts,oys,uon,bon,ufnum,bfnum,seqnum,K,y0)
    del texts    
    del oys

    print "start to translate ON Feature list to Array.  elapsed time:", time.time() - start_time, "seconds. \n "
    uonarr,uonseqsta,uonlocsta,uonlocend,bonarr,bonseqsta,bonlocsta,bonlocend = calObservexOnLoc(uon,bon,seqnum,mp)
    #print  "ubon size:", sys.getsizeof(uon)
    #print  "uobxs size:", sys.getsizeof(uobxs)
    #print  "texts size:", sys.getsizeof(texts)
    del uon
    del bon
		
    print "start to learn distribuition. elapsed time:", time.time() - start_time, "seconds. \n " 

    #return
    
    from scipy import optimize
    #theta特征训练参数
    if sys.platform=="win32" and mp==1:  # using shared memory
        theta = multiprocessing.Array(ctypes.c_double, ufnum+bfnum)
        print "allocate theta OK. elapsed time:", time.time() - start_time, "seconds. \n "
        theta = numpy.ctypeslib.as_array(theta.get_obj())
        theta = theta.reshape(ufnum+bfnum)
    else:
        theta=random_param(ufnum,bfnum)
    
    #likeli作为目标函数,#likelihood_deriv作为目标函数一阶导数
    if mp==1:  # using multi processing
       pass
    else:
        likeli = lambda x:-likelihood_sa(seqlens,fss,uonarr,uonseqsta,uonlocsta,uonlocend,
                      bonarr,bonseqsta,bonlocsta,bonlocend,x,seqnum,K,ufnum,bfnum,regtype,sigma)
    likelihood_deriv = lambda x:-gradient_likelihood(x)
    
    #使用lbfgs算法求解最优化参数
    theta,fobj,dtemp = optimize.fmin_l_bfgs_b(likeli,theta, 
            fprime=likelihood_deriv , disp=1, factr=1e12)

    with open("ubobx", 'rb') as f:
        uobxs,bobxs = pickle.load(f)

    #存储模型系数,使用python类的方式存储
    saveModel(bfnum,ufnum,tplist,obydic,uobxs,bobxs,theta,modelfile) 
    print "Training finished in ", time.time() - start_time, "seconds. \n "

def main():
    train("trainsimple.data","templatesimple.txt","model",mp=0)
            
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()