### Ensemble Refinement (ER) + Force Field Fitting (FFF)
# Script for the minimization of the loss function L(lambda,phi)
# where lambda is the array of coefficients in ER and phi the array of coefficients for FFF;
# this loss function has two hyper parameters alpha, beta, as described in the paper.
# This algorithm works only for finite values of alpha (non-zero and non-infinite);
# for alpha infinite, see ensemble_ref_run.py; alpha cannot be zero (otherwise ill-defined loss function).
# How to minimize? BFGS fails when trying to minimize directly in (lambda,phi) space,
# so you can do either natural gradient descent or nested minimization.

# %%
import numpy as np
import pandas
from scipy.optimize import minimize
import time
import os
from pathlib import Path
import sys

# %% input parameters: seed (None if no cross validation), alpha, beta

# best order: first seed, then alpha and beta; in this way, you do all for a certain seed, then change seed and so on

seed=int(sys.argv[1])
#seed=None
alpha=float(sys.argv[2])
beta=float(sys.argv[3])

# %% first cases

# if_first is the first time you do a minimization with that data set (i.e. that seed if cross validation),
# in this case you put titles to tables and do the reference ensemble (alpha,beta)=(inf,inf);

# if_fffmin is the first time you do a minimization with that alpha value and that data set (i.e. that seed if cross validation)
# in this case you do the force field fitting minimization

# if_first implies if_fffmin

if (alpha==0.01) and (beta==0.0):
    if_first=True
else:
    if_first=False

if (alpha==0.01):
    if_fffmin=True # if also force field fitting minimization (i.e. alpha=inf)
else:
    if_fffmin=False

# %% functions

def compute_newweights_par(par,f,weights):
    weights=weights/np.sum(weights)
    correction=np.matmul(f,par)
    shift=np.min(correction)
    correction-=shift
    newweights=np.exp(-correction)*weights
    Z=np.sum(newweights) # /np.exp(shift)


    return newweights/np.sum(newweights),Z,shift

def compute_newweights(lambdas,g,weights):
    weights=weights/np.sum(weights)
    correction=np.zeros(len(weights))

    l_from=0
    l_to=0
    for i_type in range(len(g)):#indices[0],indices[-1]):
        l_to+=np.shape(g[i_type])[1]
        correction+=np.matmul(g[i_type],lambdas[l_from:l_to])
        l_from=l_to

    shift=np.min(correction)
    correction-=shift
    newweights=np.exp(-correction)*weights
    Z=np.sum(newweights)

    return newweights/np.sum(newweights),Z,shift

def compute_newweights2(par,lambdas,traj,obs,weights,ntypes):

        weights=weights/np.sum(weights)
        
        correction=np.matmul(np.array(traj),par)
        shift1=np.min(correction)
        correction-=shift1
        newweights1=np.exp(-correction)*weights
        Z1=np.sum(newweights1)#/np.exp(shift1)
        
        l_from=0
        l_to=0
        correction2=np.zeros(len(weights))
        for i_type in range(ntypes):#indices[0],indices[-1]):
                l_to+=np.shape(obs[i_type])[1]
                correction2+=np.matmul(obs[i_type],lambdas[l_from:l_to])
                l_from=l_to
        shift2=np.min(correction2)
        correction2-=shift2
        newweights2=np.exp(-correction2)*newweights1
        Z2=np.sum(newweights2)#/np.exp(shift1+shift2)

        return newweights1/np.sum(newweights1),Z1,shift1,newweights2/np.sum(newweights2),Z2,shift2

def alphabeta_lossf(par1,par2,parmap,data,alpha,beta,if_gradient,if_uNOEs,indices,if_natgrad,fixed):
    # fixed=0 par1=par_lambdas, par2=None; all gradient
    # fixed=2 par1=pars, par2=lambdas; only pars gradient
    # fixed=1 par1=lambdas, par2=pars; only lambdas gradient

    nffs=np.max([item for sublist in parmap for item in sublist])+1 # n. of force field coefficients

    if fixed==0:
        par_lambdas=par1
        par=par_lambdas[:nffs]
        lambdas=par_lambdas[nffs:]
    elif fixed==2:
        par=par1
        lambdas=par2
    elif fixed==1:
        par=par2
        lambdas=par1
    else:
        print('fixed error')
        return
        
    # 1. initialization
    nsystems=np.shape(parmap)[0]
    if if_uNOEs: ntypes=2
    else: ntypes=1

    lossf=0.0
    lossf_single=[]#np.zeros(nsystems)
    if if_gradient:
        gradP=np.zeros(nffs)
        gradL=[]#np.zeros(len(lambdas))
    if (if_natgrad==1 or if_natgrad==2):
        covPL_f=np.zeros((nffs,nffs))

    def stats(): # statistics: Srels, kish, relkish
        return 0
    stats.Srel_alpha=[]
    stats.Srel_beta=[]
    stats.Srel=[]
    stats.kish=[]
    stats.relkish=[]

    errorf=[] # error function: 1/2 chi2 or the function for uNOEs (NOT reduced, notice factor 1/2)
    for i in range(nsystems):
        errorf.append([])

    # compute js and js_sys: indices for lambda corresponding to different systems and types of observables
    n_exp=[]
    for i_sys in range(nsystems):
        n_exp.append([])
        for i_type in range(indices[-1]):
            n_exp[i_sys].append(len(data.g[i_sys][i_type].T))
        if if_uNOEs:
            i_type+=1
            n_exp[i_sys].append(len(data.g[i_sys][i_type].T))
    n_exp=np.array(n_exp)
    js_sys=np.sum(n_exp,axis=1)
    js_sys=[0]+np.cumsum(js_sys).tolist()
    js=[0]+np.cumsum(n_exp).tolist()

    # 3. for over different systems
    for i_sys in range(nsystems):

        par_sys=par[parmap[i_sys]]        
        lambdas_sys=lambdas[js_sys[i_sys]:js_sys[i_sys+1]]
        
        wP,ZP,shiftP,wPL,ZPL,shiftL=compute_newweights2(par_sys,lambdas_sys,data.f[i_sys],data.g[i_sys],data.weights[i_sys],ntypes)

        stats.kish.append(np.sum(wPL**2))
        stats.relkish.append(np.sum(wPL**2/data.weights[i_sys])*np.sum(data.weights[i_sys])) # normalized w,weights

        av_g=[]
        lambda_dot_avg=0
        #ntypes=len(data.g[i_sys])
        for i_type in range(indices[0],indices[1]):
            av_g.append(np.einsum('i,ij',wPL,data.g[i_sys][i_type]))
            errorf[i_sys].append(1/2*np.sum(((av_g[i_type]-data.gexp[i_sys][i_type][:,0])/data.gexp[i_sys][i_type][:,1])**2))
            lambda_dot_avg+=np.matmul(lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]],av_g[i_type])
        if if_uNOEs: # last element: n_type-1
            i_type+=1#indices[1]
            av_g.append(np.einsum('i,ij',wPL,data.g[i_sys][i_type]))
            errorf[i_sys].append(1/2*np.sum((np.maximum(av_g[i_type]-data.gexp[i_sys][i_type][:,0],np.zeros(len(av_g[i_type])))/data.gexp[i_sys][i_type][:,1])**2))
            lambda_dot_avg+=np.matmul(lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]],av_g[i_type])
        
        stats.Srel_alpha.append(lambda_dot_avg+np.log(ZPL)-np.log(ZP)-shiftL)

        weighted_f=wP[:,None]*np.array(data.f[i_sys])
        av_f=np.sum(weighted_f,axis=0)
        
        if if_natgrad==1:
            weightedPL_f=wPL[:,None]*np.array(data.f[i_sys])
            avPL_f=np.sum(weightedPL_f,axis=0)


        par_dot_avf=np.matmul(par_sys.T,av_f)

        stats.Srel_beta.append(par_dot_avf+np.log(ZP)-shiftP)
        stats.Srel.append(par_dot_avf+lambda_dot_avg+np.log(ZPL)-shiftL-shiftP)

        lossf_single.append(np.sum(np.array(errorf[i_sys]))-alpha*np.sum(np.array(stats.Srel_alpha[i_sys]))-beta*np.sum(np.array(stats.Srel_beta[i_sys])))

        # 4. compute the gradient
        if if_gradient:
            for i_type in range(indices[0],indices[1]):
                vec1=(av_g[i_type]-data.gexp[i_sys][i_type][:,0])/data.gexp[i_sys][i_type][:,1]**2-alpha*lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]]
                if (if_natgrad==0 or if_natgrad==1): gradL.append(-vec1)

                # except only the case natural gradient and only lambda gradient
                if not ((if_natgrad==0 or if_natgrad==1) and (fixed==1)): 
                    vect=np.matmul(data.g[i_sys][i_type],vec1)
                    scal1=np.matmul(vec1,av_g[i_type])

                    # 4a. if not natural gradient, compute the lambda components of the gradient in this way
                    if if_natgrad==-1 and (fixed==0 or fixed==1):
                        gradL_single=-np.matmul(data.g[i_sys][i_type].T*wPL,vect)+scal1*av_g[i_type]
                        gradL.append(gradL_single)

                    # 4b. both for gradient and natural gradient, add this term to the phi components
                    if fixed==0 or fixed==2:
                        weightedPL_f=wPL[:,None]*np.array(data.f[i_sys])
                        avPL_f=np.sum(weightedPL_f,axis=0)
    
                        gradP_single=-np.matmul(np.array(data.f[i_sys]).T*wPL,vect)+scal1*avPL_f

            if if_uNOEs:
                i_type+=1
                vec1=np.maximum(av_g[i_type]-data.gexp[i_sys][i_type][:,0],np.zeros(len(av_g[i_type])))/data.gexp[i_sys][i_type][:,1]**2-alpha*lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]]
                if (if_natgrad==0 or if_natgrad==1): gradL.append(-vec1)
                
                # except only the case natural gradient and only lambda gradient
                if not ((if_natgrad==0 or if_natgrad==1) and (fixed==1)): 
                    vect=np.matmul(data.g[i_sys][i_type],vec1)
                    scal1=np.matmul(vec1,av_g[i_type])
                    
                    # 4a. if not natural gradient, compute the lambda components of the gradient in this way
                    if if_natgrad==-1 and (fixed==0 or fixed==1):
                        gradL_single=-np.matmul(data.g[i_sys][i_type].T*wPL,vect)+scal1*av_g[i_type]
                        gradL.append(gradL_single)
                    
                    # 4b. both for gradient and natural gradient, add this term to the phi components
                    if fixed==0 or fixed==2:
                        gradP_single=-np.matmul(np.array(data.f[i_sys]).T*wPL,vect)+scal1*avPL_f

            if fixed==0 or fixed==2:
                # 4b. phi components
                # derivative of average of f: minus the covariance matrix
                dav_f=-np.matmul(np.transpose(weighted_f),np.array(data.f[i_sys]))+np.outer(av_f,av_f)
                
                if if_natgrad==1: covPL_f+=np.matmul(np.transpose(weightedPL_f),np.array(data.f[i_sys]))+np.outer(avPL_f,avPL_f)
                
                #inverse=np.linalg.inv(dav_f)

                gradP_single-=beta*np.matmul(par_sys.T,dav_f)

                # derivative of average of g: minus the correlation matrix with f
                # calculated previously, to optimize

                gradP_single+=alpha*(avPL_f-av_f)

                for i in range(len(parmap[i_sys])):
                    gradP[parmap[i_sys][i]]+=gradP_single[i]

    lossf=np.sum(np.array(lossf_single))
    
    if (fixed==0 or fixed==2) and if_natgrad==1:
        gradP=np.matmul(np.linalg.inv(covPL_f),gradP)

    if if_gradient:
        if fixed==0: grad=np.array(gradP.tolist()+[item for sublist in gradL for item in sublist])
        elif fixed==1: grad=np.array([item for sublist in gradL for item in sublist])
        elif fixed==2: grad=gradP
        
        return lossf,grad,lossf_single,np.array(errorf)*2/np.array(n_exp),stats

    return lossf,lossf_single,np.array(errorf)*2/np.array(n_exp),stats

def gamma_function(lambdas,data,weights,alpha,if_gradient):

    # 1. initialization

    nsystems=len(data.g)

    # start from weights different from the reference ensemble data.weights
    #data.weights=[]
    #for i_sys in range(nsystems):
    #    data.weights.append(weights[i_sys])

    # compute js and js_sys: indices for lambda corresponding to different systems and types of observables
    js=[]
    for i_sys in range(nsystems):
        js.append([])
        l=0
        for i_type in range(len(data.g[i_sys])):
            js[i_sys].append(len(data.g[i_sys][i_type].T))
    js_sys=np.sum(js,axis=1)
    js_sys=[0]+np.cumsum(js_sys).tolist()
    js=[0]+np.cumsum(js).tolist()

    gammaf=0

    if if_gradient:
        grad=[]
    
    for i_sys in range(nsystems):
        
        newweights,Zlambda,shift=compute_newweights(lambdas[js_sys[i_sys]:js_sys[i_sys+1]],data.g[i_sys],weights[i_sys])#js,i_sys,nsystems)
        gammaf_single=np.log(Zlambda)-shift

        ntypes=len(data.g[i_sys])
        for i_type in range(len(data.g[i_sys])):
            gammaf_single+=np.matmul(lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]],data.gexp[i_sys][i_type][:,0])
            gammaf_single+=1/2*alpha*np.matmul(lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]]**2,data.gexp[i_sys][i_type][:,1]**2)

        #gammaf+=alpha*gammaf_single
        gammaf+=gammaf_single

        if if_gradient:
            for i_type in range(len(data.g[i_sys])):
                av_g=np.einsum('i,ij',newweights,data.g[i_sys][i_type])
                #grad_single=-alpha*(av_g-data.gexp[i_sys][i_type][:,0]-alpha*lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]]*data.gexp[i_sys][i_type][:,1]**2)
                grad_single=-(av_g-data.gexp[i_sys][i_type][:,0]-alpha*lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]]*data.gexp[i_sys][i_type][:,1]**2)
                
                grad.append(grad_single)

    if if_gradient:
        grad=[item for sublist in grad for item in sublist]
        return gammaf,grad

    return gammaf

def alphabeta_lossf_test(par_lambdas,parmap,data,data_train,data_test,alpha,beta,if_uNOEs,if_only,indices):
    # you can optimize: compute new weights only once, then split them; 
    # however, you have to compute separately Zs and shifts for both cases

    # 1. initialization
    nsystems=np.shape(parmap)[0]
    if if_uNOEs: ntypes=indices[-1]
    else: ntypes=indices[-1]-1

    nffs=np.max([item for sublist in parmap for item in sublist])+1 # n. of force field coefficients


    par=par_lambdas[:nffs]
    lambdas=par_lambdas[nffs:]

    lossf=0.0
    lossf_single=np.zeros(nsystems)

    def stats(): # statistics: Srels, kish, relkish
        return 0
    stats.Srel_alpha=[]
    stats.Srel_beta=[]
    stats.Srel=[]
    stats.kish=[]
    stats.relkish=[]

    errorf1=[] # error function: 1/2 chi2 or the function for uNOEs (NOT reduced, notice factor 1/2)
    errorf2=[]
    for i in range(nsystems):
        errorf1.append([])
        errorf2.append([])

    # 2. compute js and js_sys: indices for lambda corresponding to different systems and types of observables
    n_exptrain=[]
    for i_sys in range(nsystems):
        n_exptrain.append([])
        for i_type in range(indices[-1]):
            n_exptrain[i_sys].append(len(data_train.g[i_sys][i_type].T))
        if if_uNOEs:
            i_type+=1
            n_exptrain[i_sys].append(len(data_train.g[i_sys][i_type].T))
    js_sys=np.sum(n_exptrain,axis=1)
    js_sys=[0]+np.cumsum(js_sys).tolist()
    js=[0]+np.cumsum(n_exptrain).tolist()

    n_exptest=[]
    for i_sys in range(nsystems):
        n_exptest.append([])
        for i_type in range(indices[-1]):
            n_exptest[i_sys].append(len(data_test.g1[i_sys][i_type].T))
        if if_uNOEs:
            i_type+=1
            n_exptest[i_sys].append(len(data_test.g1[i_sys][i_type].T))

    print(n_exptrain)
    print(n_exptest)

    # 3. for over different systems
    for i_sys in range(nsystems):

        par_sys=par[parmap[i_sys]]
        lambdas_sys=lambdas[js_sys[i_sys]:js_sys[i_sys+1]]
    
        # 3a. training observables, test frames (in order to compute re-weights)
        # then, compute chi2 for training observables (test frames)

        wP,ZP,shiftP,wPL,ZPL,shiftL=compute_newweights2(par_sys,lambdas_sys,np.array(data_test.f2[i_sys]),data_test.g2[i_sys],data_test.w2[i_sys],ntypes)

        stats.kish.append(np.sum(wPL**2))
        stats.relkish.append(np.sum(wPL**2/data_test.w2[i_sys])*np.sum(data_test.w2[i_sys])) # normalized w,weights

        av_g=[]
        lambda_dot_avg=0
        ntypes=len(data.g[i_sys])
        for i_type in range(indices[0],indices[1]):
            av_g.append(np.einsum('i,ij',wPL,data_test.g2[i_sys][i_type]))
            errorf2[i_sys].append(1/2*np.sum(((av_g[i_type]-data_train.gexp[i_sys][i_type][:,0])/data_train.gexp[i_sys][i_type][:,1])**2))
            lambda_dot_avg+=np.matmul(lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]],av_g[i_type])
        if if_uNOEs: # last element
            i_type+=1#indices[1]
            av_g.append(np.einsum('i,ij',wPL,data_test.g2[i_sys][i_type]))
            errorf2[i_sys].append(1/2*np.sum((np.maximum(av_g[i_type]-data_train.gexp[i_sys][i_type][:,0],np.zeros(len(av_g[i_type])))/data_train.gexp[i_sys][i_type][:,1])**2))
            lambda_dot_avg+=np.matmul(lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]],av_g[i_type])
        
        stats.Srel_alpha.append(lambda_dot_avg+np.log(ZPL)-np.log(ZP)-shiftL)

        weighted_f=wP[:,None]*np.array(data_test.f2[i_sys])
        av_f=np.sum(weighted_f,axis=0)
        par_dot_avf=np.matmul(par_sys.T,av_f)

        stats.Srel_beta.append(par_dot_avf+np.log(ZP)-shiftP)

        weighted_PL_f=wPL[:,None]*np.array(data_test.f2[i_sys])
        avPL_f=np.sum(weighted_PL_f,axis=0)
        par_dot_avPLf=np.matmul(par_sys.T,avPL_f)
        stats.Srel.append(par_dot_avPLf+lambda_dot_avg+np.log(ZPL)-shiftL-shiftP)

        lossf_single[i_sys]=np.sum(errorf2[i_sys])-alpha*np.sum(stats.Srel_alpha[i_sys])-beta*np.sum(stats.Srel_beta[i_sys])

        # 3b. if not if_only, compute re-weights with (training observables) all frames (test+training)
        # then, compute chi2 for test observables

        if not if_only:
            wP,ZP,shiftP,wPL,ZPL,shiftL=compute_newweights2(par_sys,lambdas_sys,np.array(data.f[i_sys]),data_test.g3[i_sys],data.weights[i_sys],ntypes)

        av_g=[]
        lambda_dot_avg=0
        for i_type in range(indices[0],indices[1]):
            av_g.append(np.einsum('i,ij',wPL,data_test.g1[i_sys][i_type]))
            errorf1[i_sys].append(1/2*np.sum(((av_g[i_type]-data_test.gexp1[i_sys][i_type][:,0])/data_test.gexp1[i_sys][i_type][:,1])**2))
        if if_uNOEs: # last element
            i_type+=1 #indices[1]
            av_g.append(np.einsum('i,ij',wPL,data_test.g1[i_sys][i_type]))
            errorf1[i_sys].append(1/2*np.sum((np.maximum(av_g[i_type]-data_test.gexp1[i_sys][i_type][:,0],np.zeros(len(av_g[i_type])))/data_test.gexp1[i_sys][i_type][:,1])**2))
        ###

        lossf_single[i_sys]+=np.sum(errorf1[i_sys])


    lossf=np.sum(lossf_single)

    return lossf,lossf_single,np.array(errorf1)*2/np.array(n_exptest),np.array(errorf2)*2/np.array(n_exptrain),stats

# test set, force field fitting
# the output are the error functions, not the reduced chi2; correct this

def compute_newweights_ff(par,f,weights):
    weights=weights/np.sum(weights)
    correction=np.matmul(f,par)
    shift=np.min(correction)
    correction-=shift
    newweights=np.exp(-correction)*weights
    Z=np.sum(newweights) # /np.exp(shift)


    return newweights/np.sum(newweights),Z,shift

def betalossf_test(par,data,datatrain,datatest,parmap,beta,if_uNOEs,if_only,indices):
    # compute the chi2 for non-used observables: datatest.g1 datatest.gexp1 data.weights data.f
    # compute relative entropy for non-used frames and chi2 for non-used frames, used observables:
    # equal to beta_lossf: datatest.g2 datatrain.gexp datatest.f2 datatest.w2
    
    # 1. initialization

    nsystems=np.shape(parmap)[0]

    lossf=0.0
    lossf_single=np.zeros(nsystems)

    def stats(): # statistics: Srels, kish, relkish
        return 0
    stats.Srel=[]
    stats.kish=[]
    stats.relkish=[]

    errorf1=[] # error function: 1/2 chi2 or the function for uNOEs (NOT reduced, notice factor 1/2)
    errorf2=[] # 1 non-used observables, 2 used observables
    for i in range(nsystems):
        errorf1.append([])
        errorf2.append([])

    # which parameters are in common to different systems and which are not?

    # 2. for over different systems
    for i_sys in range(nsystems):
        
        # 1: used observables, non-used frames

        par_sys=par[parmap[i_sys]]

        newweights,Zpar,shift=compute_newweights_ff(par_sys,np.array(datatest.f2[i_sys]),datatest.w2[i_sys])
        
        stats.kish.append(np.sum(newweights**2))
        stats.relkish.append(np.sum(newweights**2/datatest.w2[i_sys])*np.sum(datatest.w2[i_sys])) # normalized w,weights
        
        weighted_f=newweights[:,None]*np.array(datatest.f2[i_sys])
        av_f=np.sum(weighted_f,axis=0)

        stats.Srel.append(np.matmul(par_sys.T,av_f)+np.log(Zpar)-shift)

        
        #if if_uNOEs: ntype-=1
        
        ### put BY HAND the error function (1/2 chi2 or uNOEs) !!
        av_g=[]
        for i_type in range(indices[0],indices[1]):
            av_g.append(np.einsum('i,ij',newweights,datatest.g2[i_sys][i_type]))
            errorf2[i_sys].append(1/2*np.sum(((av_g[i_type]-datatrain.gexp[i_sys][i_type][:,0])/datatrain.gexp[i_sys][i_type][:,1])**2))
        
        if if_uNOEs: # last element: indices[1] 
            av_g.append(np.einsum('i,ij',newweights,datatest.g2[i_sys][indices[1]]))
            errorf2[i_sys].append(1/2*np.sum((np.maximum(av_g[indices[1]]-datatrain.gexp[i_sys][indices[1]][:,0],np.zeros(len(av_g[indices[1]])))/datatrain.gexp[i_sys][indices[1]][:,1])**2))
        ###

        lossf_single[i_sys]=np.sum(errorf2[i_sys])-beta*np.sum(stats.Srel[i_sys])
        
        # 2. non-used observables, all or (if if_only) non-used frames
        
        if not if_only:
            newweights,Zpar,shift=compute_newweights_ff(par_sys,np.array(data.f[i_sys]),data.weights[i_sys])

        av_g=[]
        for i_type in range(indices[0],indices[1]):
            av_g.append(np.einsum('i,ij',newweights,datatest.g1[i_sys][i_type]))
            errorf1[i_sys].append(1/2*np.sum(((av_g[i_type]-datatest.gexp1[i_sys][i_type][:,0])/datatest.gexp1[i_sys][i_type][:,1])**2))
    
            if if_uNOEs: # last element: indices[1]
                av_g.append(np.einsum('i,ij',newweights,datatest.g1[i_sys][indices[1]]))
                errorf1[i_sys].append(1/2*np.sum((np.maximum(av_g[indices[1]]-datatest.gexp1[i_sys][indices[1]][:,0],np.zeros(len(av_g[indices[1]])))/datatest.gexp1[i_sys][indices[1]][:,1])**2))
        
        lossf_single[i_sys]+=np.sum(errorf1[i_sys])

    lossf=np.sum(lossf_single)

    n_exptrain=[]
    for i_sys in range(nsystems):
        n_exptrain.append([])
        for i_type in range(indices[-1]):
            n_exptrain[i_sys].append(len(datatrain.g[i_sys][i_type].T))
        if if_uNOEs:
            i_type+=1
            n_exptrain[i_sys].append(len(datatrain.g[i_sys][i_type].T))

    n_exptest=[]
    for i_sys in range(nsystems):
        n_exptest.append([])
        for i_type in range(indices[-1]):
            n_exptest[i_sys].append(len(datatest.g1[i_sys][i_type].T))
        if if_uNOEs:
            i_type+=1
            n_exptest[i_sys].append(len(datatest.g1[i_sys][i_type].T))

    print('watch out')
    print(errorf1)
    print(n_exptest)
    print(errorf2)
    print(n_exptrain)

    return lossf,lossf_single,np.array(errorf1)*2/np.array(n_exptest),np.array(errorf2)*2/np.array(n_exptrain),stats

# select training and test set after choice of n. of replicas
# positions: the indices for each different replica
# to be done on each molecule
# if_only=True if only non-used frames for non-trained observables (rather than all frames)

# if_same, do the same choice done for ER in path_ER
# watch out: distinguish between non-uNOEs and uNOEs observables
def select_traintest(data,n_test_replicas,n_test_obs,positions,seed,if_only=False,if_same=False,path_ER=None):

    #i_seed=1

    #test_obs.loc[i_seed,:]
    #test_contraj.loc[i_seed,:]
    if not if_same:
        rng=np.random.default_rng(seed=seed)
    
    ### initialization
    choice_obs_all=[]
    choice_rep_all=[]

    def datatest():
        return 0
    datatest.gexp1=[]
    datatest.g1=[]
    datatest.g2=[]
    datatest.g3=[] # all frames, used observables
    for i in range(len(positions)):
        datatest.gexp1.append([])
        datatest.g1.append([])
        datatest.g2.append([])
        datatest.g3.append([])
    datatest.w2=[]
    datatest.f2=[]

    def datatrain():
        return 0
    datatrain.gexp=[]
    datatrain.g=[]
    for i in range(len(positions)):
        datatrain.gexp.append([])
        datatrain.g.append([])
    datatrain.weights=[]
    datatrain.f=[]
    ###


    for i_sys in range(len(positions)):
        n_replicas=len(positions[0])
        n_type=len(data.g[i_sys])

        if n_type!=len(n_test_obs[i_sys]):
            print('ntype: ',n_type)
            print('len n test obs: ',len(n_test_obs[i_sys]))
            print('error 1')
            return
        if (n_test_replicas >= n_replicas):# or (n_test_obs >= n_obs): # for each kind
            print('error 2')
            return

        # compute the frame indices for the test set

        if if_same:
            choice_rep=np.array(pandas.read_csv('%s/%s_test_contraj' % (path_ER,Sequences[i_sys]),header=None,index_col=0).iloc[:,:-1].astype(int).loc[seed,:])
        else:
            choice_rep=np.sort(rng.choice(n_replicas,n_test_replicas,replace=False))

        choice_rep_all.append(choice_rep)
        fin=[]
        for i in range(n_test_replicas):
            fin=np.concatenate((fin,positions[i_sys][choice_rep[i]].flatten()),axis=0)
        fin=np.array(fin).astype(int)
        

        # split gexp, weights, f, g into:
        # train, test1 ('non-trained' obs, all frames), test2 ('trained' obs, 'non-used' frames)

        # split weights into train and test
        print('i_sys',i_sys)
        print('fin',fin)
        datatest.w2.append(data.weights[i_sys][fin])
        datatrain.weights.append(np.delete(data.weights[i_sys],fin))

        # split f into train and test
        datatest.f2.append(data.f[i_sys].iloc[fin])
        datatrain.f.append(np.delete(np.array(data.f[i_sys]),fin,axis=0))

        for i_type in range(n_type):
            
            # independent choice for each tetramer
            n_obs=data.gexp[i_sys][i_type].shape[0]
            if n_test_obs[i_sys][i_type]>=n_obs:
                print('error 3')
                return

            if if_same:
                if i_type==0:
                    choice_obs=np.array(pandas.read_csv('%s/%s_test_obs' % (path_ER,Sequences[i_sys]),header=None,index_col=0).iloc[:,:-1].astype(int).loc[seed,:n_test_obs[i_sys,0]])
                elif i_type==1:
                    choice_obs=np.array(pandas.read_csv('%s/%s_test_obs' % (path_ER,Sequences[i_sys]),header=None,index_col=0).iloc[:,:-1].astype(int).loc[seed,(n_test_obs[i_sys,0]+1):])
            else:
                choice_obs=np.sort(rng.choice(n_obs,n_test_obs[i_sys][i_type],replace=False))
            
            choice_obs_all.append(choice_obs)

            # split gexp into train and test
            datatest.gexp1[i_sys].append(data.gexp[i_sys][i_type][choice_obs])
            datatrain.gexp[i_sys].append(np.delete(data.gexp[i_sys][i_type],choice_obs,axis=0))
 
            # split g into: train, test1 (non-trained obs, all frames or only non-used ones), test2 (trained obs, non-used frames)
            if if_only==False:
                datatest.g1[i_sys].append(data.g[i_sys][i_type][:,choice_obs])
            elif if_only==True:
                datatest.g1[i_sys].append(data.g[i_sys][i_type][fin].T)[choice_obs].T
            datatest.g3[i_sys].append(np.delete(data.g[i_sys][i_type],choice_obs,axis=1))
            datatest.g2[i_sys].append(datatest.g3[i_sys][i_type][fin])

            train_g=np.delete(data.g[i_sys][i_type],fin,axis=0)
            datatrain.g[i_sys].append(np.delete(train_g,choice_obs,axis=1))

    return datatrain, datatest, choice_obs_all, choice_rep_all#_w,train_gexp,train_g,train_f,test2_w,test1_gexp,test1_g,test2_g,test2_f

def normalize_observables(gexp,g):
    # normalize observables
    normg_mean=[]
    normg_std=[]

    for i in range(len(g)):
        normg_mean.append([])
        normg_std.append([])

    for i_sys in range(len(g)):
        for i_par in range(len(g[i_sys])):
            normg_mean[i_sys].append(np.mean(g[i_sys][i_par],axis=0))
            normg_std[i_sys].append(np.std(g[i_sys][i_par],axis=0))

            gexp[i_sys][i_par][:,0]=(gexp[i_sys][i_par][:,0]-normg_mean[i_sys][i_par])/normg_std[i_sys][i_par]
            gexp[i_sys][i_par][:,1]=gexp[i_sys][i_par][:,1]/normg_std[i_sys][i_par]

            g[i_sys][i_par]=(g[i_sys][i_par]-normg_mean[i_sys][i_par])/normg_std[i_sys][i_par]

    return g,gexp,normg_mean,normg_std

def load_data(Sequences,path,types_obs,types_obs_exp,types_angles,types_ff,if_weights,if_skip,step,if_normalize):

    def data(): return 0

    data.g=[]
    data.gexp=[]
    #data.cosangles=[]
    data.f=[]
    data.weights=[]
    data.angles=[]

    for i_sys,seq in enumerate(Sequences):

        # 1. g and gexp
        data.g.append([])
        data.gexp.append([])
        
        for i_type,type in enumerate(types_obs):
            data.g[i_sys].append(np.load(path+'observables/%s/%s.npy' % (seq,type),mmap_mode='r'))
            #data.g[i_sys].append(np.load(path+'observables/data.g[%s][%s].npy' % (seq,type),mmap_mode='r'))
        for i_type,type in enumerate(types_obs_exp):
            data.gexp[i_sys].append(np.load(path+'g_exp/%s/%s.npy' % (seq,type)))
            #data.gexp[i_sys].append(np.load(path+'observables/data.gexp[%s][%s].npy' % (seq,type)))

        # 2. cos angles (input of the forward model)
        #data.cosangles.append([])

        #for i_type,type in enumerate(types_angles):
        #    data.cosangles[i_sys].append(np.cos(np.load(path+'angles/%s/%s.npy' % (seq,type))))

        # 3. weights
        if not if_weights: data.weights.append(np.ones(len(data.g[i_sys][0])))
        # same length of data.g, data.angles and data.f (otherwise signaled error)
        else: print('missing weights')

    # 4. force field correction terms
        data.f.append(pandas.read_csv(path+path_ff_corrections % seq))

        # for col in types_ff (types_ff), sum columns starting with col and save with label col
        # keep only these new columns and delete the others (columns in input cannot start with '_') 
        for column in types_ff:
            filter_col = [col for col in data.f[i_sys] if col.startswith(column)]
            data.f[i_sys]['_'+column]=np.sum(data.f[i_sys][filter_col],axis=1)
        
        filter_col = [col for col in data.f[i_sys] if not col.startswith('_')]
        data.f[i_sys]=data.f[i_sys].drop(columns=filter_col)
        data.f[i_sys].columns=data.f[i_sys].columns.str[1:]
    
    if np.any(np.array(Sequences)=='UUUU'):
        nwhere=np.where(np.array(Sequences)=='UUUU')[0][0]
        data.f[nwhere]=data.f[nwhere][:836000]
        #for i_type in range(len(types_angles)):
        #    data.cosangles[nwhere][i_type]=data.cosangles[nwhere][i_type][:836000]

    # 5. Karplus coefficients (as a single vector)
    Karplus_0=[]
    for s in types_angles:
        Karplus_0.append(np.load(path+'Karplus_coeffs/%s_%s.npy' %(s,path_Karplus)))

    data.Karplus_coeffs_0=np.array([item2 for item in Karplus_0 for item2 in item])

    # 6. do you want to skip frames?
    if if_skip:
        for i_sys in range(len(Sequences)):
            data.f[i_sys]=data.f[i_sys][::step]
            data.weights[i_sys]=data.weights[i_sys][::step]
            for i_type in range(len(data.g[i_sys])):
                data.g[i_sys][i_type]=data.g[i_sys][i_type][::step]
            #for i_type,type in enumerate(types_angles):
            #    data.cosangles[i_sys][i_type]=data.cosangles[i_sys][i_type][::step]

    data.weights[i_sys]=data.weights[i_sys]/np.sum(data.weights[i_sys])

    # 7. do you want to normalize observables?
    if if_normalize:
        data.g,data.gexp,data.normg_mean,data.normg_std = normalize_observables(data.gexp,data.g)
    
    # 8. number of frames and number of observables
    n_systems=len(Sequences)
    n_frames=np.zeros(n_systems)
    n_experiments=[]
    
    for i in range(n_systems):
        n_frames[i]=np.shape(data.g[i][0])[0]
        
        n_experiments.append([])
        for j in range(len(data.g[i])):
            n_experiments[i].append(np.shape(data.g[i][j])[1])

    # to be in agreement with alphabeta_run.py and ensemble_ref_run.py scripts
    for i in range(len(Sequences)):
        temp = np.hstack(data.g[i][:4])
        del data.g[i][1:4]
        data.g[i][0] = temp

        data.weights[i] = data.weights[i]/np.sum(data.weights[i])

        temp = np.vstack(data.gexp[i][:4])
        del data.gexp[i][1:4]
        data.gexp[i][0] = temp

        temp = np.hstack(data.normg_mean[0][:4])
        del data.normg_mean[i][1:4]
        data.normg_mean[i][0] = temp

        temp = np.hstack(data.normg_std[0][:4])
        del data.normg_std[i][1:4]
        data.normg_std[i][0] = temp

    n_frames = n_frames.astype(int)

    return data,n_frames,n_experiments

# %% input data
# directory DATA is on Zenodo

# 1. select (training) molecules
Sequences=['AAAA','CCCC','GACC','UUUU','UCAAUC']

n_systems=len(Sequences)
print('n systems: ',n_systems)

path='DATA/'

# 2. select force field correction and corresponding map of parameters

# case a (only alpha angles)
path_ff_corrections = 'ff_terms/sincos%s'
types_ff=['sinalpha','cosalpha']
single_parmap=[0,1]

# case b (independent coefficients for sinalpha, cosalpha, sinzeta, coszeta)
# path_ff_corrections = 'ff_terms/sincos%s'
# types_ff=['sinalpha','cosalpha','sinzeta','coszeta']
# single_parmap=[0,1,2,3]

# case c (same coefficients for sinalpha,sinzeta and for cosalpha,coszeta)
# path_ff_corrections = 'ff_terms/sincos%s'
#types_ff=['sin','cos']
# single_parmap=[0,1]

# case d (chi dihedral angles)
# path_ff_corrections = 'ff_terms_chi_correction/sincos%s'
# types_ff=['sinchi','coschi']
# single_parmap=[0,1]

# 2b. observables

types_obs_exp=np.array(['backbone1_gamma_3J','backbone2_beta_epsilon_3J','sugar_3J','NOEs','uNOEs'])
types_obs = types_obs_exp
types_angles=np.array(['backbone1_gamma','backbone2_beta_epsilon','sugar'])
path_Karplus = 'original'
if_weights = False

# 3. map of parameters for FFF (IN AGREEMENT WITH types_ff)
# parmap is in the same order as types_ff (data.f read through read_data);
# nffs is the n. of force-field correction coefficients par

parmap=[]
for s in Sequences: parmap.append(single_parmap)
nffs=np.max([item for sublist in parmap for item in sublist])+1

# 4. choice of frames and observables: 
# if if_same==True, select same train/test choice of frames and observables as in ER (path_ER)
# otherwise, select n_replicas_test and frac_obs_test

if_same=True
path_ER='results/ER_skip10'

n_replicas_test=8
frac_obs_test=0.3 # 30%, both for not uNOEs and for uNOEs

# 5. if_uNOEs:
# if if_uNOEs, n_types = 2, else n_types = 1

if_uNOEs=True
n_types=2

indices=[0,1] # from indices[0] to indices[1] excluded for not uNOEs, indices[1] for uNOEs if if_uNOEs

# 6. skip frames (in order to select a subset of them)? if if_skip==True, select step

if_skip=True

if if_skip==False: step=None
else: step=10 # in agreement with the n. of frames in ER path if if_same

if_normalize = True

# 7. path of demuxing trajectories

path_demuxing = 'DATA/demuxing/replica_temp'

# 8. select folder to put results there

userdoc='results_oligomers/ERFFF_sincos_alphazeta_skip%s' % step
#userdoc='results_oligomers/ERFFF_sincos_skip%s/whole_data' % step
#userdoc='results_oligomers/ERFFF_chi_skip%s' % step

if_samefolder=True # sure to save in a folder already existing?

# 9. select minimizer

# L-BFGS-B options:
# gtol : float
# The iteration will stop when max{|proj g_i | i = 1, ..., n} <= gtol where pg_i is the i-th component of the projected gradient.

min_method=3 # 1 for 'alternate' (minimizer1), 2 for 'natural' (minimizer2), 3 for 'nested' (minimizer3), 

if min_method==3 or min_method==1:
    if_natgrad=-1
    gtol1=1e-5 # tolerance for gamma function minimization, by "default" 1e-5
    gtol2=1e-5 # tolerance for force field fitting minimization, by "default" 1e-5
    if_returnall=False

elif min_method==2:
    if_natgrad=1
    tol=1

    a=pandas.read_csv('epsilon_natural',index_col=0)
    a.columns.name=a.index.name
    a.index.name=a.index[0]
    a.columns=a.columns.astype(float)
    a=a.iloc[1:]
    a.index=a.index.astype(float)
    epsilon=a.loc[alpha,beta]/10
    del a

# start from zero?
if_startzero=True

# or use as starting point the point of minimum found with a more coarse-grained selection of frames;
# in this case, select the path for userdoc_startingpoint

# %% load data

data,n_frames,n_exps=load_data(Sequences,path,types_obs,types_obs_exp,types_angles,types_ff,if_weights,if_skip,step,if_normalize)

for i in range(len(data.f)):
    print('f[%i]:' %i, data.f[i])

if not if_uNOEs: n_exps=np.array(n_exps)[:,:-1]

n_exp=np.sum(n_exps)

if if_uNOEs:
    a=np.array([np.sum(np.array(n_exps)[:,:-1],axis=1),np.array(n_exps)[:,-1]]).T
    nobs_test=np.rint(frac_obs_test*a).astype(int)
else:
    a=np.array([np.sum(np.array(n_exps),axis=1)]).T
    nobs_test=np.rint(frac_obs_test*a).astype(int)

print('f shape: ',data.f[0].shape)
print('n_exps: ',n_exps)
print(nobs_test)

n_exptrain=(n_exp-np.sum(nobs_test)).astype(int)

# %% if not if_startzero, select the starting point

if not if_startzero:
    # starting point 
    if not if_uNOEs:
        upto=np.sum(n_exp[:,0])-np.sum(nobs_test[:,0])+nffs+3 # 3 because of seed, alpha, beta
    else:
        upto=np.sum(n_exp)-np.sum(nobs_test)+nffs+3 # 3 because of seed, alpha, beta
    upto=upto.astype(int)
    print('upto: ',upto)

    starting_points=pandas.read_csv(os.path.join(userdoc_startingpoint,"minparlambdas"),header=None,usecols=range(upto),skiprows=[0])#lambda x: x not in rows)

    cols=pandas.read_csv(os.path.join(userdoc_startingpoint,"minparlambdas"),header=None,usecols=range(nffs+3),nrows=1).values.tolist()[0]
    cols+=['lambdas[%i]' % i for i in range(upto-nffs-3)] # lambda coefficients for different molecules
    starting_points.columns=cols

# %% make folder

if not os.path.isdir(userdoc):
    Path(userdoc).mkdir(parents=True, exist_ok=True)
    print('created directory')
elif not if_samefolder: # preventing from overwriting errors
    print('userdoc already existing')
    exit()

# %% import demuxing data

if not seed==None:
    n_temp=5 # replica at 300K is the 6th counting from 1
    positions_all=[]

    print('n systems: ',n_systems)
    for i_sys in range(n_systems):
        Seq=Sequences[i_sys]

        if not (Seq=='UCAAUC' or Seq=='UCUCGU'):
            replica_temp=np.array(pandas.read_csv(path_demuxing+Seq,header=None))
            if if_skip:
                replica_temp=replica_temp[::step]
            replica_temp=replica_temp.astype(int)
            n_replicas=np.shape(replica_temp)[1] 

            replica_index=replica_temp.argsort(axis=1)

            positions=[]
            for i in range(n_replicas):
                positions.append(np.argwhere(replica_index[:,n_temp]==i))
                print(positions[-1].shape)
            positions_all.append(positions)

        else: #if Seq=='UCAAUC' or Seq=='UCUCGU': # fake (do demuxing also on hexamers)
            n_replicas=24

            #if if_skip: length=np.int(n_frames[i_sys]/step)
            #else: length=n_frames[i_sys]
            #length=n_frames[i_sys]

            print('n_frames: ',n_frames[i_sys])
            #print('length: ',length)

            replica_index=np.zeros(n_frames[i_sys])
            for m in range(np.int(n_frames[i_sys]/n_replicas)):
                for i in range(n_replicas):
                    replica_index[n_replicas*m+i]=i
            for i in range(np.mod(n_frames[i_sys],n_replicas)):
                replica_index[n_replicas*(m+1)+i]=i
            np.random.shuffle(replica_index)
    
            positions=[]
            for i in range(n_replicas):
                positions.append(np.argwhere(replica_index==i))
                print(positions[-1].shape)
            positions_all.append(positions)

            # compute positions_all: for all the tetramers, positions_all[i_sys] has the indices corresponding to the different replicas

        del replica_index


n_experiments=np.zeros((n_systems,n_types))
for i in range(n_systems):
    for j in range(n_types):
        n_experiments[i,j]=len(data.gexp[i][j])
n_experiments=n_experiments.astype(int)
print(n_experiments)

#fractions=[]
#for i in range(n_systems):
#    fractions.append((n_replicas-n_replicas_test)/n_replicas*np.sum(n_experiments[i,:]-nobs_test)/np.sum(n_experiments[i,:]))
#print('fraction of training data (observables): for each tetramer approx. ',fractions)

# %%
# write input data in myfile.txt

if if_first:

    text=open(userdoc+'/myfile.txt','w')

    text.write('data of the simulation:\n')
    text.write('n. of systems: %s\n' % n_systems)
    text.write('systems: %s\n' % Sequences)
    text.write('skip? %s\n' % if_skip)
    if if_skip: text.write('step: %s\n' % step)
    text.write('n. of frames: %s\n' % n_frames)
    text.write('use also unobserved NOEs? %s\n' % if_uNOEs)
    text.write('force field correction terms: %s\n' % types_ff)
    text.write('n. of force field correction terms: %s\n' % nffs)
    text.write('n. of experiments:\n %s\n' % n_experiments)
    #text.write('n. repetitions: %i\n' % n_repeat)
    #text.write('alpha values: %s\n' % alphas)
    #text.write('beta values: %s\n' % betas)
    if not if_same:
        text.write('n. replicas test: %s\n' % n_replicas_test)
    else:
        text.write('same choice as ensemble refinement \n')
    text.write('n. observables test: \n%s\n' % nobs_test)
    text.write('minimization method: %s\n' % min_method)
    text.write('start from zero? %s\n' % if_startzero)
    if min_method==3 or min_method==1:
        text.write('gtol1 (inner minimization) %s\n' % gtol1)
        text.write('gtol2 (outer minimization) %s\n' % gtol2)
    elif min_method==2:
        text.write('gtol %s\n' % tol)
        text.write('epsilon %s\n' % epsilon)

    text.close()

# %% minimizations

# if you want to plot minimization steps (intermediate values of the loss function)
if_show=False

if if_show:
    min_steps=[]
    class CallbackFunctor: # for lossf_nested (minimizer 3)
        def __init__(self, obj_fun):
            #self.best_fun_vals = [np.inf]
            #self.best_sols = []
            
         #   self.xs=[]
            self.fun_vals=[]
            
         #   self.num_calls = 0
            self.obj_fun = obj_fun
                
        def __call__(self, x):#, args):#data_train,alpha,if_gradient,if_uNOEs,indices):lossf_nested, pars0, args=(lambdas0,parmap,data,alpha,beta,if_uNOEs,indices,bounds)
            bounds=None
            fun_val = self.obj_fun(x,lambdas,parmap,data,alpha,beta,if_uNOEs,indices,bounds)[0] # data or data_train
            #fun_val = self.obj_fun(x,data,alpha,if_gradient,if_uNOEs,indices)[0]
            print('BFGS step: ',fun_val)
            #self.num_calls += 1
            #if fun_val < self.best_fun_vals[-1]:
                #self.best_sols.append(x)
                #self.best_fun_vals.append(fun_val)
            #self.xs.append(x)
            self.fun_vals.append(fun_val)
            
    
        #def save_sols(self, filename):
        #    sols = np.array([sol for sol in self.best_sols])
        #    np.savetxt(filename, sols)

    print('CallBack initialized')

# %% minimizer1 is alternate directions (startint with force field fitting)

# Nmax=500
def minimizer1(par_lambdas,parmap,data,alpha,beta,if_uNOEs,indices,bounds):
    
    start=time.time()

    if_natgrad=-1

    pars=par_lambdas[:nffs]
    lambdas=par_lambdas[nffs:]

    if_saveandprint=True

    if if_saveandprint:

        fs=[]

        fixed=0
        if_gradient=False
        fun=alphabeta_lossf(par_lambdas,None,parmap,data,alpha,beta,if_gradient,if_uNOEs,indices,if_natgrad,fixed)[0]
        print(fun)
        fs.append(fun)

    if_gradient=True

    f1=0
    f2=10

    iter=0
    times=0
    failed_ER=0

    while np.abs(f1-f2)>1e-3 and iter<500 and times<36000:

        fixed=2

        if if_saveandprint: print('\nforce field fitting direction')
        
        mini=minimize(alphabeta_lossf, pars, args=(lambdas,parmap,data,alpha,beta,if_gradient,if_uNOEs,indices,if_natgrad,fixed), method='BFGS',jac=True,options={'return_all': if_returnall,'gtol': gtol2})#,callback=cb)
        pars=mini.x

        if iter==0: pars_FF=mini.x

        if if_saveandprint:
            print(mini.success)
            print(mini.message)
            f1=mini.fun
            print(f1)
            fs.append(f1)

        # compute weights
        weights=[]
        for i_sys in range(len(Sequences)):
            weights.append(compute_newweights_par(pars,np.array(data.f[i_sys]),data.weights[i_sys])[0])

        fixed=1

        if if_saveandprint: print('\nensemble refinement direction')
        #mini=minimize(alphabeta_lossf, lambdas, args=(pars,parmap,data,alpha,beta,if_gradient,if_uNOEs,indices,if_natgrad,fixed), method='BFGS',jac=True)#,callback=cb)
        mini=minimize(gamma_function, lambdas, args=(data,weights,alpha,if_gradient), method='L-BFGS-B',jac=True,bounds=bounds,options={'gtol': gtol1})#,options={'gtol': 1e-15, 'disp': False})
        lambdas=mini.x

        if if_saveandprint:
            print(mini.success)
            print(mini.message)

            f2=alphabeta_lossf(lambdas,pars,parmap,data,alpha,beta,if_gradient,if_uNOEs,indices,if_natgrad,fixed)[0]

            print(f2)
            fs.append(f2)

        if mini.message=='Max. number of function evaluations reached': failed_ER+=1

        iter+=1
        times=time.time()-start

    par_lambdas=pars.tolist()+lambdas.tolist()

    return iter,failed_ER,par_lambdas,pars_FF

# %% 
# minimizer2 is (pseudo-natural) gradient descent (with fixed epsilon)

def minimizer2(if_natgrad,tol,epsilon,x0,pars2,fixed,parmap,data,alpha,beta):
    ###

    # if_natgrad = -1 means gradient descent, 0 means natural gradient descent with approx. 0
    # 1 means natural gradient descent with approx. 1

    # fixed = 0 means both pars and lambdas, 1 means only lambdas minimization, 2 means only pars minimization
    print('fixed is: ',fixed)

    if_save_min=False # True if you want to save the loss function at intermediate steps

    if_gradient=True # to get the gradient (if if_natgrad=-1) or natural gradient (if if_natgrad=0,1)

    Nmax=5000000 # max. n. of iterations (at given epsilon)
    #Nmax_all=4*Nmax # max. n. of iterations (for all the epsilons)
    #epsilon=0.01 # length step: epsilon times the norm of the (natural) gradient
    # tol=0.1 # stop when Nmax is reached or norm of (natural) gradient less than tol

    print('epsilon: ',epsilon)

    if if_save_min: lossfs=[]

    x=+x0

    out=alphabeta_lossf(x,pars2,parmap,data,alpha,beta,if_gradient,if_uNOEs,indices,if_natgrad,fixed)

    lossf=out[0]
    lossf0=lossf
    length=np.linalg.norm(out[1])

    if if_save_min: 
        lossfs.append(lossf)
        print('lossf: ',lossf)
        print('gradient length:',length)
    
    lossf_new=0
    iter=0
    iter_all=0
    increments=0

    while (iter<Nmax) and (length>tol) and (increments<10):
        
        x-=epsilon*out[1]#/np.linalg.norm(out[1])
        out=alphabeta_lossf(x,pars2,parmap,data,alpha,beta,if_gradient,if_uNOEs,indices,if_natgrad,fixed)
        lossf_new=out[0]
        length=np.linalg.norm(out[1])

        if if_save_min:
            lossfs.append(lossf_new)
            print('lossf: ',lossf_new)
            print('gradient length:',length)

        if lossf_new>lossf: increments+=(lossf_new-lossf)*(iter+1) # penalize increments, in particular at long times

        else:
            lossf_min=lossf
            x_min=x+epsilon*out[1] # the ones corresponding to lossf

        iter+=1
        lossf=+lossf_new
    
    if length<=tol:
        print('converged with tolerance %f' % tol)
        failed=0
        x_min=x
        lossf_min=lossf
    else:
        print('not converged')
        failed=1

    if if_save_min: return x_min,lossf_min,iter,lossfs
    return x_min,lossf_min,iter,failed

# %%
# minimizer3 is nested minimization

def lossf_nested(pars,parmap,data,alpha,beta,if_uNOEs,indices,bounds): # implicit input: lambdas
    
    global lambdas
    
    print('lambdas: ',lambdas)

    # bounds
    if if_uNOEs: method='L-BFGS-B'
    else: method='BFGS'

    weights=[]
    for i_sys in range(len(Sequences)):
        par_sys=pars[parmap[i_sys]]  
        weights.append(compute_newweights_par(par_sys,np.array(data.f[i_sys]),data.weights[i_sys])[0])

    mini=minimize(gamma_function, lambdas, args=(data,weights,alpha,True), method=method,jac=True,bounds=bounds,options={'gtol': gtol1})#,options={'gtol': 1e-15, 'disp': False})
    lambdas=mini.x
    
    result=alphabeta_lossf(pars,lambdas,parmap,data,alpha,beta,True,if_uNOEs,indices,-1,2)

    return result[0],result[1],lambdas

def minimizer3(par_lambdas,parmap,data,alpha,beta,if_uNOEs,indices,bounds,if_returnall,gtol2):
    
    global lambdas

    pars0=par_lambdas[:nffs]
    lambdas=par_lambdas[nffs:] # i.e. lambdas0

    #if if_show:
    #    cb = CallbackFunctor(lossf_nested)
    #else:
    #    cb = None
    
    mini=minimize(lossf_nested, pars0, args=(parmap,data,alpha,beta,if_uNOEs,indices,bounds), method='BFGS',jac=True,options={'return_all': if_returnall,'gtol': gtol2})
    pars=mini.x
    iter=mini.nit
    failed_ER=mini.success

    if if_returnall:
        mini_steps=np.array(mini.allvecs)
    else:
        mini_steps=None

    par_lambdas=pars.tolist()+lambdas.tolist()

    return iter,failed_ER,par_lambdas,mini_steps


# %% whole data set

def run_wholedataset():

    if if_uNOEs:
        bounds=[]
        for i_sys in range(len(Sequences)):
            for i in range(data.g[i_sys][0].shape[1]):
                bounds.append([-np.inf,+np.inf])
            if if_uNOEs:
                for i in range(data.g[i_sys][1].shape[1]):
                    bounds.append([0,+np.inf])
    else:
        bounds=None

    if if_startzero: par_lambdas0=np.zeros(nffs+np.sum(n_exps))
    
    txt_table=open(os.path.join(userdoc,"all_table_%s_%s" % (alpha,beta)),'a')
    txt_minpars=open(os.path.join(userdoc,"all_minparlambdas_%s_%s" % (alpha,beta)),'a')

    if if_fffmin:
        txt_table_ff=open(os.path.join(userdoc,"ff_all_table_%s" % beta),'a')
        txt_minpars_ff=open(os.path.join(userdoc,"ff_all_minpars_%s" % beta),'a')

    if if_first:
        # 1. put titles
        s=[]
        for i in range(n_systems):
            for j in range(n_types):
                s.append('redchi2['+str(i)+']['+str(j)+']')
        table_columns=['alpha', 'beta', 'time', 'niter', 'nfail', 'lossf']+['lossf[%s]' %i for i in range(n_systems)]+s+['Srel[%s]' %i for i in range(n_systems)]+['Srel_alpha[%s]' %i for i in range(n_systems)]+['Srel_beta[%s]' %i for i in range(n_systems)]+['kish[%s]' %i for i in range(n_systems)]+['relkish[%s]' %i for i in range(n_systems)]
        table_columns_ff=['beta','lossf']+['lossf[%s]' %i for i in range(n_systems)]+s+['Srel[%s]' %i for i in range(n_systems)]+['Srel_alpha[%s]' %i for i in range(n_systems)]+['Srel_beta[%s]' %i for i in range(n_systems)]+['kish[%s]' %i for i in range(n_systems)]+['relkish[%s]' %i for i in range(n_systems)]

        np.savetxt(txt_table,table_columns,fmt='%s',newline=',')
        np.savetxt(txt_table_ff,table_columns_ff,fmt='%s',newline=',')

        title_ff=['alpha','beta']+types_ff#,'sin alpha','cos alpha','sin zeta','cos zeta'] # then there are all the lambdas coefficients
        np.savetxt(txt_minpars,title_ff,fmt='%s',newline=',')
        np.savetxt(txt_minpars_ff,title_ff,fmt='%s',newline=',')

        # 2. reference ensemble

        lossf,lossf_single,redchi2,stats=alphabeta_lossf(par_lambdas0,None,parmap,data,0,0,if_gradient=False,if_uNOEs=if_uNOEs,indices=indices,if_natgrad=-1,fixed=0)
        v=[+np.infty,+np.infty,0,0,0,lossf]+lossf_single+np.array(redchi2).flatten().tolist()+stats.Srel+stats.Srel_alpha+stats.Srel_beta+stats.kish+stats.relkish

        txt_table.write('\n')
        np.savetxt(txt_table,v,fmt="%.8f",newline=",")
        txt_minpars.write('\n')
        np.savetxt(txt_minpars,[+np.infty,+np.infty]+par_lambdas0.tolist(),fmt="%.8f",newline=",")

    # 3. iterations over alpha, beta


    # nobs_test for each kind of experiments: in this case let's take nobs_test=[nobs_test_no_uNOEs,nobs_test_uNOEs]
    # for each iteration: minimize on the training set with uNOEs and evaluate the point of minimum also on the test set

    start=time.time()

    if min_method==1: 
        [iter,failed_ER,min_parlambdas,min_pars_FF]=minimizer1(par_lambdas0,parmap,data,alpha,beta,if_uNOEs,indices,bounds)
    elif min_method==2:
        [min_parlambdas,lossf,iter,failed_ER]=minimizer2(if_natgrad,tol,epsilon,par_lambdas0,None,0,parmap,data,alpha,beta)
    elif min_method==3:
        [iter,failed_ER,min_parlambdas,mini_steps]=minimizer3(par_lambdas0,parmap,data,alpha,beta,if_uNOEs,indices,bounds,if_returnall,gtol2)
        if if_returnall: np.savetxt(os.path.join(userdoc,"mini_steps_%s_%s" % (alpha,beta)),mini_steps)

    #elif min_method==2:
        #if_natgrad=-1
        #minimizer2(if_natgrad,tol,epsilon,x0,pars2,fixed,parmap,data,alpha,beta)

    times=time.time()-start
    
    ################# if if_fffmin: do force field fitting

    if if_fffmin:
        if if_startzero:
            pars0=np.zeros(nffs)
            lambdas0=np.zeros(np.sum(n_exps))

        if min_method==2:
            [min_pars_FF,lossf,iter,failed_ER]=minimizer2(1,tol,epsilon,pars0,lambdas0,2,parmap,data,alpha,beta)

        elif min_method==3:

            mini=minimize(alphabeta_lossf, pars0, args=(lambdas0,parmap,data,np.array([0]),beta,True,if_uNOEs,indices,-1,2), method='BFGS',jac=True,options={'return_all': if_returnall,'gtol': gtol2}) #,callback=cb)
            
            if if_returnall:
                mini_steps=np.array(mini.allvecs)
                np.savetxt(os.path.join(userdoc,"ff_mini_steps_%s" % beta),mini_steps)

            min_pars_FF=mini.x

        lossf,lossf_single,redchi2,stats=alphabeta_lossf(min_pars_FF,np.zeros(np.sum(n_exps)),parmap,data,np.array([0]),beta,if_gradient=False,if_uNOEs=if_uNOEs,indices=indices,if_natgrad=None,fixed=2)

        v=[np.infty,beta,lossf]+lossf_single+np.array(redchi2).flatten().tolist()+stats.Srel+stats.Srel_alpha+stats.Srel_beta+stats.kish+stats.relkish

        txt_table_ff.write('\n')
        np.savetxt(txt_table_ff,v,fmt="%.8f",newline=",")
        txt_minpars_ff.write('\n')
        np.savetxt(txt_minpars_ff,[np.infty,beta]+min_pars_FF.tolist(),fmt="%.8f",newline=",")

        txt_table_ff.close()
        txt_minpars_ff.close()

    ####################

    #lossf=mini.fun
    lossf,lossf_single,redchi2,stats=alphabeta_lossf(np.array(min_parlambdas),None,parmap,data,alpha,beta,if_gradient=False,if_uNOEs=if_uNOEs,indices=indices,if_natgrad=-1,fixed=0)

    # in agreement with table's labels
    v=[alpha,beta,times,iter,failed_ER,lossf]+lossf_single+np.array(redchi2).flatten().tolist()+stats.Srel+stats.Srel_alpha+stats.Srel_beta+stats.kish+stats.relkish

    ######### SAVE DATA
    txt_table.write('\n')
    np.savetxt(txt_table,v,fmt="%.8f",newline=",")
    txt_minpars.write('\n')
    np.savetxt(txt_minpars,[alpha,beta]+min_parlambdas,fmt="%.8f",newline=",")

    txt_table.close()
    txt_minpars.close()

    return()

# %% cross validation: training and test set

# use also types_ff
def run_crossvalidation():

    if if_uNOEs:
        bounds=[]
        for i_sys in range(len(Sequences)):
            for i in range(data.g[i_sys][0].shape[1]-nobs_test[i_sys][0]):
                bounds.append([-np.inf,+np.inf])
            if if_uNOEs:
                for i in range(data.g[i_sys][1].shape[1]-nobs_test[i_sys][1]):
                    bounds.append([0,+np.inf])
    else:
        bounds=None

    txt_table=open(os.path.join(userdoc,"table_%s_%s_%s" % (alpha,beta,seed)),'a')
    txt_minpars=open(os.path.join(userdoc,"minparlambdas_%s_%s_%s" % (alpha,beta,seed)),'a')

    if if_fffmin:
        txt_table_ff=open(os.path.join(userdoc,"ff_table_%s_%s" % (beta,seed)),'a')
        txt_minpars_ff=open(os.path.join(userdoc,"ff_minpars_%s_%s" % (beta,seed)),'a')

    if if_first:

        # 1. title lines
        s=[]
        s1=[]
        s2=[]
        for i in range(n_systems):
            for j in range(n_types):
                s.append('redchi2['+str(i)+']['+str(j)+']')
                s1.append('redchi2_1['+str(i)+']['+str(j)+']')
                s2.append('redchi2_2['+str(i)+']['+str(j)+']')

        table_columns=['alpha', 'beta', 'seed','time', 'niter', 'nfail', 'train_lossf', 'test_lossf']+['train_lossf[%s]' %i for i in range(n_systems)]+['test_lossf[%s]' %i for i in range(n_systems)]+s+s1+s2+['train_Srel[%s]' %i for i in range(n_systems)]+['train_Srel_alpha[%s]' %i for i in range(n_systems)]+['train_Srel_beta[%s]' %i for i in range(n_systems)]+['test_Srel[%s]' %i for i in range(n_systems)]+['test_Srel_alpha[%s]' %i for i in range(n_systems)]+['test_Srel_beta[%s]' %i for i in range(n_systems)]+['kish[%s]' %i for i in range(n_systems)]+['relkish[%s]' %i for i in range(n_systems)]+['test_kish[%s]' %i for i in range(n_systems)]+['test_relkish[%s]' %i for i in range(n_systems)]
        table_columns_ff=['alpha','beta','seed','train_lossf','test_lossf']+['train_lossf[%s]' %i for i in range(n_systems)]+['test_lossf[%s]' %i for i in range(n_systems)]+s+s1+s2+['train_Srel[%s]' %i for i in range(n_systems)]+['test_Srel[%s]' %i for i in range(n_systems)]+['kish[%s]' %i for i in range(n_systems)]+['relkish[%s]' %i for i in range(n_systems)]+['test_kish[%s]' %i for i in range(n_systems)]+['test_relkish[%s]' %i for i in range(n_systems)]

        np.savetxt(txt_table,table_columns,fmt='%s',newline=',')
        np.savetxt(txt_table_ff,table_columns_ff,fmt='%s',newline=',')

        #title_ff=['alpha','beta','seed','sin alpha','cos alpha','sin zeta','cos zeta'] # then there are all the lambdas coefficients
        title_ff=['alpha','beta','seed']+types_ff
        np.savetxt(txt_minpars,title_ff,fmt='%s',newline=',')
        np.savetxt(txt_minpars_ff,title_ff,fmt='%s',newline=',')

    # 2. different choices of training and test set, based on the seed


    data_train,data_test,choice_obs_all,choice_rep_all=select_traintest(data,n_replicas_test,nobs_test,positions_all,seed,if_only=False,if_same=if_same,path_ER=path_ER)
    print('train',data_train.g[0][0].shape)

    if min_method==3: global par_lambdas0

    if if_first:
        
        test_obs=open(os.path.join(userdoc,"test_obs_%s" % seed),'a')
        test_contraj=open(os.path.join(userdoc,"test_contraj_%s" % seed),'a')

        # save choice_obs_all, choice_rep_all
        test_obs.write('\n')
        np.savetxt(test_obs,[seed]+[item for sublist in choice_obs_all for item in sublist],fmt="%d",newline=",")
        test_contraj.write('\n')
        np.savetxt(test_contraj,[seed]+[item for sublist in choice_rep_all for item in sublist],fmt="%d",newline=",")

        test_obs.close()
        test_contraj.close()
        
        # 3. reference ensemble
        par_lambdas0=np.zeros(nffs+np.sum(n_exps)-np.sum(nobs_test)) # zeros because reference ensemble

        #alphabeta_lossf(par1,par2,parmap,data,alpha,beta,if_gradient,if_uNOEs,indices,if_natgrad,fixed)
        # alphabeta_lossf_test(par_lambdas,parmap,data,data_train,data_test,alpha,beta,if_uNOEs,if_only,indices)
        lossf,lossf_single,redchi2,stats=alphabeta_lossf(par_lambdas0,None,parmap,data_train,np.array([0]),np.array([0]),if_gradient=False,if_uNOEs=if_uNOEs,indices=indices,if_natgrad=-1,fixed=0)
        test_lossf,test_lossf_single,redchi2_1,redchi2_2,test_stats=alphabeta_lossf_test(par_lambdas0,parmap,data,data_train,data_test,np.array([0]),np.array([0]),if_uNOEs=if_uNOEs,if_only=False,indices=indices)

        # in agreement with table's labels
        v=[+np.infty,+np.infty,seed,0,0,0,lossf,test_lossf]+lossf_single+test_lossf_single.tolist()+np.array(redchi2).flatten().tolist()+np.array(redchi2_1).flatten().tolist()+np.array(redchi2_2).flatten().tolist()+stats.Srel+stats.Srel_alpha+stats.Srel_beta+test_stats.Srel+test_stats.Srel_alpha+test_stats.Srel_beta+stats.kish+stats.relkish+test_stats.kish+test_stats.relkish

        txt_table.write('\n')
        np.savetxt(txt_table,v,fmt="%.8f",newline=",")
        txt_minpars.write('\n')
        np.savetxt(txt_minpars,[+np.infty,+np.infty,seed]+par_lambdas0.tolist(),fmt="%.8f",newline=",")
            
    # nobs_test for each kind of experiments: in this case let's take nobs_test=[nobs_test_no_uNOEs,nobs_test_uNOEs]
    # for each iteration: minimize on the training set with uNOEs and evaluate the point of minimum also on the test set
    if if_startzero: par_lambdas0=np.zeros(nffs+np.sum(n_exps)-np.sum(nobs_test))
    else: par_lambdas0=np.array(starting_points[starting_points['alpha']==alpha][starting_points['beta']==beta][starting_points['seed']==seed].iloc[:,3:])[0,:]


    start=time.time()

    if min_method==1:
        [iter,failed_ER,min_parlambdas,min_pars_ff]=minimizer1(par_lambdas0,parmap,data_train,alpha,beta,if_uNOEs,indices,bounds)
    elif min_method==2:
        [min_parlambdas,lossf,iter,failed_ER]=minimizer2(if_natgrad,tol,epsilon,par_lambdas0,None,0,parmap,data_train,alpha,beta)
    elif min_method==3:
        [iter,failed_ER,min_parlambdas,mini_steps]=minimizer3(par_lambdas0,parmap,data_train,alpha,beta,if_uNOEs,indices,bounds,if_returnall,gtol2)
        if if_returnall: np.savetxt(os.path.join(userdoc,"mini_steps_%s_%s" % (alpha,beta)),mini_steps)

    times=time.time()-start
    #print('done')

    ##if if_alpha_first: # once for each (beta, seed)
    # A. force field fitting

    if if_fffmin:
        if if_startzero:
            pars0=np.zeros(nffs)
            lambdas0=np.zeros(np.sum(n_exps))
        else:
            pars0=par_lambdas0[:nffs]
            lambdas0=par_lambdas0[nffs:]

        if min_method==2:
            [min_pars_ff,lossf,iter,failed_ER]=minimizer2(1,tol,epsilon,pars0,lambdas0,2,parmap,data_train,alpha,beta)
 

        elif min_method==3:

            mini=minimize(alphabeta_lossf, pars0, args=(lambdas0,parmap,data_train,np.array([0]),beta,True,if_uNOEs,indices,-1,2), method='BFGS',jac=True,options={'return_all': if_returnall,'gtol': gtol2}) #,callback=cb)
            
            if if_returnall:
                mini_steps=np.array(mini.allvecs)
                np.savetxt(os.path.join(userdoc,"ff_mini_steps_%s" % beta),mini_steps)

            min_pars_ff=mini.x

        lossf,lossf_single,redchi2,stats=alphabeta_lossf(min_pars_ff,np.zeros(np.sum(n_exps)),parmap,data_train,np.array([0]),beta,if_gradient=False,if_uNOEs=if_uNOEs,indices=indices,if_natgrad=-1,fixed=2)
        print('pars (FFF): ',min_pars_ff)
        print('lossf (FFF): ',lossf)
        #  lossf,lossf_single,errorf1,errorf2,stats=betalossf_test(par,data,datatrain,datatest,parmap,beta,if_uNOEs,if_only)
        test_lossf,test_lossf_single,redchi2_1,redchi2_2,test_stats=betalossf_test(min_pars_ff,data,data_train,data_test,parmap,beta,if_uNOEs,if_only=False,indices=indices)

        v=[np.infty,beta,seed,lossf,test_lossf]+lossf_single+test_lossf_single.tolist()+np.array(redchi2).flatten().tolist()+np.array(redchi2_1).flatten().tolist()+np.array(redchi2_2).flatten().tolist()+stats.Srel+test_stats.Srel+stats.kish+stats.relkish+test_stats.kish+test_stats.relkish
        
        txt_table_ff.write('\n')
        np.savetxt(txt_table_ff,v,fmt="%.8f",newline=",")
        txt_minpars_ff.write('\n')
        np.savetxt(txt_minpars_ff,[np.infty,beta,seed]+min_pars_ff.tolist(),fmt="%.8f",newline=",")

        txt_table_ff.close()
        txt_minpars_ff.close()

    # B. force field fitting + ensemble refinement

    # lossf,lossf_single,errorf,stats=beta_lossf(par,data,parmap,beta,if_gradient,if_uNOEs)
    lossf,lossf_single,redchi2,stats=alphabeta_lossf(np.array(min_parlambdas),None,parmap,data_train,alpha,beta,if_gradient=False,if_uNOEs=if_uNOEs,indices=indices,if_natgrad=-1,fixed=0)

    #  lossf,lossf_single,errorf1,errorf2,stats=betalossf_test(par,data,datatrain,datatest,parmap,beta,if_uNOEs,if_only)
    test_lossf,test_lossf_single,redchi2_1,redchi2_2,test_stats=alphabeta_lossf_test(np.array(min_parlambdas),parmap,data,data_train,data_test,alpha,beta,if_uNOEs=if_uNOEs,if_only=False,indices=indices)

    # in agreement with table's labels
    v=[alpha,beta,seed,times,iter,failed_ER,lossf,test_lossf]+lossf_single+test_lossf_single.tolist()+np.array(redchi2).flatten().tolist()+np.array(redchi2_1).flatten().tolist()+np.array(redchi2_2).flatten().tolist()+stats.Srel+stats.Srel_alpha+stats.Srel_beta+test_stats.Srel+test_stats.Srel_alpha+test_stats.Srel_beta+stats.kish+stats.relkish+test_stats.kish+test_stats.relkish
    
    txt_table.write('\n')
    np.savetxt(txt_table,v,fmt="%.8f",newline=",")
    txt_minpars.write('\n')
    np.savetxt(txt_minpars,[alpha,beta,seed]+min_parlambdas,fmt="%.8f",newline=",")

    txt_table.close()
    txt_minpars.close()

    return()

# %% let's compute

if seed==None: 
    run_wholedataset()
else:
    run_crossvalidation()