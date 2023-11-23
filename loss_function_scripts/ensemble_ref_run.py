### Ensemble Refinement: 
# notebook for minimizing the loss function of Ensemble Refinement, with hyper parameter alpha (beta=infty)

# %%
import numpy as np
import pandas
from scipy.optimize import minimize
import time
import os
from pathlib import Path
import sys

# %% input parameters: seed (None if no cross validation), molecule, alpha

#seed=int(sys.argv[1])
seed=None

molecule=sys.argv[1]

alpha=float(sys.argv[2])

# %% first cases

# if_first is the first time you do a minimization with that data set (i.e. that seed if cross validation),
# in this case you put titles to tables and do the reference ensemble alpha = inf;

if alpha==0.01:
    if_first=True
else:
    if_first=False

# %% functions

def compute_newweights(lambdas,g,weights):
    #print('lambdas: ',lambdas)
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

def alpha_lossf(lambdas,data,alpha,if_gradient,if_uNOEs,indices,if_natgrad): # data.weights .f .g .gexp
    
    #print('lambdas: ',lambdas)
    # 1. initialization

    nsystems=len(data.g)

    # compute js and js_sys: indices for lambda corresponding to different systems and types of observables
    js=[]
    for i_sys in range(nsystems):
        js.append([])
        for i_type in range(indices[-1]):
            js[i_sys].append(len(data.g[i_sys][i_type].T))
        if if_uNOEs:
            i_type+=1
            js[i_sys].append(len(data.g[i_sys][i_type].T))
    js_sys=np.sum(js,axis=1)
    js_sys=[0]+np.cumsum(js_sys).tolist()
    js=[0]+np.cumsum(js).tolist()

    lossf=0.0

    global stats,errorf,lossf_single

    lossf_single=np.zeros(nsystems)
    if if_gradient:
        grad=[]

    def stats(): # statistics: Srels, kish, relkish
        return 0
    stats.Srel=[]
    stats.kish=[]
    stats.relkish=[]

    errorf=[] # error function: 1/2 chi2 or the function for uNOEs (NOT reduced, notice factor 1/2)
    for i in range(nsystems):
        errorf.append([])

    # 2. for over different systems
    for i_sys in range(nsystems):

        newweights,Zpar,shift=compute_newweights(lambdas[js_sys[i_sys]:js_sys[i_sys+1]],data.g[i_sys],data.weights[i_sys])#,indices)#js,i_sys,nsystems)

        stats.kish.append(np.sum(newweights**2))
        stats.relkish.append(np.sum(newweights**2/data.weights[i_sys])*np.sum(data.weights[i_sys])) # normalized w,weights
        
        ##weighted_f=newweights[:,None]*np.array(data.f[i_sys])
        ##av_f=np.sum(weighted_f,axis=0)

        ##stats.Srel.append(np.matmul(par_sys.T,av_f)+np.log(Zpar)-shift)

        #n_type=len(data.g[i_sys])
        #if if_uNOEs: ntype-=1
        
        ### put BY HAND the error function (1/2 chi2 or uNOEs) !!
        av_g=[]
        lambda_dot_avg=0
        ntypes=len(data.g[i_sys])
        for i_type in range(indices[0],indices[1]):#n_type-1):
            av_g.append(np.einsum('i,ij',newweights,data.g[i_sys][i_type]))
            errorf[i_sys].append(1/2*np.sum(((av_g[i_type]-data.gexp[i_sys][i_type][:,0])/data.gexp[i_sys][i_type][:,1])**2))
            lambda_dot_avg+=np.matmul(lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]],av_g[i_type])
        if if_uNOEs: # last element: n_type-1
            i_type+=1#indices[1]
            av_g.append(np.einsum('i,ij',newweights,data.g[i_sys][i_type]))
            errorf[i_sys].append(1/2*np.sum((np.maximum(av_g[i_type]-data.gexp[i_sys][i_type][:,0],np.zeros(len(av_g[i_type])))/data.gexp[i_sys][i_type][:,1])**2))
            lambda_dot_avg+=np.matmul(lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]],av_g[i_type])
        ###

        stats.Srel.append(lambda_dot_avg+np.log(Zpar)-shift)

        lossf_single[i_sys]=np.sum(errorf[i_sys])-alpha*np.sum(stats.Srel[i_sys])

        if if_gradient:

            for i_type in range(indices[0],indices[1]):

                vec1=(av_g[i_type]-data.gexp[i_sys][i_type][:,0])/data.gexp[i_sys][i_type][:,1]**2-alpha*lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]]
                
                if if_natgrad:
                    grad.append(-vec1)
                else:
                    vect=np.matmul(data.g[i_sys][i_type],vec1)
                    scal1=np.matmul(vec1,av_g[i_type])
                    grad_single=-np.matmul(data.g[i_sys][i_type].T*newweights,vect)+scal1*av_g[i_type]
                    grad.append(grad_single)

            if if_uNOEs:
                i_type+=1
                
                vec1=np.maximum(av_g[i_type]-data.gexp[i_sys][i_type][:,0],np.zeros(len(av_g[i_type])))/data.gexp[i_sys][i_type][:,1]**2-alpha*lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]]
                
                if if_natgrad:
                    grad.append(-vec1)
                else:
                    vect=np.matmul(data.g[i_sys][i_type],vec1)
                    scal1=np.matmul(vec1,av_g[i_type])
                    grad_single=-np.matmul(data.g[i_sys][i_type].T*newweights,vect)+scal1*av_g[i_type]
                    grad.append(grad_single)
    
    lossf=np.sum(lossf_single)
    
    if if_gradient:
        grad=np.array([item for sublist in grad for item in sublist])
        #print('grad: ',grad[0:3])
        return lossf,grad,lossf_single,errorf,stats
    return lossf,lossf_single,errorf,stats

def gamma_new(lambdas,data,alpha,if_gradient):

    # 1. initialization

    nsystems=len(data.g)

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
        
        newweights,Zlambda,shift=compute_newweights(lambdas[js_sys[i_sys]:js_sys[i_sys+1]],data.g[i_sys],data.weights[i_sys])#,indices)#js,i_sys,nsystems)
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

def alphalossf_test(lambdas,data,datatrain,datatest,alpha,if_uNOEs,if_only,indices): # data.weights .f .g .gexp
    
    # 1. initialization

    nsystems=len(data.g)

    # compute js and js_sys: indices for lambda corresponding to different systems and types of observables
    # (for training set: the observables with corresponding lambdas)
    js=[]
    for i_sys in range(nsystems):
        js.append([])
        for i_type in range(indices[-1]):
            js[i_sys].append(len(datatrain.g[i_sys][i_type].T))
        if if_uNOEs:
            i_type+=1
            js[i_sys].append(len(datatrain.g[i_sys][i_type].T))
    js_sys=np.sum(js,axis=1)
    js_sys=[0]+np.cumsum(js_sys).tolist()
    js=[0]+np.cumsum(js).tolist()

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

    # 2. for over different systems
    for i_sys in range(nsystems):

        # 1: used observables, non-used frames
        
        newweights,Zpar,shift=compute_newweights(lambdas[js_sys[i_sys]:js_sys[i_sys+1]],datatest.g2[i_sys],datatest.w2[i_sys])#,indices)#js,i_sys,nsystems)

        stats.kish.append(np.sum(newweights**2))
        stats.relkish.append(np.sum(newweights**2/datatest.w2[i_sys])*np.sum(datatest.w2[i_sys])) # normalized w,weights
        
        ##weighted_f=newweights[:,None]*np.array(data.f[i_sys])
        ##av_f=np.sum(weighted_f,axis=0)

        ##stats.Srel.append(np.matmul(par_sys.T,av_f)+np.log(Zpar)-shift)

        #n_type=len(data.g[i_sys])
        #if if_uNOEs: ntype-=1
        
        ### put BY HAND the error function (1/2 chi2 or uNOEs) !!
        av_g=[]
        lambda_dot_avg=0
        ntypes=len(data.g[i_sys])
        for i_type in range(indices[0],indices[1]):#n_type-1):
            av_g.append(np.einsum('i,ij',newweights,datatest.g2[i_sys][i_type]))
            lambda_dot_avg+=np.matmul(lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]],av_g[i_type])
            errorf2[i_sys].append(1/2*np.sum(((av_g[i_type]-datatrain.gexp[i_sys][i_type][:,0])/datatrain.gexp[i_sys][i_type][:,1])**2))
        if if_uNOEs: # last element: n_type-1
            i_type+=1#indices[1]
            av_g.append(np.einsum('i,ij',newweights,datatest.g2[i_sys][i_type]))
            lambda_dot_avg+=np.matmul(lambdas[js[i_sys*ntypes+i_type]:js[i_sys*ntypes+i_type+1]],av_g[i_type])
            errorf2[i_sys].append(1/2*np.sum((np.maximum(av_g[i_type]-datatrain.gexp[i_sys][i_type][:,0],np.zeros(len(av_g[i_type])))/datatrain.gexp[i_sys][i_type][:,1])**2))
            
        ###

        stats.Srel.append(lambda_dot_avg+np.log(Zpar)-shift)

        lossf_single[i_sys]=np.sum(errorf2[i_sys])-alpha*np.sum(stats.Srel[i_sys])
        

        # 2. non-used observables, all or (if if_only) non-used frames

        if not if_only:
            newweights,Zpar,shift=compute_newweights(lambdas[js_sys[i_sys]:js_sys[i_sys+1]],datatest.g3[i_sys],data.weights[i_sys])#,indices)

        av_g=[]
        lambda_dot_avg=0
        for i_type in range(indices[0],indices[1]):#n_type-1):
            av_g.append(np.einsum('i,ij',newweights,datatest.g1[i_sys][i_type]))
            errorf1[i_sys].append(1/2*np.sum(((av_g[i_type]-datatest.gexp1[i_sys][i_type][:,0])/datatest.gexp1[i_sys][i_type][:,1])**2))
        if if_uNOEs: # last element: n_type-1
            i_type+=1#indices[1]
            av_g.append(np.einsum('i,ij',newweights,datatest.g1[i_sys][i_type]))
            errorf1[i_sys].append(1/2*np.sum((np.maximum(av_g[i_type]-datatest.gexp1[i_sys][i_type][:,0],np.zeros(len(av_g[i_type])))/datatest.gexp1[i_sys][i_type][:,1])**2))
            
        ###

        lossf_single[i_sys]+=np.sum(errorf1[i_sys])

    lossf=np.sum(lossf_single)

    return lossf,lossf_single,errorf1,errorf2,stats

# select training and test set after choice of n. of replicas
# positions: the indices for each different replica
# to be done on each tetramer
# if_only=True if only non-used frames for non-trained observables (rather than all frames)

def select_traintest(data,n_test_replicas,n_test_obs,positions,seed,if_only=False):
    
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
    #datatest.f2=[]

    def datatrain():
        return 0
    datatrain.gexp=[]
    datatrain.g=[]
    for i in range(len(positions)):
        datatrain.gexp.append([])
        datatrain.g.append([])
    datatrain.weights=[]
    #datatrain.f=[]
    ###


    for i_sys in range(len(positions)):
        n_replicas=len(positions[0])
        n_type=len(data.g[i_sys])

        if n_type!=len(n_test_obs):
            print('ntype: ',n_type)
            print('len n test obs: ',len(n_test_obs))
            print('error 1')
            return
        if (n_test_replicas >= n_replicas):# or (n_test_obs >= n_obs): # for each kind
            print('error 2')
            return

        # compute the frame indices for the test set
        choice_rep=np.sort(rng.choice(n_replicas,n_test_replicas,replace=False))
        choice_rep_all.append(choice_rep)
        fin=[]
        for i in range(n_test_replicas):
            fin=np.concatenate((fin,positions[i_sys][choice_rep[i]].flatten()),axis=0)
        fin=np.array(fin).astype(int)
        

        # split gexp, weights, f, g into:
        # train, test1 ('non-trained' obs, all frames), test2 ('trained' obs, 'non-used' frames)

        # split weights into train and test
        datatest.w2.append(data.weights[i_sys][fin])
        datatrain.weights.append(np.delete(data.weights[i_sys],fin))

        # split f into train and test
        #datatest.f2.append(data.f[i_sys].iloc[fin])
        #datatrain.f.append(np.delete(np.array(data.f[i_sys]),fin,axis=0))

        for i_type in range(n_type):
            
            # independent choice for each tetramer
            n_obs=data.gexp[i_sys][i_type].shape[0]
            if n_test_obs[i_type]>=n_obs:
                print('error 3')
                return

            choice_obs=np.sort(rng.choice(n_obs,n_test_obs[i_type],replace=False))
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

# read data, including normalize the observables data

# select force field corrections
# implicit input: weights_dir

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

        # for col in ff_cols (types_ff), sum columns starting with col and save with label col
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

# %% INPUT DATA for the simulation

# - Sequences (which single molecular system do you want to consider?)
# - if_skip=True, step (do you want to skip frames? at which step?)
# - userdoc (where do you want to save the results?)
# - weights_dir (where are the original weights? '' if uniform weights)
# - curr_dir_names, curr_dir_obs (where are the names and observables of the molecules?)
#
# almost fixed:
# n_systems=1, if_uNOEs=True, indices=[0,1], n_types=2, n_replicas_test=8, frac_obs_test=0.3
# min_method='L-BFGS-B', gtol=1e-5, if_startzero=True, if_saveanycase=True, if_samefolder=True, if_natgrad=False

#Sequences=['AAAA']#['AAAA','CAAU','CCCC','GACC','UUUU']
Sequences=[molecule] # one molecule at a time, otherwise you have to modify select_traintest nobs_test and bounds

n_systems=1

if_uNOEs=True

if_skip=True
if if_skip==False:
    step=None
else:
    step=10

#userdoc = os.path.join(os.path.expanduser("~"),'loss_function/no_uNOEs/ER')
#userdoc='results_oligomers/valid_molecules/alphazeta/ER_skip%s_rewFFF' % step
userdoc='results_oligomers/ER_skip%s/whole' % step

# download from Zenodo folder DATA
curr_dir_names='DATA/names/'
path='DATA/'
if_weights = False # if weights are all equals, otherwise specify directory
#weights_dir='results_oligomers/valid_molecules/alphazeta/reweights/FFF_%s.npy'

types_obs_exp=np.array(['backbone1_gamma_3J','backbone2_beta_epsilon_3J','sugar_3J','NOEs','uNOEs'])
types_obs = types_obs_exp
types_angles=np.array(['backbone1_gamma','backbone2_beta_epsilon','sugar'])
path_ff_corrections = 'ff_terms/sincos%s'

path_Karplus = 'original'
if_weights = False

indices=[0,1] # from indices[0] to indices[1] excluded for not uNOEs, indices[1] for uNOEs

n_types=2

#n_repeat=30
#alphas=[1e-2,1e-1,1,10,20,50,100,200,500,1e3,1e4,1e5,1e6]
#alphas=np.flip(alphas)

n_replicas_test=8

# path of demuxing trajectories
path_demuxing = 'DATA/demuxing/replica_temp'

#nobs_test=[10,20] # no uNOEs, uNOEs
frac_obs_test=0.3 # 30%, both for not uNOEs and for uNOEs

min_method='L-BFGS-B' # or...
gtol=1e-5 # same as in alphabeta_run.py

if_startzero=True # if if_startzero then it is useless repeat several times if minimization not successful
#if_startreverse=False

if_saveanycase=True # save results of minimization even if not successful
if_samefolder=True # sure to save in a folder already existing?

if_natgrad=False # if you want to minimize with natural gradient
if if_natgrad:
    tol=1 #e-3
    
    #a=pandas.read_csv('epsilon_natural',index_col=0)
    #a.columns.name=a.index.name
    #a.index.name=a.index[0]
    #a.columns=a.columns.astype(float)
    #a=a.iloc[1:,-1] # ensemble refinement
    #a.index=a.index.astype(float)
    #epsilon=a.loc[alpha]/10
    epsilon=1e-8
    print('epsilon: ',epsilon)
    #del a

# %% load data

data,n_frames,n_exps=load_data(Sequences,path,types_obs,types_obs_exp,types_angles,[],if_weights,if_skip,step,True) # implicit input: weights_dir

print('g: ',data.g)
# weights: original ones (uniform because Temperature Replica Exchange) or reweighted

n_types=len(data.gexp[0])
print('n types: ',n_types)

#if not if_uNOEs:
#    n_exps=np.array(n_exps)[:,:-1]

#print('gexp: ',data.gexp[0])
#n_types=len(data.gexp[0])
n_systems=len(data.gexp)
n_experiments=np.zeros((n_systems,n_types))
for i in range(n_systems):
    for j in range(n_types):
        n_experiments[i,j]=len(data.gexp[i][j])
print(n_experiments)

print(n_exps)

n_exp=np.sum(n_exps)

if if_uNOEs:
    a=np.array([np.sum(np.array(n_exps)[:,:-1],axis=1),np.array(n_exps)[:,-1]]).T
    nobs_test=np.rint(frac_obs_test*a).astype(int)
else:
    a=np.array([np.sum(np.array(n_exps),axis=1)]).T
    nobs_test=np.rint(frac_obs_test*a).astype(int)

print(nobs_test)#[0][1]) # nobs_test[i_sys][i_type]

n_exptrain=(n_exp-np.sum(nobs_test)).astype(int) 

# %% minimizer function with natural gradient descent (if if_natgrad == True)

def nat_minimizer(tol,epsilon,lambdas0,data,alpha):
    ###

    if_natgrad = True # True means natural gradient descent, False otherwise

    if_save_min=False # True if you want to save the loss function at intermediate steps

    if_gradient=True # to get the gradient (if if_natgrad=-1) or natural gradient (if if_natgrad=0,1)

    Nmax=5000000 # max. n. of iterations (at given epsilon)
    #Nmax_all=4*Nmax # max. n. of iterations (for all the epsilons)
    #epsilon=0.01 # length step: epsilon times the norm of the (natural) gradient
    # tol=0.1 # stop when Nmax is reached or norm of (natural) gradient less than tol

    print('epsilon: ',epsilon)

    if if_save_min: lossfs=[]

    lambdas=+lambdas0

    out=alpha_lossf(lambdas,data,alpha,if_gradient,if_uNOEs,indices,if_natgrad)

    lossf=out[0]
    length=np.linalg.norm(out[1])

    if if_save_min: 
        lossfs.append(lossf)
        print('lossf: ',lossf)
        print('gradient length:',length)
    
    lossf_new=0
    iter=0
    increments=0

    while (iter<Nmax) and (length>tol) and (increments<10):
        
        lambdas-=epsilon*out[1]#/np.linalg.norm(out[1])
        out=alpha_lossf(lambdas,data,alpha,if_gradient,if_uNOEs,indices,if_natgrad)
        lossf_new=out[0]
        length=np.linalg.norm(out[1])

        if if_save_min:
            lossfs.append(lossf_new)
            print('lossf: ',lossf_new)
            print('gradient length:',length)

        if lossf_new>lossf: increments+=(lossf_new-lossf)*(iter+1) # penalize increments, in particular at long times

        else:
            lossf_min=lossf
            lambdas_min=lambdas+epsilon*out[1] # the ones corresponding to lossf

        iter+=1
        lossf=+lossf_new
    
    if length<=tol:
        print('converged with tolerance %f' % tol)
        failed=0
        lambdas_min=lambdas
        lossf_min=lossf
    else:
        print('not converged')
        failed=1

    if if_save_min: return lambdas_min,lossf_min,iter,lossfs
    return lambdas_min,lossf_min,iter,failed

# %%
# if needed, create directory
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
            #n_replicas=24

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

# %% write input data in myfile.txt

if if_first:
    text=open(userdoc+'/myfile%s.txt' % Sequences,'w')

    text.write('data of the simulation:\n')
    text.write('n. of systems: %s\n' % n_systems)
    text.write('systems: %s\n' % Sequences)
    text.write('n. of frames: %s\n' % n_frames)
    text.write('n. of experiments:\n %s\n' % n_exps)
    #text.write('n. repetitions: %i\n' % n_repeat)
    #text.write('alpha values: %s\n' % alphas)
    text.write('n. replicas test: %s\n' % n_replicas_test)
    text.write('n. observables test: %s\n' % nobs_test)
    text.write('minimization method: %s\n' % min_method)
    text.write('start from zero? %s\n' % if_startzero)
    #text.write('start from min. at higher value of alpha? %s\n' % if_startreverse)

    text.close()

# %% if you want to plot minimization steps (intermediate values of the loss function)
if_show=False

if if_show:
    min_steps=[]
    class CallbackFunctor:
        def __init__(self, obj_fun):
            #self.best_fun_vals = [np.inf]
            #self.best_sols = []
            
            self.xs=[]
            self.fun_vals=[]
            
            self.num_calls = 0
            self.obj_fun = obj_fun
                
        def __call__(self, x):#, args):#data_train,alpha,if_gradient,if_uNOEs,indices):
            fun_val = self.obj_fun(x,data,alpha,if_gradient)[0] # data or data_train
            #fun_val = self.obj_fun(x,data,alpha,if_gradient,if_uNOEs,indices)[0]
            print(fun_val)
            self.num_calls += 1
            #if fun_val < self.best_fun_vals[-1]:
                #self.best_sols.append(x)
                #self.best_fun_vals.append(fun_val)
            self.xs.append(x)
            self.fun_vals.append(fun_val)
            
    
        def save_sols(self, filename):
            sols = np.array([sol for sol in self.best_sols])
            np.savetxt(filename, sols)

    

    import matplotlib.pyplot as plt

# %% whole data set
# it uses: alpha_lossf and gamma_new

def compute_wholedataset(alpha,if_first):

    if if_uNOEs:
        bounds=[]
        for i in range(data.g[0][0].shape[1]):
            bounds.append([-np.inf,+np.inf])
        for i in range(data.g[0][1].shape[1]):
            bounds.append([0,+np.inf])
    else: bounds=None

    if not if_first:
        txt_table=open(os.path.join(userdoc,"all_%s_table_%s" % (Sequences[0], alpha)),'a')
        txt_minlambdas=open(os.path.join(userdoc,"all_%s_minlambdas_%s" % (Sequences[0],alpha)),'a')
        txt_kish=open(os.path.join(userdoc,"all_%s_kish_%s" % (Sequences[0],alpha)),'a')

    if if_first: # if if_first, write the title lines and evaluate original ensemble
        # 1. initialization of output data files
        s=[]
        n_types=len(data.gexp[0])
        for i in range(n_systems):
            for j in range(n_types):
                s.append('errorf['+str(i)+']['+str(j)+']')
        table_columns=['alpha', 'time', 'niter', 'nfail', 'train_lossf']+['train_lossf[%s]' %i for i in range(n_systems)]+s+['train_Srel[%s]' %i for i in range(n_systems)]

        txt_table=open(os.path.join(userdoc,"all_%s_table_%s" % (Sequences[0],alpha)),'a')
        np.savetxt(txt_table,table_columns,fmt='%s',newline=',')

        txt_minlambdas=open(os.path.join(userdoc,"all_%s_minlambdas_%s" % (Sequences[0],alpha)),'a')

        kishes_columns=['alpha']+['kish[%s]' %i for i in range(n_systems)]+['relkish[%s]' %i for i in range(n_systems)]
        txt_kish=open(os.path.join(userdoc,"all_%s_kish_%s" % (Sequences[0],alpha)),'a')
        np.savetxt(txt_kish,kishes_columns,fmt='%s',newline=',')

        # 2. starting values

        if if_startzero:
            lambda0=np.zeros(n_exp)
        #if if_startreverse:
        #    minlambda=np.zeros(n_exp)
        if_gradient=False # suppose you are using jax

        # 3. first iteration: original ensemble (formally, alpha infinite)

        minlambda=np.zeros(n_exp)

        lossf,lossf_single,errorf,stats=alpha_lossf(minlambda,data,alpha=0,if_gradient=False,if_uNOEs=if_uNOEs,indices=indices,if_natgrad=False)

        v=[+np.infty,0,0,0,lossf]+lossf_single.tolist()+np.array(errorf).flatten().tolist()+stats.Srel
        kish=stats.kish+stats.relkish

        txt_table.write('\n')
        np.savetxt(txt_table,v,fmt="%.8f",newline=",")
        txt_minlambdas.write('\n')
        np.savetxt(txt_minlambdas,[+np.infty]+minlambda.tolist(),fmt="%.8f",newline=",")
        txt_kish.write('\n')
        np.savetxt(txt_kish,[+np.infty]+kish,fmt="%.8f",newline=",")

    # 4. iterate minimizations over finite alphas


    #print('alpha: ',alpha)
    # for each iteration: minimize on the training set with uNOEs and evaluate the point of minimum also on the test set

    nfail=0
    start=time.time()

    if if_startzero:
        lambda0=np.zeros(n_exp)
        print(n_exp)
    else:
        lambda0=minlambda

    if_gradient=True
    if if_show:
        cb = CallbackFunctor(gamma_new)
    else:
        cb = None
    
    if not if_natgrad:
        mini=minimize(gamma_new, lambda0, args=(data,alpha,if_gradient), method=min_method,jac=True,callback=cb,bounds=bounds,options={'gtol': gtol})#, 'disp': False})
    
        times=time.time()-start

        suc=mini.success
        print('suc: ',suc)
        if not suc:
            nfail+=1

        if if_show:
            min_steps.append(cb.fun_vals)

            plt.figure()
            plt.plot(cb.fun_vals,'.')
            plt.grid()
            plt.show()

        n_iter=mini.nit
        
        minlambda=mini.x
    else:
        [minlambda,lossf_min,n_iter,nfail]=nat_minimizer(tol,epsilon,lambda0,data,alpha)
        times=time.time()-start

    print('done')

    #lossf=mini.fun
    lossf,lossf_single,errorf,stats=alpha_lossf(minlambda,data,alpha,if_gradient=False,if_uNOEs=if_uNOEs,indices=indices,if_natgrad=False)

    # in agreement with table's labels
    v=[alpha,times,n_iter,nfail,lossf]+lossf_single.tolist()+np.array(errorf).flatten().tolist()+stats.Srel
    kish=stats.kish+stats.relkish

    ######### SAVE DATA
    txt_table.write('\n')
    np.savetxt(txt_table,v,fmt="%.8f",newline=",")
    txt_minlambdas.write('\n')        
    np.savetxt(txt_minlambdas,[alpha]+minlambda.tolist(),fmt="%.8f",newline=",")
    txt_kish.write('\n')
    np.savetxt(txt_kish,[alpha]+kish,fmt="%.8f",newline=",")

    txt_table.close()
    txt_minlambdas.close()
    txt_kish.close()

# %% cross validation
# it uses: alpha_lossf, alphalossf_test and gamma_new

def compute_crossvalidation(alpha,seed,if_first):

    if if_uNOEs:
        bounds=[]
        for i in range((n_experiments[0][0]-nobs_test[0][0]).astype(int)):
            bounds.append([-np.inf,+np.inf])
        for i in range((n_experiments[0][1]-nobs_test[0][1]).astype(int)):
            bounds.append([0,+np.inf])
    else: bounds=None

    # 1. initialization of output data files
    if not if_first: # do a (alpha,seed) minimization step
        txt_table=open(os.path.join(userdoc,"%s_table_%s_%s" % (Sequences[0], alpha, seed)),'a')
        txt_minlambdas=open(os.path.join(userdoc,"%s_minlambdas_%s_%s" % (Sequences[0],alpha, seed)),'a')
        txt_kish=open(os.path.join(userdoc,"%s_kish_%s_%s" % (Sequences[0],alpha,seed)),'a')

    else: # title line and reference ensemble, then do the first (alpha,seed) minimization step

        s=[]
        s1=[]
        s2=[]
        for i in range(n_systems):
            for j in range(n_types):
                s.append('errorf['+str(i)+']['+str(j)+']')
                s1.append('errorf1['+str(i)+']['+str(j)+']')
                s2.append('errorf2['+str(i)+']['+str(j)+']')
        table_columns=['alpha', 'seed', 'time', 'niter', 'nfail', 'train_lossf', 'test_lossf']+['train_lossf[%s]' %i for i in range(n_systems)]+['test_lossf[%s]' %i for i in range(n_systems)]+s+s1+s2+['train_Srel[%s]' %i for i in range(n_systems)]+['test_Srel[%s]' %i for i in range(n_systems)]

        txt_table=open(os.path.join(userdoc,"%s_table_%s_%s" % (Sequences[0],alpha,seed)),'a')
        np.savetxt(txt_table,table_columns,fmt='%s',newline=',')

        txt_minlambdas=open(os.path.join(userdoc,"%s_minlambdas_%s_%s" % (Sequences[0],alpha,seed)),'a')

        kishes_columns=['alpha','seed']+['kish[%s]' %i for i in range(n_systems)]+['relkish[%s]' %i for i in range(n_systems)]+['test_kish[%s]' %i for i in range(n_systems)]+['test_relkish[%s]' %i for i in range(n_systems)]
        txt_kish=open(os.path.join(userdoc,"%s_kish_%s_%s" % (Sequences[0],alpha,seed)),'a')
        np.savetxt(txt_kish,kishes_columns,fmt='%s',newline=',')

        test_obs=open(os.path.join(userdoc,"%s_test_obs_%s" % (Sequences[0],seed)),'a')
        test_contraj=open(os.path.join(userdoc,"%s_test_contraj_%s" % (Sequences[0],seed)),'a')

        # 2. starting values

    if if_startzero:
        lambda0=np.zeros(n_exptrain)
    
        #if_gradient=False

        # repeat for each seed

            # 3. first iteration: original ensemble (formally, alpha infinite)
            
        #if if_startreverse:
        #    minlambda=np.zeros(n_exptrain)
    data_train,data_test,choice_obs_all,choice_rep_all=select_traintest(data,n_replicas_test,nobs_test[0].astype(int),positions_all,seed,if_only=False)

    if if_first:
        # save choice_obs_all, choice_rep_all
        test_obs.write('\n')
        np.savetxt(test_obs,[seed]+[item for sublist in choice_obs_all for item in sublist],fmt="%.8f",newline=",")
        test_contraj.write('\n')
        np.savetxt(test_contraj,[seed]+[item for sublist in choice_rep_all for item in sublist],fmt="%.8f",newline=",")


        ##alpha=0

        lossf,lossf_single,errorf,stats=alpha_lossf(lambda0,data_train,alpha=0,if_gradient=False,if_uNOEs=if_uNOEs,indices=indices,if_natgrad=False)
        test_lossf,test_lossf_single,errorf1,errorf2,test_stats=alphalossf_test(lambda0,data,data_train,data_test,alpha=0,if_uNOEs=if_uNOEs,if_only=False,indices=indices)
        
        # in agreement with table's labels
        v=[+np.infty,seed,0,0,0,lossf,test_lossf]+lossf_single.tolist()+test_lossf_single.tolist()+np.array(errorf).flatten().tolist()+np.array(errorf1).flatten().tolist()+np.array(errorf2).flatten().tolist()+stats.Srel+test_stats.Srel
        kish=stats.kish+stats.relkish+test_stats.kish+test_stats.relkish

        txt_table.write('\n')
        np.savetxt(txt_table,v,fmt="%.8f",newline=",")
        txt_minlambdas.write('\n')
        np.savetxt(txt_minlambdas,[+np.infty,seed]+lambda0.tolist(),fmt="%.8f",newline=",")
        txt_kish.write('\n')
        np.savetxt(txt_kish,[+np.infty,seed]+kish,fmt="%.8f",newline=",")

    # 4. iterate minimizations over finite alphas
    
        # for each iteration: minimize on the training set with uNOEs and evaluate the point of minimum also on the test set

    # minimization
    suc=False
    nfail=0
    start=time.time()

    if not if_startzero:
        lambda0=minlambda
    else:
        lambda0=np.zeros(n_exptrain)
        #np.random.normal(size=n_exp)
    
    if_gradient=True
    if if_show:
        cb = CallbackFunctor(alpha_lossf)
    else:
        cb = None
    mini=minimize(gamma_new, lambda0, args=(data_train,alpha,if_gradient),method=min_method,jac=True,callback=cb,bounds=bounds,options={'gtol': gtol})#1e-15, 'disp': False})
    times=time.time()-start
    suc=mini.success
    
    if not suc:
        nfail+=1

    if (suc or if_saveanycase):
        if if_show:
            min_steps.append(cb.fun_vals)

            plt.figure()
            plt.plot(cb.fun_vals,'.')
            plt.grid()
            plt.show()

        n_iter=mini.nit

        print('done')
        
        minlambda=mini.x
        #if if_startreverse: lambdamin=minlambda

        lossf,lossf_single,errorf,stats=alpha_lossf(minlambda,data_train,alpha,if_gradient=False,if_uNOEs=if_uNOEs,indices=indices,if_natgrad=False)
        test_lossf,test_lossf_single,errorf1,errorf2,test_stats=alphalossf_test(minlambda,data,data_train,data_test,alpha,if_uNOEs=if_uNOEs,if_only=False,indices=indices)
        
        # in agreement with table's labels
        v=[alpha,seed,times,n_iter,nfail,lossf,test_lossf]+lossf_single.tolist()+test_lossf_single.tolist()+np.array(errorf).flatten().tolist()+np.array(errorf1).flatten().tolist()+np.array(errorf2).flatten().tolist()+stats.Srel+test_stats.Srel
        kish=stats.kish+stats.relkish+test_stats.kish+test_stats.relkish

        ######### SAVE DATA
        txt_table.write('\n')
        txt_minlambdas.write('\n')
        txt_kish.write('\n')

        np.savetxt(txt_table,v,fmt="%.8f",newline=",")
        np.savetxt(txt_minlambdas,[alpha,seed]+minlambda.tolist(),fmt="%.8f",newline=",")
        np.savetxt(txt_kish,[alpha,seed]+kish,fmt="%.8f",newline=",")
        #np.savetxt(os.path.join(userdoc,"%s_table" % Sequences[0]),v,fmt=",%.8f"*len(v),comments=',',header=','.join(table_columns))

        txt_table.close()
        txt_minlambdas.close()
        txt_kish.close()
        
        if if_first:
            test_obs.close()
            test_contraj.close()

# %% let's compute

if seed==None:
    compute_wholedataset(alpha,if_first)
else:
    compute_crossvalidation(alpha,seed,if_first)