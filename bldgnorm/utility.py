import numpy as np
from theano import tensor as tt
import theano
import pandas as pd
import re
import scipy.stats
__all__=['create_toy_dataset','np_softplus','create_initial_inputs','ordered_sample_converter','create_sample_dict','create_prior_values','create_input_values','weekly_mean','predict_group']


def create_toy_dataset(seed=138855):
    '''Create toy dataset.
    Args:
        seed (int): seed number for random number generation.

    Raises:

    Returns:
        dims (dictionary): dimension of each variable. 
        df (dataframe): created toy data as Pandas dataframe.
    '''
    np.random.seed(seed)
    c_winter_val=10 # heating to transition season cutoff 
    c_summer_val=18.3 # transition to cooling season cutoff
    K_val=4 # maximum number of groups K #  z=0..K-1
    C_val=3 # number of seasons C, s_w=0..C-1
    H_val=40 # number of residential units h=0...H-1
    W_val=9 # number of weeks w=1...W-1
    
    # outdoor air temperature t_out
    t_out_val=np.concatenate(
        (np.random.uniform(-10,c_winter_val-5,size=np.int(W_val/C_val)),
            np.random.uniform(c_winter_val+2,c_summer_val-2,size=np.int(W_val/C_val)),
            np.random.uniform(c_summer_val+5,35,size=np.int(W_val/C_val))))
    
    # s_w values
    s_val=np.concatenate((np.repeat(0,W_val/3),
                            np.repeat(1,W_val/3),
                            np.repeat(2,W_val/3)),axis=None)
    
    # pi_c
    pi_val=np.zeros([K_val,C_val]) # K x C dim
    pi_val[:,0]=np.array([0.25,0.25,0.25,0.25]) # 4 groups in heating season
    pi_val[:,1]=np.array([0.33,0.33,0.34-1e-9,1e-9]) # 3 groups in transition season
    pi_val[:,2]=np.array([0.5,0.5-2*1e-9,1e-9,1e-9]) # 2 groips in summer season.

    # groups assignment z
    z_val=np.zeros([H_val,W_val])#
    for w in np.arange(W_val):
        z_val[:,w]=(np.random.choice(a=np.arange(K_val),size=H_val,p=pi_val[:,s_val[w]]))
    z_val=-np.sort(-z_val,axis=0)
    z_val=z_val.astype('int')
    
    # beta_0
    beta0_val=np.zeros([K_val,W_val])
    for i in np.arange(K_val):
        beta0_val[i,:]=np.random.normal(loc=(0+i),scale=1,size=W_val)
    
    # mu_beta1
    mu_beta1_val=np.random.normal(loc=0,scale=.1,size=[K_val,C_val])
    
    # beta_1
    beta1_val=np.zeros([K_val,C_val,W_val])
    for i in np.arange(K_val):
        for j in np.arange(C_val):
            for k in np.arange(W_val):
                beta1_val[i,j,k]=np.random.lognormal(mean=mu_beta1_val[i,j],sigma=.1,size=1)+i/2.

    # x
    x_val=np.random.normal(loc=25,scale=1.2,size=H_val*W_val).reshape(H_val,W_val)
    # sigma
    sigma_val=1/np.random.gamma(shape=2.,scale=1/1,size=H_val)
    # y~N(mu,sigma)
    mu_val=np.zeros([H_val,W_val])
    # y
    y_val=np.zeros([H_val,W_val])
    
    for i in np.arange(H_val):
        for j in np.arange(W_val):
            mu_val[i,j]=beta0_val[z_val[i,j],j]+beta1_val[z_val[i,j],s_val[j],j]*x_val[i,j]
            y_val[i,j]=np.random.normal(loc=mu_val[i,j],scale=sigma_val[i],size=1)/30.

    id_w_val=np.tile(np.arange(W_val),H_val) 
    id_h_val=np.repeat(np.arange(H_val),W_val) 
    y_val=np.hstack(y_val)   
    x_val=np.hstack(x_val) 
    z_val=np.hstack(z_val) 
    t_out=np.tile(t_out_val,H_val)
    n_I=id_h_val.shape[0]
    n_K=K_val
    n_W=W_val
    n_C=C_val
    n_H=H_val

    df=pd.DataFrame(data={
                    "t_in":x_val,
                    "e":y_val,
                    "z":z_val,
                    "t_out":t_out,
                    "id_w":id_w_val,
                    "id_h":id_h_val,
                    "start_date":pd.Timestamp("2020-01-05",tz="UTC")+pd.Timedelta(days=7)*pd.DataFrame({'id_w':id_w_val})['id_w']
                    })
    df=df.sort_values(by=['start_date','id_w','id_h'],ascending=True)
    df=df.reset_index(inplace=False,drop=True)
    # dimensional data 
    dims={"n_K":n_K,
          "n_C":n_C,
          "n_H":n_H,
          "x_val_max":np.array([np.ceil(np.max(x_val)*1.5)]),
          "y_val_max":np.array([np.ceil(np.max(y_val)*1.5)])} 
    return dims, df


def np_softplus(x):
    '''softplus function
    Args:
        x (float): data

    Returns:
        np.log(np.exp(x)+1)
    '''
    return np.log(np.exp(x)+1)


def weekly_mean(id_w_val,x_val_s,y_val_s):
    '''weekly average of input data to be used for min-max scaling.
    Args:
        id_w_val (numpy array as int): 
        x_val_s (numpy array): scaled x (i.e., x/x_max)
        y_val_s (numpy array): scaled y (i.e., y/y_max)

    Returns:
        x_val_s_mean (numpy array): weekly average of x_val_s
        y_val_s_mean (numpy array): weekly average of y_val_s
    '''
    df_s_xy=pd.DataFrame({"id_w":id_w_val,"x_val_s":x_val_s.flatten(),"y_val_s":y_val_s.flatten()})
    df_s_xy_mean=df_s_xy.groupby(['id_w'],as_index=False).mean().rename(columns={"x_val_s":"x_val_s_mean","y_val_s":"y_val_s_mean"})
    df_s_xy_all=pd.merge(df_s_xy,df_s_xy_mean,on='id_w',how='left')
    x_val_s_mean=df_s_xy_all['x_val_s_mean'].to_numpy()[:,np.newaxis] 
    y_val_s_mean=df_s_xy_all['y_val_s_mean'].to_numpy()[:,np.newaxis] 
    return x_val_s_mean, y_val_s_mean

# function to create prior_values, inputs_values
def create_initial_inputs(df,dims):
    '''Create initial input values and priors for the first week (week 0).
    
    Args:
        dims (dictionary): dimension of each variable. 
        df (dataframe): created toy data as Pandas dataframe.

    Returns:
        prior_values (dictionary): prior values for week 0.
        input_values: input values for week 0.
    '''
    df=df.reset_index(inplace=False,drop=True)
    start_date=df['start_date'][0].strftime('%Y-%m-%d')
    n_K=dims['n_K'] 
    n_C=dims['n_C']
    n_H=dims['n_H'] # Get n_H from input
    x_val_max=dims['x_val_max']
    y_val_max=dims['y_val_max']
    
    # check id_w starts from 0. Find minimum value and make it to 0.
    w_diff=np.min(df['id_w'].to_numpy()) if np.min(df['id_w'].to_numpy())!=0 else 0
    id_w_original=df['id_w'].to_numpy()
    df.loc[:,('id_w')]=id_w_original-w_diff # change it starts from 0
    ws=np.unique(df['id_w']) # find unique w index
    hs=np.unique(df['id_h']) # find unique h index
    
    ws.sort() 
    hs.sort()

    # raise errors
    if not(np.array_equal(ws,np.arange(ws.shape[0]))): raise ValueError("w index should start from 0 with increasement of 1")
    if not(np.array_equal(hs,np.arange(hs.shape[0]))): raise ValueError("h index should start from 0 with increasement of 1")
    
    n_W=ws.shape[0]
    x_val=df['t_in'].to_numpy()[:,np.newaxis]
    y_val=df['e'].to_numpy()[:,np.newaxis]
    t_out_val=df['t_out'].to_numpy()
    id_w_val=df['id_w'].to_numpy().astype('int')
    id_h_val=df['id_h'].to_numpy().astype('int')

    n_I=y_val.shape[0] #total number of data
    n_X=x_val.shape[1]
    ''' 
    Prior values for weeky 0. See the paper. 
    In the ADVI, positive distribution such as Gamma, LogNoraml, HalfNormal get exponential transformation
    Instead of relying on transformation in PyMC3, we explicitly take exp(N(mu,sigma)) for sequential Bayesian update.
    '''
    prior_values={
            "mu_phi_winter":np.array([10.]), #prior_values['mu_phi_summer'] #init 50
            "mu_phi_summer":np.array([18.3]), #prior_values['mu_phi_winter'] #init 60
            "sd_phi_summer":np.array([3.]), #prior_values['sd_phi_summer'] #init 3
            "sd_phi_winter":np.array([3.]), #prior_values['sd_phi_winter'] #init 3
            "mu_pi_alpha0":np.tile([-0.9],n_K-1), # approximattion of gamma (2,4)
            "sd_pi_alpha0":np.tile([0.5],n_K-1), # approximattion of gamma (2,4)
            "mu_pi_alpha1":np.tile([-0.9],n_K-1), # approximattion of gamma (2,4)
            "sd_pi_alpha1":np.tile([.5],n_K-1), # approximattion of gamma (2,4)
            "mu_pi_alpha2":np.tile([-0.9],n_K-1), # approximattion of gamma (2,4)
            "sd_pi_alpha2":np.tile([.5],n_K-1), # approximattion of gamma (2,4)
            "mu_mu_beta1_0":np.linspace(-3.,0.5,n_K), # #linspace from -3. 0.5
            "mu_mu_beta1_1":np.linspace(-3.,0.5,n_K), # #linspace from -3. 0.5
            "mu_mu_beta1_2":np.linspace(-3.,0.5,n_K), # #linspace from -3. 0.5
            "sd_mu_beta1_0":np.repeat(1.,n_K), # 
            "sd_mu_beta1_1":np.repeat(1.,n_K), # 
            "sd_mu_beta1_2":np.repeat(1.,n_K), # 
            "mu_sd":np.tile([-4.0],n_H), # for sigma_h
            "sd_sd":np.tile([0.5],n_H) # for sigma_h
        }

    if x_val_max.shape[0]!=n_X: raise ValueError("Check x_val_max or x dimension")
    x_val_s=x_val/x_val_max #scaled x values
    y_val_s=y_val/y_val_max #scaled y values 
    x_val_s_mean, y_val_s_mean= weekly_mean(id_w_val=id_w_val,x_val_s=x_val_s,y_val_s=y_val_s)
    input_values={
        "n_I":n_I,#(x_val[np.isin(id_w_val,week_number)]).shape[0],
        "n_W":n_W,
        "n_K":n_K,
        "n_C":n_C,
        "n_H":n_H,
        "n_X":n_X,
        "t_out":t_out_val[:,np.newaxis],
        "x_s":x_val_s-x_val_s_mean,
        "y_s":y_val_s-y_val_s_mean,
        "id_w":id_w_val,
        "id_h":id_h_val,
        "x_val_s_mean":x_val_s_mean, # only specific for current data
        "y_val_s_mean":y_val_s_mean, # only specific for current data
        "start_date":start_date
    }
    return prior_values, input_values


# function to create prior_values, inputs_values
def create_input_values(df,dims):
    '''Create input values for Update period. i.e., week>0.
    
    Args:
        dims (dictionary): dimension of each variable. 
        df (dataframe): created toy data as Pandas dataframe.

    Returns:
        input_values (dictionary): input values for week>0.
    '''
    # x_val,x_val_max,y_val,y_val_max,t_out_val,id_w_val,id_h_val,n_H,n_K=4,n_W=1,n_C=3
    start_date=df['start_date'][0].strftime('%Y-%m-%d')
    n_K=dims['n_K']
    n_C=dims['n_C']
    
    x_val_max=dims['x_val_max']
    y_val_max=dims['y_val_max']
    # check id_w starts from 0. Find minimum value and make it to 0.
    w_diff=np.min(df['id_w'].to_numpy()) if np.min(df['id_w'].to_numpy())!=0 else 0
    id_w_original=df['id_w'].to_numpy()
    df.loc[:, ('id_w')]=id_w_original-w_diff # change it starts from 0

    ws=np.unique(df['id_w'].to_numpy())  # find unique w index
    hs=np.unique(df['id_h'].to_numpy()) # find unique h index
    ws.sort()
    hs.sort()
    # Update n_H
    n_H_prev=dims['n_H'] #n_H value from previous calculation 
    n_H_new=(np.max(hs)+1).astype('int') # n_H value from current input data
    n_H=n_H_new if n_H_new>n_H_prev else n_H_prev #choose max value (either increase house number or keep using previous one)
    n_W=ws.shape[0] 
    # w index check
    if not(np.array_equal(ws,np.arange(ws.shape[0]))): raise ValueError("w index should start from 0 with increasement of 1")

    
    x_val=df['t_in'].to_numpy()[:,np.newaxis]
    y_val=df['e'].to_numpy()[:,np.newaxis]
    t_out_val=df['t_out'].to_numpy()
    id_w_val=df['id_w'].to_numpy().astype('int')
    id_h_val=df['id_h'].to_numpy().astype('int')
    n_I=y_val.shape[0]
    n_X=x_val.shape[1]
    
    if x_val_max.shape[0]!=n_X: raise ValueError("Check x_val_max or x dimension")


    x_val_s=x_val/x_val_max #scaled x values
    y_val_s=y_val/y_val_max #scaled y values 
    x_val_s_mean, y_val_s_mean= weekly_mean(id_w_val=id_w_val,x_val_s=x_val_s,y_val_s=y_val_s)

    input_values={
        "n_I":n_I,#(x_val[np.isin(id_w_val,week_number)]).shape[0],
        "n_W":n_W,
        "n_K":n_K,
        "n_C":n_C,
        "n_H":n_H,
        "n_X":n_X,
        "t_out":t_out_val[:,np.newaxis],
        "x_s":x_val_s-x_val_s_mean,
        "y_s":y_val_s-y_val_s_mean,
        "id_w":id_w_val,
        "id_h":id_h_val,
        "x_val_s_mean":x_val_s_mean, # only specific for current data
        "y_val_s_mean":y_val_s_mean, # only specific for current data
        "start_date":start_date
    }
    return input_values



def ordered_sample_converter(osample):
    '''Ordered sample converter.
    The ordered transformation in PyMC3 uses x1,x2,x3 as x1,x1+exp(x2),x1+exp(x2)+exp(x3) to ensure the order.
    We explicitly calculate these values.
    
    Args:
        osample (numpy array)

    Returns:
        osample (numpy array)
    '''
    # convert ordered sample to the original scale
    if len(osample.shape)==2:
        #1d ordered samples
        ordered_dim=osample.shape[1]
        for o in np.arange(1,ordered_dim):
            osample[:,(o)]=osample[:,(o-1)]+np.exp(osample[:,(o)])
    elif len(osample.shape)==3:
        #2d ordered samples
        ordered_dim=osample.shape[1]
        for o in np.arange(1,ordered_dim):
            osample[:,(o),:]=osample[:,(o-1),:]+np.exp(osample[:,(o),:])
    else:
        raise ValueError("check sample again. Neither 1d or 2d ordered samples")
    return osample

def create_sample_dict(samples):
    '''create sample dictionary from advi.approx.sample
    The ordered transformation in PyMC3 uses x1,x2,x3 as x1,x1+exp(x2),x1+exp(x2)+exp(x3) to ensure the order.
    We explicitly calculate these values.
    
    Args:
        samples (PyMC3 sample class)

    Returns:
        sample_dict (dictionary)
    '''
    sample_dict={}
    for ss in samples.varnames:
        if 'ordered__' in ss:
            # ordered variable
            # remove _ordered__ in variable name and convert ordered values to original scale
            #sample_dict.update({re.sub("_ordered__","",ss):ordered_sample_converter(samples[ss])})
            pass
        else:
            # normal variable
            sample_dict.update({ss:samples[ss]})
    return sample_dict
def create_prior_values(sample_dict):
     '''Create prior values for Update period. i.e., week>0.
        For simplicity, we calculate mean and std from the posterior samples at the previous week.
    Args:
        sample_dict (dictionary): samples of previous week posterior as a dictionary format.

    Returns:
        prior_values (dictionary): input values for week>0.
    '''   

    prior_values={
        "mu_phi_winter":np.mean(sample_dict['phi'][:,0],axis=0),
        "mu_phi_summer":np.mean(sample_dict['phi'][:,1],axis=0),
        "sd_phi_winter":np.minimum(np.std(sample_dict['phi'][:,0],axis=0),1.0),
        "sd_phi_summer":np.minimum(np.std(sample_dict['phi'][:,1],axis=0),1.0),
        "mu_pi_alpha0":np.mean(sample_dict['pi_alpha0'],axis=0),
        "sd_pi_alpha0":np.std(sample_dict['pi_alpha0'],axis=0),
        "mu_pi_alpha1":np.mean(sample_dict['pi_alpha1'],axis=0),
        "sd_pi_alpha1":np.std(sample_dict['pi_alpha1'],axis=0),
        "mu_pi_alpha2":np.mean(sample_dict['pi_alpha2'],axis=0),
        "sd_pi_alpha2":np.std(sample_dict['pi_alpha2'],axis=0),
        "mu_mu_beta1_0":np.mean(sample_dict['mu_beta1_0'],axis=0), # nK
        "mu_mu_beta1_1":np.mean(sample_dict['mu_beta1_1'],axis=0), # nK
        "mu_mu_beta1_2":np.mean(sample_dict['mu_beta1_2'],axis=0), # nK
        "sd_mu_beta1_0":np.std(sample_dict['mu_beta1_0'],axis=0), # nK
        "sd_mu_beta1_1":np.std(sample_dict['mu_beta1_1'],axis=0), # nK
        "sd_mu_beta1_2":np.std(sample_dict['mu_beta1_2'],axis=0), # nK
        "mu_sd":np.mean(sample_dict['sigma'],axis=0),
        "sd_sd":np.std(sample_dict['sigma'],axis=0)
    }
    return prior_values

def predict_group(input_values,sample_dict):
    
    '''Predict group (mean prediction) from data and mean of posteriors
    
    Args:
        input_values: input values for the target week
        sample_dict (dictionary): posterior samples for the target week as dictionary format

    Returns:
        z_pred (numpy array): mean prediction of z (group)
        s_pred (numpy array): mean prediction of s (season)
    '''    
    
    n_I=input_values['n_I']
    id_w=input_values['id_w']
    id_h=input_values['id_h']
    y_s=input_values['y_s'].flatten()
    x_s=input_values['x_s']
    z_pred=[]
    z2_pred=[]
    y_s_pred=[]
    s_pred=[]

    # Mean prediction
    beta0_mean=np.mean(sample_dict['beta0'],axis=0)# K*W
    beta1_mean=np.mean(sample_dict['beta1'],axis=0)# K*C*W
    sigma_mean=np.mean(sample_dict['sigma'],axis=0) # H
    lbmda_mean=np.mean(sample_dict['lmbda'],axis=0)# C*W
    pi_mean=np.mean(sample_dict['pi'],axis=0) # K*C

    for i in np.arange(n_I):
        # See Eq. 21.
        pi_c=np.dot(pi_mean,lbmda_mean[:,i])
        mus_=(beta0_mean[:,id_w[i]]+(np.tensordot(np_softplus(beta1_mean[:,:,id_w[i]]),lbmda_mean[:,i],axes=(1,0))*x_s[i].flatten() ) ).flatten()
        y_pdf=scipy.stats.norm(loc=mus_, scale=np_softplus(sigma_mean[id_h[i]])).pdf(y_s[i])
        probs_=(pi_c*y_pdf)/np.sum(pi_c*y_pdf)
        s_pred.append(np.argmax(lbmda_mean[:,id_w[i]]))
        z_pred.append(np.argsort(probs_)[-1])  # 1st most probable group
        z2_pred.append(np.argsort(probs_)[-2]) # 2nd most probable group 
        y_s_pred.append(mus_)
    z_pred=np.array(z_pred).astype('int')
    s_pred=np.array(s_pred).astype('int')
    z2_pred=np.array(z2_pred).astype('int')
    y_s_pred=np.array(y_s_pred)
    unique_elements, count_elements = np.unique(z_pred, return_counts=True)
    remove_groups=unique_elements[count_elements<=3] # remove groups if the number of units is less than 4.
    for ix in np.arange(z_pred.shape[0]):
        if z_pred[ix] in remove_groups:
            z_pred[ix]=z2_pred[ix]
    return z_pred, s_pred
    

