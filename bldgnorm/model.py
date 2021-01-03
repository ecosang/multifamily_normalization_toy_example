import pymc3 as pm
import numpy as np
from theano import tensor as tt
import theano
from bldgnorm.model_utility import *
import pymc3.distributions.transforms as tr

__all__=['normalization_model']

def normalization_model(prior_values,input_values):
    '''Predict group (mean prediction) from data and mean of posteriors
    
    Args:
        prior_values (dictionary): prior values for week 0.
        input_values: input values for week 0.

    Returns:
        advi: PyMC3 ADVI object
    ''' 
    mu_phi_summer=prior_values['mu_phi_summer'].flatten() #
    mu_phi_winter=prior_values['mu_phi_winter'].flatten() #
    sd_phi_summer=prior_values['sd_phi_summer'].flatten() # 
    sd_phi_winter=prior_values['sd_phi_winter'].flatten() #    
    mu_pi_alpha0=prior_values['mu_pi_alpha0'].flatten() # 
    mu_pi_alpha1=prior_values['mu_pi_alpha1'].flatten() # 
    mu_pi_alpha2=prior_values['mu_pi_alpha2'].flatten() # 
    sd_pi_alpha0=prior_values['sd_pi_alpha0'].flatten() # 
    sd_pi_alpha1=prior_values['sd_pi_alpha1'].flatten() # 
    sd_pi_alpha2=prior_values['sd_pi_alpha2'].flatten() # 

    mu_mu_beta1_0=prior_values['mu_mu_beta1_0'].flatten() # 
    mu_mu_beta1_1=prior_values['mu_mu_beta1_1'].flatten() #
    mu_mu_beta1_2=prior_values['mu_mu_beta1_2'].flatten() # 
    sd_mu_beta1_0=prior_values['sd_mu_beta1_0'].flatten() #
    sd_mu_beta1_1=prior_values['sd_mu_beta1_1'].flatten() # 
    sd_mu_beta1_2=prior_values['sd_mu_beta1_2'].flatten() # 

    mu_sd=prior_values['mu_sd'].flatten() #np.tile(-2,30) # n_H
    sd_sd=prior_values['sd_sd'].flatten() #np.tile(0.5,30) # n_H
    # input_values
    n_I=input_values['n_I']
    n_W=input_values['n_W']
    n_K=input_values['n_K']
    n_C=input_values['n_C']
    n_H=input_values['n_H']
    n_X=input_values['n_X']
    
    t_t_out=tt.constant((input_values['t_out'].flatten())[:,np.newaxis])
    t_x=tt.constant(input_values['x_s']) 
    t_y=tt.constant(input_values['y_s'])
    t_id_w_val=tt.constant(input_values['id_w'].flatten(),dtype='int64')
    t_id_h_val=tt.constant(input_values['id_h'].flatten(),dtype='int64')
    
    #fixed everyweek same
    mu_beta0=np.repeat(np.linspace(-.1,.1,n_K)[:,np.newaxis],n_W,axis=1) 
    sd_beta0=np.repeat(np.repeat(.5,n_K)[:,np.newaxis],n_W,axis=1)

    # check dimensions
    model=pm.Model()
    with model:
        # cutpoints for normal
        phi=pm.Normal("phi",
                      mu=np.array([mu_phi_winter,mu_phi_summer]).flatten(),
                      sigma=np.array([sd_phi_winter,sd_phi_summer]).flatten(),
                      testval=np.array([mu_phi_winter,mu_phi_summer]).flatten(),
                      shape=(np.array([mu_phi_winter,mu_phi_summer]).flatten()).shape[0],transform=tr.Ordered())
        # probability at each season
        lmbda=pm.Deterministic("lmbda",(ordered_logit_kernel(mu=t_t_out,cutpoint_low=phi[0],cutpoint_high=phi[1]))) #cx I
        
        pi_alpha0=tt.exp(pm.Normal('pi_alpha0', mu=mu_pi_alpha0, sigma=sd_pi_alpha0,shape=n_K-1)) # alpha~Gamma(a,b)

        pi_alpha1=tt.exp(pm.Normal('pi_alpha1', mu=mu_pi_alpha1, sigma=sd_pi_alpha1,shape=n_K-1)) # alpha~Gamma(a,b)

        pi_alpha2=tt.exp(pm.Normal('pi_alpha2', mu=mu_pi_alpha2, sigma=sd_pi_alpha2,shape=n_K-1)) # alpha~Gamma(a,b)

        pi_=tt.zeros((n_K,n_C))
        pi_=tt.inc_subtensor(pi_[:,0],tr.t_stick_breaking(1e-12).backward(pi_alpha0)) #c=0
        pi_=tt.inc_subtensor(pi_[:,1],tr.t_stick_breaking(1e-12).backward(pi_alpha1))#c=1
        pi_=tt.inc_subtensor(pi_[:,2],tr.t_stick_breaking(1e-12).backward(pi_alpha2))#c=2
        
        
        pi=pm.Deterministic("pi",pi_)

        beta0=pm.Normal('beta0',
                        mu=mu_beta0,
                        sigma=sd_beta0,
                        shape=(n_K,n_W),transform=Ordered2D(),
                        testval=mu_beta0)
         
        mu_beta1_0=pm.Normal("mu_beta1_0",mu=mu_mu_beta1_0,sigma=sd_mu_beta1_0,shape=(n_K),testval=mu_mu_beta1_0,transform=tr.Ordered())
        mu_beta1_1=pm.Normal("mu_beta1_1",mu=mu_mu_beta1_1,sigma=sd_mu_beta1_1,shape=(n_K),testval=mu_mu_beta1_1)
        mu_beta1_2=pm.Normal("mu_beta1_2",mu=mu_mu_beta1_2,sigma=sd_mu_beta1_2,shape=(n_K),testval=mu_mu_beta1_2,transform=tr.Ordered())
        
        mu_beta1_=tt.zeros((n_K,n_C))               
        mu_beta1_=tt.inc_subtensor(mu_beta1_[:,0],mu_beta1_0)
        mu_beta1_=tt.inc_subtensor(mu_beta1_[:,1],mu_beta1_1)
        mu_beta1_=tt.inc_subtensor(mu_beta1_[:,2],mu_beta1_2[::-1]) #inverse order for summer season
        mu_beta1=pm.Deterministic("mu_beta1",mu_beta1_)

        beta1=tt.exp(pm.Normal('beta1',
                               mu=mu_beta1[...,None],
                               sigma=0.1,
                               shape=(n_K,n_C,n_W))) #k x c x X x W

        sigma=tt.exp(pm.Normal("sigma",mu=mu_sd,sigma=sd_sd,shape=n_H))
        # calculate log-likelihood shown in Appendix A
        ll=pm.Deterministic('ll',
                        fn_ll(beta0=beta0,
                              beta1=beta1,
                              sigma=sigma,
                              lmbda=lmbda,
                              pi=pi,
                              x=t_x,
                              y=t_y,
                              id_w_val=t_id_w_val,
                              id_h_val=t_id_h_val,
                              n_K=n_K,
                              n_I=n_I))
        ll_sum=pm.DensityDist(name='ll_sum',
                               logp=fn_ll_sum(ll=ll),
                               observed=1.)
        advi=pm.ADVI() #advi inference 
    return advi
