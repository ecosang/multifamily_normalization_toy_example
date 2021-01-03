import pymc3 as pm
import numpy as np
from theano import tensor as tt
import theano
import pandas as pandas

__all__=['Ordered2D','softmax','ordered_logit_func','ordered_logit_kernel','fn_i','fn_k','fn_ll','fn_ll_sum','stick_breaking']

class Ordered2D(pm.distributions.transforms.ElemwiseTransform):
    '''2D ordered transformation
    https://discourse.pymc.io/t/order-statistics-in-pymc3/617/3
    https://docs.pymc.io/api/distributions/transforms.html#pymc3.distributions.transforms.Ordered
    '''
    name = "ordered"
    def forward(self, x):
        out = tt.zeros(x.shape)
        out = tt.inc_subtensor(out[0,:], x[0,:])
        out = tt.inc_subtensor(out[1:,:], tt.log(x[1:,:] - x[:-1,:]))
        return out
    
    def forward_val(self, x, point=None):
        x, = pm.distributions.distribution.draw_values([x], point=point)
        return self.forward(x)

    def backward(self, y):
        out = tt.zeros(y.shape)
        out = tt.inc_subtensor(out[0,:], y[0,:])
        out = tt.inc_subtensor(out[1:,:], tt.exp(y[1:,:]))
        return tt.cumsum(out, axis=0)

    def jacobian_det(self, y):
        return tt.sum(y[1:,:], axis=0, keepdims=True)


# Theano functions used for Ordered logit model
def softmax(eta):
    return((tt.exp(eta))/(1+tt.exp(eta)))

def ordered_logit_func(mu,cutpoint_low,cutpoint_high):
    softmax_low=softmax(mu-cutpoint_low)
    softmax_high=softmax(mu-cutpoint_high)
    pi=tt.concatenate([(1-softmax_low),
                   (softmax_low-softmax_high),
                   (softmax_high)])
    return (pi+1e-12)/tt.sum(pi+1e-12) # to remove complete zero
def ordered_logit_kernel(mu,cutpoint_low,cutpoint_high):
    output,updates=theano.scan(fn=ordered_logit_func,
                              non_sequences=[cutpoint_low,cutpoint_high],
                              sequences=[mu])
    return output.T #give c x I


# Functions used for Pymc3 model
## for dirichlet process
def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

# Log-likelihood in Theano. See Appendix A
def fn_i(i,beta0_k,beta1_k,sigma,pi_ki,x,y,id_w_val,id_h_val,lmbda):
    w_=id_w_val[i] # w_index  , i is data index
    h_=id_h_val[i] # 
    mu=beta0_k[w_]+(x[i]*tt.dot(lmbda[:,i],beta1_k[:,w_])) # n_X * (n_C * n_C x 1)
    sigma_h=sigma[h_]
    res_logp=(-tt.log(sigma_h**2)-((y[i]-mu)**2)/(sigma_h**2))/2+tt.log(pi_ki[i])
    return res_logp

def fn_k(k,beta0,beta1,sigma,lmbda,pi,x,y,id_w_val,id_h_val,n_K,n_I):
    beta0_k=beta0[k,:] # n_W
    beta1_k=beta1[k,:,:] # n_C x n_W
    pi_k=pi[k,:] # n_C
    pi_ki=tt.dot(pi_k,lmbda) # n_C * n_C x n_I= n_I
    res_k,_=theano.scan(fn_i,
                non_sequences=[beta0_k,beta1_k,sigma,pi_ki,x,y,id_w_val,id_h_val,lmbda],
                sequences=tt.arange(n_I))
    return res_k

def fn_ll(beta0,beta1,sigma,lmbda,pi,x,y,id_w_val,id_h_val,n_K,n_I): #sigma
    ll,_=theano.scan(fn=fn_k,
             non_sequences=[beta0,beta1,sigma,lmbda,pi,x,y,id_w_val,id_h_val,n_K,n_I],
             sequences=tt.arange(n_K))   # K x  i 
    return ll

def fn_ll_sum(ll):
    def logp(value):
        ans=tt.sum(pm.math.logsumexp(ll,axis=(0)))
        return (tt.switch(tt.isnan(ans), -float("inf"), ans)) #discard errorneous values
    return logp

