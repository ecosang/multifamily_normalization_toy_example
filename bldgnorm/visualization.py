import numpy as np
import pandas as pd
import arviz as az
import pickle
from bldgnorm.utility import np_softplus
import scipy.stats
import matplotlib.pyplot as plt

__all__=['PostProcessing']

class PostProcessing:
    # Posterior predictive simulation and visualization
    def __init__(self,output_dir=None):
        
        if output_dir is None:
            raise ValueError("output dir shoule be a folder like output/synthetic/result_2020-01-05_2020-01-12/ that contains sample_dict, param_post, dims, input_values pickles.")
        self.output_dir=output_dir 
        self.sample_dict=pickle.load(open(f'{output_dir}sample_dict.pkl','rb'))
        self.param_post=pickle.load(open(f'{output_dir}param_post.pkl','rb'))
        self.dims=pickle.load(open(f'{output_dir}dims.pkl','rb'))
        self.input_values=pickle.load(open(f'{output_dir}input_values.pkl','rb'))
        self.start_date=self.input_values['start_date']
    
    def predict_class(self):
        # predict group/season
        n_I=self.input_values['n_I']
        id_w=self.input_values['id_w']
        id_h=self.input_values['id_h']
        
        y_s=self.input_values['y_s'].flatten()
        x_s=self.input_values['x_s']
        z_pred=[] #predicted z
        z2_pred=[] #predicted 2nd most probable z
        y_s_pred=[] # predicted scaled y
        
        beta0_mean=np.mean(self.sample_dict['beta0'],axis=0)# K*W
        beta1_mean=np.mean(self.sample_dict['beta1'],axis=0)# K*C*W
        sigma_mean=np.mean(self.sample_dict['sigma'],axis=0) # H
        lbmda_mean=np.mean(self.sample_dict['lmbda'],axis=0)# C*W
        pi_mean=np.mean(self.sample_dict['pi'],axis=0) # K*C
        
        for i in np.arange(n_I):
            pi_c=np.dot(pi_mean,lbmda_mean[:,i])
            #                k x W                                                    # beta1 (k*c*w)(c*1) =>   k*w[i] X x =>k*w[i]
            mus_=(beta0_mean[:,id_w[i]]+(np.tensordot(np_softplus(beta1_mean[:,:,id_w[i]]),lbmda_mean[:,i],axes=(1,0))*x_s[i].flatten() ) ).flatten()

            y_pdf=scipy.stats.norm(loc=mus_, scale=np_softplus(sigma_mean[id_h[i]])).pdf(y_s[i])
            probs_=(pi_c*y_pdf)/np.sum(pi_c*y_pdf)
            z_pred.append(np.argsort(probs_)[-1]) 
            z2_pred.append(np.argsort(probs_)[-2])

            y_s_pred.append(mus_)
        z_pred=np.array(z_pred).astype('int')
        z2_pred=np.array(z2_pred).astype('int')
        y_s_pred=np.array(y_s_pred)

        unique_elements, count_elements = np.unique(z_pred, return_counts=True)
        remove_groups=unique_elements[count_elements<=3]
        for ix in np.arange(z_pred.shape[0]):
            if z_pred[ix] in remove_groups:
                z_pred[ix]=z2_pred[ix]
        self.z_pred=z_pred
        self.y_s_pred=y_s_pred
        
    def ppc_all(self):
        # posterior predictive check
        self.predict_class()
        n_samples=self.sample_dict[list(self.sample_dict.keys())[0]].shape[0] # get #samples
        n_W=self.input_values['n_W']
        n_K=self.input_values['n_K']
        n_H=self.input_values['n_H']
        n_I=self.input_values['n_I']
        id_w=self.input_values['id_w'].flatten()
        id_h=self.input_values['id_h'].flatten()
        x_mean=self.input_values['x_val_s_mean'].flatten()
        y_mean=self.input_values['y_val_s_mean'].flatten()
        x_max=self.dims['x_val_max']
        y_max=self.dims['y_val_max']
        lbmda_mean=np.mean(self.sample_dict['lmbda'],axis=0)# C*W
        
        y_rep=np.zeros([n_samples,n_I,n_K])
        for k in range(n_K):
            for i in range(n_I):
                np.random.seed(k*1000+i)
                s_pred=np.argmax(lbmda_mean[:,id_w[i]])
                y_temp=((  (np_softplus(self.sample_dict['beta1'][:,k,s_pred,id_w[i]])*(self.input_values['x_s'][i].flatten()))+self.sample_dict['beta0'][:,k,id_w[i]]+np.random.normal(0,np_softplus(self.sample_dict['sigma'][:,id_h[i]]),size=n_samples))+y_mean[i])*y_max
                y_rep[:,i,k]=y_temp
        self.y_rep=y_rep

        if n_K==5:
            k_colors=['k','r','g','b','y']
        elif n_K==4:
            k_colors=['k','r','g','b']
        elif n_K==3:
            k_colors=['k','r','g','b']
        else:
            raise ValueError("Have n_K 4 or 3 for now.")
        
        self.x_plot=((self.input_values['x_s'].flatten())+x_mean)*x_max
        self.y_plot=((self.input_values['y_s'].flatten())+y_mean)*y_max
            
        for w in np.arange(n_W).astype('int'):
            x_plot=self.x_plot[id_w==w]
            y_plot=self.y_plot[id_w==w]
            z_pred=self.z_pred[id_w==w]
            y_rep=self.y_rep[:,id_w==w,:]
            plot_sort=np.argsort(x_plot)
            plot_k=np.unique(z_pred)
            
            fig, ax = plt.subplots(nrows=1, ncols=1,  figsize=(8,6)) # sharey = "all"
            for k in plot_k:
                ax.plot(x_plot[z_pred==k].flatten(),y_plot[z_pred==k].flatten(),f'{k_colors[k]}x',label=f'Data{k}')
                ax.plot(x_plot[plot_sort], np.median(y_rep[:,plot_sort,k],axis=0),f'{k_colors[k]}--', alpha=0.5,label='Median')
                az.plot_hpd(x_plot[plot_sort], y_rep[:,plot_sort,k],
                            color=f'{k_colors[k]}', plot_kwargs={"ls": "--"},
                            fill_kwargs={"alpha":0.1,"color":f'{k_colors[k]}','label': 'HPD'},ax=ax)
                ax.set(ylabel="Average power [W]",xlabel="Temperature[$^{\circ}$C]")
                ax.legend(fontsize=10,loc='best')
            plt.show()
