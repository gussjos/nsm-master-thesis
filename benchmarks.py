import pickle
import matplotlib.pyplot as plt
import numpy as np

def save_pkl(filepath,file):
    save_file = open(filepath, "wb")
    pickle.dump(file, save_file)
    save_file.close()
    print("Saved file " + filepath)

def load_pkl(filepath):
    load_file = open(filepath, "rb")
    load_file = pickle.load(load_file)
    print("Loaded file " + filepath)
    return load_file

path = '/home/gustaf/Downloads/mixed-preds-full-images.pkl'
preds = load_pkl(path)


#%% Scatterplot
import copy

iOC_strs = ["iOC7.5e-05","iOC0.0001","iOC0.0002","iOC0.0005"]
D_strs = ["D10","D20","D50"]
iOC_stds = []
D_stds = []
fs = 18
        
#%% ... Add Barboras data
import copy
barboras_preds = copy.deepcopy(preds)
barboras_preds["iOC7.5e-05"]["D10"]["D"] = 36.3348
barboras_preds["iOC7.5e-05"]["D20"]["D"] = 44.7275
barboras_preds["iOC7.5e-05"]["D50"]["D"] = 51.3466
barboras_preds["iOC7.5e-05"]["D10"]["iOC"] = 0.00010901
barboras_preds["iOC7.5e-05"]["D20"]["iOC"] = 0.00010656
barboras_preds["iOC7.5e-05"]["D50"]["iOC"] = 0.00010663
barboras_preds["iOC7.5e-05"]["D10"]["D_std"] = 19.4982
barboras_preds["iOC7.5e-05"]["D20"]["D_std"] = 17.0395
barboras_preds["iOC7.5e-05"]["D50"]["D_std"] = 17.4471
barboras_preds["iOC7.5e-05"]["D10"]["iOC_std"] = 2.0601e-05
barboras_preds["iOC7.5e-05"]["D20"]["iOC_std"] = 1.9316e-05
barboras_preds["iOC7.5e-05"]["D50"]["iOC_std"] = 2.2946e-05

barboras_preds["iOC0.0001"]["D10"]["iOC"] = 0.0001165
barboras_preds["iOC0.0001"]["D20"]["iOC"] = 0.00011466
barboras_preds["iOC0.0001"]["D50"]["iOC"] = 0.00011075
barboras_preds["iOC0.0001"]["D10"]["iOC_std"] = 1.8058e-05
barboras_preds["iOC0.0001"]["D20"]["iOC_std"] = 1.5888e-05
barboras_preds["iOC0.0001"]["D50"]["iOC_std"] = 1.5749e-05
barboras_preds["iOC0.0001"]["D10"]["D"] = 27.1404
barboras_preds["iOC0.0001"]["D20"]["D"] = 36.1983
barboras_preds["iOC0.0001"]["D50"]["D"] = 47.5582
barboras_preds["iOC0.0001"]["D10"]["D_std"] = 17.7379
barboras_preds["iOC0.0001"]["D20"]["D_std"] = 15.5178
barboras_preds["iOC0.0001"]["D50"]["D_std"] = 18.7576

barboras_preds["iOC0.0002"]["D10"]["iOC"] = 0.00019032
barboras_preds["iOC0.0002"]["D20"]["iOC"] = 0.00018823
barboras_preds["iOC0.0002"]["D50"]["iOC"] = 0.00017243
barboras_preds["iOC0.0002"]["D10"]["iOC_std"] = 2.0883e-05
barboras_preds["iOC0.0002"]["D20"]["iOC_std"] = 2.2193e-05
barboras_preds["iOC0.0002"]["D50"]["iOC_std"] = 2.0234e-05
barboras_preds["iOC0.0002"]["D10"]["D"] = 14.7349
barboras_preds["iOC0.0002"]["D20"]["D"] = 24.5987
barboras_preds["iOC0.0002"]["D50"]["D"] = 47.6833
barboras_preds["iOC0.0002"]["D10"]["D_std"] = 7.0556
barboras_preds["iOC0.0002"]["D20"]["D_std"] = 8.9485
barboras_preds["iOC0.0002"]["D50"]["D_std"] = 14.4649

barboras_preds["iOC0.0005"]["D10"]["D"] = 11.5343
barboras_preds["iOC0.0005"]["D20"]["D"] = 20.6109
barboras_preds["iOC0.0005"]["D50"]["D"] = 45.6122
barboras_preds["iOC0.0005"]["D10"]["iOC"] = 0.00048937
barboras_preds["iOC0.0005"]["D20"]["iOC"] = 0.00048197
barboras_preds["iOC0.0005"]["D50"]["iOC"] = 0.00045931
barboras_preds["iOC0.0005"]["D10"]["D_std"] = 5.7277
barboras_preds["iOC0.0005"]["D20"]["D_std"] = 5.7076
barboras_preds["iOC0.0005"]["D50"]["D_std"] = 12.5897
barboras_preds["iOC0.0005"]["D10"]["iOC_std"] = 4.9603e-05
barboras_preds["iOC0.0005"]["D20"]["iOC_std"] = 3.6242e-05
barboras_preds["iOC0.0005"]["D50"]["iOC_std"] = 3.6587e-05


#%% 
import scipy.io as IO
DmB = IO.loadmat('matlab_scripts/D_means_non-mixed_data_barbora.mat')
DsB = IO.loadmat('matlab_scripts/D_stds_non-mixed_data_barbora.mat')
ImB = IO.loadmat('matlab_scripts/iOC_means_non-mixed_data_barbora.mat')
IsB = IO.loadmat('matlab_scripts/iOC_stds_non-mixed_data_barbora.mat')

iOC_stds_barbora = {}
iOC_means_barbora = {}
D_means_barbora = {}
D_stds_barbora = {}


#%%
for j,iOC_str in enumerate(iOC_strs):
    D_means_barbora[iOC_str] = DmB['Deff_mean'][j]
    D_stds_barbora[iOC_str] = DsB['Deff_std'][j]
    
for j,D_str in enumerate(D_strs):
    iOC_means_barbora[D_str] = ImB['iOC_mean'][:,j]
    iOC_stds_barbora[D_str] = IsB['iOC_std'][:,j]

#%% ### Run this first ###
plt.figure(figsize=(16,16))
plt.subplot2grid((24,24), (0, 0), colspan=8,rowspan = 16)
comparison_plot = 1


D_stds_ours = {}
D_means_ours = {}
for iOC_str in iOC_strs:
    D_stds_ours[iOC_str] = []
    D_means_ours[iOC_str] = []
    
iOC_stds_ours = {}
iOC_means_ours = {}
for D_str in D_strs:
    iOC_stds_ours[D_str] = []
    iOC_means_ours[D_str] = []

pred_dict = {'our_preds':preds,'barboras_preds':barboras_preds}
mean_dict = copy.deepcopy(pred_dict)
expected_values = copy.deepcopy(preds)

clrs1 = ["tab:blue","tab:orange","tab:green"]
clrs1 = ["blue","orange","green","black"]
markers = ['d','o','s','^']
alphas = 4*[1]#[0.1,0.2,0.5,1]

iOCs = np.array([0.75,1,2,5])
Ds = [10,20,50]
plt.axvline(1,color='black',alpha=0.15,linestyle='--')
plt.axhline(1,color='black',alpha=0.15,linestyle='--')
for jj, pred_str in enumerate(list(pred_dict.keys())):
    for j, iOC_str in enumerate(iOC_strs):
        alpha = alphas[j]
        for k, D_str in enumerate(D_strs):
        
            m = markers[k]
            
            D_exp_val = Ds[k]#D_exp[i]
            iOC_exp_val = iOCs[j]#iOC_exp[j]
            
            v = 0
            if pred_str == "our_preds":

                iOC_mean = np.mean(pred_dict[pred_str][iOC_str][D_str]["iOC"])
                D_mean = np.mean(pred_dict[pred_str][iOC_str][D_str]["D"])
                iOC_std = np.std(pred_dict[pred_str][iOC_str][D_str]["iOC"])
                D_std = np.std(pred_dict[pred_str][iOC_str][D_str]["D"])
                c = 'blue'
                x_str = ' ANN'
                v = 1
            
            elif pred_str == 'barboras_preds':

                iOC_mean = barboras_preds[iOC_str][D_str]["iOC"]*1e4
                D_mean = barboras_preds[iOC_str][D_str]["D"]
                iOC_std = barboras_preds[iOC_str][D_str]["iOC_std"]*1e4
                D_std = barboras_preds[iOC_str][D_str]["D_std"]
                c = 'red'
                x_str = ' Barbora'

            if v == 1:
                D_stds_ours[iOC_str].append(D_std)
                iOC_stds_ours[D_str].append(iOC_std)
                D_means_ours[iOC_str].append(D_mean)
                iOC_means_ours[D_str].append(iOC_mean)
                
            D_mean = abs((D_mean)/D_exp_val)
            D_std = abs((D_std)/D_exp_val)
            iOC_mean = abs((iOC_mean)/iOC_exp_val)
            iOC_std = abs((iOC_std)/iOC_exp_val)
                


            f = 0.05
            
            if iOC_str == "iOC0.0005":
                plt.scatter(iOC_mean,D_mean,color=c,marker=m,s=45,alpha=alpha,edgecolor='black',
                    label = r'iOC: {:.0f}e-4 $\mu$m, D: {:.0f} $\mu$m$^2/$s'.format(iOC_exp_val,D_exp_val)+x_str)
            else:
                plt.scatter(iOC_mean,D_mean,color=c,marker=m,s=45,alpha=alpha,edgecolor='black')
    plt.xlabel(r'iOC/iOC$_{expected}$',fontsize=fs)
    plt.ylabel(r'D/D$_{expected}$',fontsize=fs)

    #plt.title('Predictions on non-mixed dataset',fontsize=fs)   
    diff = 0.5
    #plt.xlim(1-diff,1+diff)
    #plt.ylim(1-diff,1+diff)
    plt.legend(fontsize=8)
plt.close('all')

#%% ### Joint phase plot ###
plt.figure(figsize=(20,16))
colspan = 7
rowspan = int(1.8*colspan)

s1 = 27
s2 = 27


plt.subplot2grid((s1,s2), (0, 0), colspan=colspan,rowspan=rowspan)

ms = 8
plot_grid = 1
if plot_grid:
    a = 0.1
    for iOC in iOCs:
        plt.axvline(iOC,linestyle='--',alpha=a,color='black')
    for D in Ds:
        plt.axhline(D,linestyle='--',alpha=a,color='black')
    
for j,iOC in enumerate(iOCs):
    aa = 0.35
    for k,D in enumerate(Ds):
        J = j+k
        if J == 0:
            plt.plot(np.array(1*[iOC]), D, marker=markers[j],color='grey',alpha=aa,label='Expected',markersize=ms)
        else:
            plt.plot(np.array(1*[iOC]), D, marker=markers[j],color='grey',alpha=aa,markersize=ms)
        
for jj, pred_str in enumerate(list(pred_dict.keys())):
    for j, iOC_str in enumerate(iOC_strs):
        alpha = alphas[j]
        
        m = markers[j]
        for k, D_str in enumerate(D_strs):
            
            D_exp_val = Ds[k]#D_exp[i]
            iOC_exp_val = iOCs[j]#iOC_exp[j]
            
            
            if pred_str == "our_preds":

                iOC_mean = np.mean(pred_dict[pred_str][iOC_str][D_str]["iOC"])
                D_mean = np.mean(pred_dict[pred_str][iOC_str][D_str]["D"])
                iOC_std = np.std(pred_dict[pred_str][iOC_str][D_str]["iOC"])
                D_std = np.std(pred_dict[pred_str][iOC_str][D_str]["D"])
                c = 'tab:blue'
                x_str = ' ANN'           
                
            if pred_str == "barboras_preds":

                iOC_mean = iOC_means_barbora[D_str][j]*1e4
                D_mean = D_means_barbora[iOC_str][k]
                iOC_std = iOC_stds_barbora[D_str][j]*1e4
                D_std = D_stds_barbora[iOC_str][k]
                
                c = 'tab:red'
                x_str = ' Barbora'              
                
            f = 0.05
            plt.errorbar(iOC_mean,D_mean,xerr=iOC_std/2,yerr=D_std/2,marker=m,
                         markersize=0,fmt='.', color=c,alpha=alpha,elinewidth=1,
                         capsize=2)
            if iOC_str == "iOC0.0005" and D_str == "D50":
                plt.plot(iOC_mean,D_mean,marker=m,color=c,alpha=alpha,markeredgecolor='black',
                         label = x_str,markersize=ms)#r'{:.0f} $\mu$m$^2/$s'.format(D_exp_val)+x_str)
                         #label = r'iOC: {:.0f}e-4 $\mu$m, D: {:.0f} $\mu$m$^2/$s'.format(iOC_exp_val,D_exp_val)+x_str)
            else:
                plt.plot(iOC_mean,D_mean,marker=m,color=c,alpha=alpha,markeredgecolor='black',
                         markersize=ms)

    plt.title('Joint predictions on non-mixed dataset',fontsize=14)   
    plt.legend(fontsize=9)
    plt.xlabel(r'iOC (1e-4$\mu$m)')
    plt.ylabel(r'D $(\mu$m$^2$/s)')
    
prev_col_idx = np.copy(colspan)

#%% ### MEAN(iOC) ###
prev_row_idx = -2
col_idx = prev_col_idx + 1 #prev_col_idx +1
colspan = 4
rowspan = 4
for j, iOC_str in enumerate(iOC_strs):
    iOC = iOCs[j]
    row_idx = prev_row_idx + 2
    plt.subplot2grid((s1,s2), (row_idx,col_idx), colspan=colspan,rowspan=rowspan)
    alpha = alphas[j]
    Ds = [50,20,10]
    iOC = iOCs[j]
    prev_row_idx = row_idx + colspan
    for jj, pred_str in enumerate(list(pred_dict.keys())):
        if pred_str == 'our_preds':
            c = 'tab:blue'
            Ds_y = (np.flip(D_means_ours[iOC_str])-Ds)/Ds
            if iOC == 5:
                plt.plot(Ds,abs(Ds_y),color=c,alpha=alpha,label='iOC={:.0f}e-4 $\mu$m ANN'.format(iOC))
            else:
                plt.plot(Ds,abs(Ds_y),color=c,alpha=alpha)
                
        else:
            c = 'tab:red'
            Ds_y = ((D_means_barbora[iOC_str])-Ds)/Ds
            if iOC == 5:
                plt.plot(Ds,abs(Ds_y),color=c,alpha=alpha,label='iOC={:.0f}e-4 $\mu$m Barbora'.format(iOC))
            else:
                plt.plot(Ds,abs(Ds_y),color=c,alpha=alpha)
        if j==len(iOC_strs)-1:
            plt.xlabel(r'D $(\mu$m$^2$/s)')
        #plt.ylabel(r'$(D-D_{true})/D_{true}$')
        plt.title('iOC: {:.2f}e-4'.format(iOC),fontsize=11,fontweight='bold')
        

#%% ### STD(iOC) ###
prev_col_idx = col_idx+colspan
prev_row_idx = -2
col_idx = prev_col_idx + 1
for j, iOC_str in enumerate(iOC_strs):
    iOC = iOCs[j]
    row_idx = prev_row_idx + 2
    plt.subplot2grid((s1,s2), (row_idx,col_idx), colspan=colspan,rowspan=rowspan)
    alpha = alphas[j]
    Ds = [50,20,10]
    iOC = iOCs[j]
    prev_row_idx = row_idx + colspan
    
    alpha = 1
    for jj, pred_str in enumerate(list(pred_dict.keys())):
        if pred_str == 'our_preds':
            c = 'tab:blue'
            if iOC == 5:
                plt.plot(Ds,np.flip(D_stds_ours[iOC_str]),color=c,alpha=alpha,label='iOC={:.0f}e-4 $\mu$m ANN'.format(iOC))
            else:
                plt.plot(Ds,np.flip(D_stds_ours[iOC_str]),color=c,alpha=alpha)
                
        else:
            c = 'tab:red'
            if iOC == 5:
                plt.plot(Ds,(D_stds_barbora[iOC_str]),color=c,alpha=alpha,label='iOC={:.0f}e-4 $\mu$m Barbora'.format(iOC))
            else:
                plt.plot(Ds,(D_stds_barbora[iOC_str]),color=c,alpha=alpha)
        if j==len(iOC_strs)-1:
            plt.xlabel(r'D $(\mu$m$^2$/s)')
        #plt.ylabel(r'$(D-D_{true})/D_{true}$')
        plt.title('iOC: {:.2f}e-4'.format(iOC),fontsize=11,fontweight='bold')

#%% ### MEAN(D) ###
prev_col_idx = col_idx+colspan
prev_row_idx = -2
col_idx = prev_col_idx + 1
for j, D_str in enumerate(D_strs):
    D = Ds[j]
    row_idx = prev_row_idx + 2
    plt.subplot2grid((s1,s2), (row_idx,col_idx), colspan=colspan,rowspan=rowspan)
    alpha = alphas[j]
    Ds = [50,20,10]
    iOC = iOCs[j]
    prev_row_idx = row_idx + colspan
    alpha = 1
    for jj, pred_str in enumerate(list(pred_dict.keys())):
        if pred_str == 'our_preds':
            iOCs_y = (iOC_means_ours[D_str]-iOCs)/iOCs
            c = 'tab:blue'
            if D == 50:
                plt.plot(iOCs,abs(iOCs_y),color=c,alpha=alpha,label='D={:.0f}$\mu$m$^2$/s ANN'.format(D))
            else:
                plt.plot(iOCs,abs(iOCs_y),color=c,alpha=alpha)
        else:
            iOCs_y = (1e4*iOC_means_barbora[D_str]-iOCs)/iOCs
            c = 'tab:red'
            if D == 50:
                plt.plot(iOCs,abs(iOCs_y),color=c,alpha=alpha,label='D={:.0f}$\mu$m$^2$/s Barbora'.format(D))
            else:
                plt.plot(iOCs,abs(iOCs_y),color=c,alpha=alpha)
                
        #plt.ylabel(r'(iOC-iOC$_{true})/$iOC$_{true}$')
        if j==len(D_strs)-1:
            plt.xlabel(r'iOC (1e-4$\mu$m)')
    plt.title('D: {:.0f}'.format(D),fontweight='bold')
    
prev_col_idx = col_idx + colspan

#%% ### STD(D) ###
prev_col_idx = col_idx+colspan
prev_row_idx = -2
col_idx = prev_col_idx + 1
for j, D_str in enumerate(D_strs):
    D = Ds[j]
    row_idx = prev_row_idx + 2
    plt.subplot2grid((s1,s2), (row_idx,col_idx), colspan=colspan,rowspan=rowspan)
    alpha = alphas[j]
    Ds = [50,20,10]
    iOC = iOCs[j]
    prev_row_idx = row_idx + colspan
    alpha = 1
    for jj, pred_str in enumerate(list(pred_dict.keys())):
        if pred_str == 'our_preds':
            c = 'tab:blue'
            if D == 50:
                plt.plot(iOCs,iOC_stds_ours[D_str],color=c,alpha=alpha,label='D={:.0f}$\mu$m$^2$/s ANN'.format(D))
            else:
                plt.plot(iOCs,iOC_stds_ours[D_str],color=c,alpha=alpha)
        else:
            c = 'tab:red'
            if D == 50:
                plt.plot(iOCs,1e4*iOC_stds_barbora[D_str],color=c,alpha=alpha,label='D={:.0f}$\mu$m$^2$/s Barbora'.format(D))
            else:
                plt.plot(iOCs,1e4*iOC_stds_barbora[D_str],color=c,alpha=alpha)
         
        if j == len(D_strs)-1:
            plt.xlabel(r'iOC (1e-4$\mu$m)')
        #plt.ylabel(r'std(iOC) (1e-4$\mu$m)')
    plt.title('D: {:.0f}'.format(D),fontweight='bold')

plt.tight_layout()

