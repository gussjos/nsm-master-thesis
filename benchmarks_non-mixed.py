#%% Imports
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

#%% Load ANN preds
import copy
mixed_data = 0
if mixed_data:
    D_strs = ["D10"]
    iOC_strs = ["iOC0.0001","iOC0.0002","iOC0.0005"]
    path = '/home/gustaf/Downloads/mixed-preds-ANN_feb-23.pkl'
else:
    D_strs = ["D10","D20","D50"]
    iOC_strs = ["iOC7.5e-05","iOC0.0001","iOC0.0002","iOC0.0005"]
    path = '/home/gustaf/Downloads/non-mixed-preds-full-images.pkl'
    
preds = load_pkl(path)
iOC_stds = []
D_stds = []
fs = 18


markers = ['d','o','s','^']
alphas = 4*[1]

iOCs = np.array([0.75,1,2,5])
Ds = [10,20,50]

#%% Save means/stds of ANN preds


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

pred_dict = {'our_preds':preds,'barboras_preds':[]}

for j, iOC_str in enumerate(iOC_strs):
    for k, D_str in enumerate(D_strs):
    
        m = markers[k]

        iOC_mean = np.mean(preds[iOC_str][D_str]["iOC"])
        D_mean = np.mean(preds[iOC_str][D_str]["D"])
        iOC_std = np.std(preds[iOC_str][D_str]["iOC"])
        D_std = np.std(preds[iOC_str][D_str]["D"])

        D_stds_ours[iOC_str].append(D_std)
        iOC_stds_ours[D_str].append(iOC_std)
        D_means_ours[iOC_str].append(D_mean)
        iOC_means_ours[D_str].append(iOC_mean)


#%% Load Barboras preds
import scipy.io as IO

DmB = IO.loadmat('matlab_scripts/D_means_non-mixed_data_barbora.mat')
DsB = IO.loadmat('matlab_scripts/D_stds_non-mixed_data_barbora.mat')
ImB = IO.loadmat('matlab_scripts/iOC_means_non-mixed_data_barbora.mat')
IsB = IO.loadmat('matlab_scripts/iOC_stds_non-mixed_data_barbora.mat')

iOC_stds_barbora = {}
iOC_means_barbora = {}
D_means_barbora = {}
D_stds_barbora = {}

# Fix dicts
for j,iOC_str in enumerate(iOC_strs):
    D_means_barbora[iOC_str] = DmB['Deff_mean'][j]
    D_stds_barbora[iOC_str] = DsB['Deff_std'][j]
    
for j,D_str in enumerate(D_strs):
    iOC_means_barbora[D_str] = ImB['iOC_mean'][:,j]
    iOC_stds_barbora[D_str] = IsB['iOC_std'][:,j]
#%% Joint phase plot 
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
                c = 'blue'
                x_str = ' ANN'           
                
            if pred_str == "barboras_preds":

                iOC_mean = iOC_means_barbora[D_str][j]*1e4
                D_mean = D_means_barbora[iOC_str][k]
                iOC_std = iOC_stds_barbora[D_str][j]*1e4
                D_std = D_stds_barbora[iOC_str][k]
                
                c = 'red'
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
    plt.xscale('log')
    plt.yscale('log')
    
prev_col_idx = np.copy(colspan)

#%% Plot mean(iOC) 
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
            c = 'blue'
            Ds_y = (np.flip(D_means_ours[iOC_str])-Ds)/Ds
            if iOC == 5:
                plt.plot(Ds,abs(Ds_y),color=c,alpha=alpha,label='iOC={:.0f}e-4 $\mu$m ANN'.format(iOC))
            else:
                plt.plot(Ds,abs(Ds_y),color=c,alpha=alpha)
                
        else:
            c = 'red'
            Ds_y = ((D_means_barbora[iOC_str])-Ds)/Ds
            if iOC == 5:
                plt.plot(Ds,abs(Ds_y),color=c,alpha=alpha,label='iOC={:.0f}e-4 $\mu$m Barbora'.format(iOC))
            else:
                plt.plot(Ds,abs(Ds_y),color=c,alpha=alpha)
        if j==len(iOC_strs)-1:
            plt.xlabel(r'D $(\mu$m$^2$/s)')
        #plt.ylabel(r'$(D-D_{true})/D_{true}$')
        plt.title('iOC: {:.2f}e-4'.format(iOC),fontsize=11,fontweight='bold')
        

#%% Plot std(iOC) 
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
            c = 'blue'
            if iOC == 5:
                plt.plot(Ds,np.flip(D_stds_ours[iOC_str]),color=c,alpha=alpha,label='iOC={:.0f}e-4 $\mu$m ANN'.format(iOC))
            else:
                plt.plot(Ds,np.flip(D_stds_ours[iOC_str]),color=c,alpha=alpha)
                
        else:
            c = 'red'
            if iOC == 5:
                plt.plot(Ds,(D_stds_barbora[iOC_str]),color=c,alpha=alpha,label='iOC={:.0f}e-4 $\mu$m Barbora'.format(iOC))
            else:
                plt.plot(Ds,(D_stds_barbora[iOC_str]),color=c,alpha=alpha)
        if j==len(iOC_strs)-1:
            plt.xlabel(r'D $(\mu$m$^2$/s)')
        #plt.ylabel(r'$(D-D_{true})/D_{true}$')
        plt.title('iOC: {:.2f}e-4'.format(iOC),fontsize=11,fontweight='bold')

#%% Plot mean(D)
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
            c = 'blue'
            if D == 50:
                plt.plot(iOCs,abs(iOCs_y),color=c,alpha=alpha,label='D={:.0f}$\mu$m$^2$/s ANN'.format(D))
            else:
                plt.plot(iOCs,abs(iOCs_y),color=c,alpha=alpha)
        else:
            iOCs_y = (1e4*iOC_means_barbora[D_str]-iOCs)/iOCs
            c = 'red'
            if D == 50:
                plt.plot(iOCs,abs(iOCs_y),color=c,alpha=alpha,label='D={:.0f}$\mu$m$^2$/s Barbora'.format(D))
            else:
                plt.plot(iOCs,abs(iOCs_y),color=c,alpha=alpha)
                
        #plt.ylabel(r'(iOC-iOC$_{true})/$iOC$_{true}$')
        if j==len(D_strs)-1:
            plt.xlabel(r'iOC (1e-4$\mu$m)')
    plt.title('D: {:.0f}'.format(D),fontweight='bold')
    
prev_col_idx = col_idx + colspan

#%% Plot std(D)
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
            c = 'blue'
            if D == 50:
                plt.plot(iOCs,iOC_stds_ours[D_str],color=c,alpha=alpha,label='D={:.0f}$\mu$m$^2$/s ANN'.format(D))
            else:
                plt.plot(iOCs,iOC_stds_ours[D_str],color=c,alpha=alpha)
        else:
            c = 'red'
            if D == 50:
                plt.plot(iOCs,1e4*iOC_stds_barbora[D_str],color=c,alpha=alpha,label='D={:.0f}$\mu$m$^2$/s Barbora'.format(D))
            else:
                plt.plot(iOCs,1e4*iOC_stds_barbora[D_str],color=c,alpha=alpha)
         
        if j == len(D_strs)-1:
            plt.xlabel(r'iOC (1e-4$\mu$m)')
        #plt.ylabel(r'std(iOC) (1e-4$\mu$m)')
    plt.title('D: {:.0f}'.format(D),fontweight='bold')

plt.tight_layout()

