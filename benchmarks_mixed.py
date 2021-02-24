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
mixed_data = 1
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


markers = ['o','s','^']
alphas = 4*[1]

iOCs = np.array([1,2,5])
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

iOC_means = np.flip([4.97, 1.92, 1.10])
iOC_stds = np.array([1.84, 1.178, 1.174])/10
D_means = [43.7,20.15,10.07]
D_stds = np.flip(np.array([1.25,3.125,19]))

iOC_strs = ['iOC0.0001', 'iOC0.0002', 'iOC0.0005']
iOCs = [1,2,5]
D_strs = ["D50","D20","D10"]
Ds = [50,20,10]
# Fix dicts
for j,iOC_str in enumerate(iOC_strs):
    iOC_means_barbora[iOC_str] = iOC_means[j]
    iOC_stds_barbora[iOC_str] = iOC_stds[j]
    D_means_barbora[iOC_str] = D_means[j]
    D_stds_barbora[iOC_str] = D_stds[j]
#%% Joint phase plot 
plt.close('all')
plt.figure(figsize=(20,16))
colspan = 7
rowspan = int(1.8*colspan)

s1 = 27
s2 = 27
plt.subplot2grid((s1,s2), (0, 0), colspan=colspan,rowspan=rowspan)

ms = 8
plot_grid = 1
Ds = [50,20,10]
if plot_grid:
    a = 0.1
    for iOC in iOCs:
        plt.axvline(iOC,linestyle='--',alpha=a,color='black')
    for D in Ds:
        plt.axhline(D,linestyle='--',alpha=a,color='black')
    
for j,iOC in enumerate(iOCs):
    aa = 0.35
    for k,D in enumerate(np.flip(Ds)):
        if not ((iOC == 1 and D == 50) or (iOC == 2 and D == 20) or (iOC == 5 and D == 10)):
            continue
        J = j+k
        if J == 0:
            plt.plot(np.array(1*[iOC]), D, marker=markers[j],color='grey',alpha=aa,label='Expected',markersize=ms)
        else:
            plt.plot(np.array(1*[iOC]), D, marker=markers[j],color='grey',alpha=aa,markersize=ms)
        
for jj, pred_str in enumerate(list(pred_dict.keys())):
    for j, iOC_str in enumerate(iOC_strs):
        iOC = iOCs[j]
        alpha = 1
        
        m = markers[j]
        for k, D_str in enumerate(D_strs):
            D = Ds[k]
            if not ((iOC == 1 and D == 50) or (iOC == 2 and D == 20) or (iOC == 5 and D == 10)):
                continue        
            print(iOC,D)       
            
            if pred_str == "our_preds":
                D_str = 'D10'
                iOC_mean = np.mean(pred_dict[pred_str][iOC_str][D_str]["iOC"])
                D_mean = np.mean(pred_dict[pred_str][iOC_str][D_str]["D"])
                iOC_std = np.std(pred_dict[pred_str][iOC_str][D_str]["iOC"])
                D_std = np.std(pred_dict[pred_str][iOC_str][D_str]["D"])
                c = 'tab:blue'
                x_str = ' ANN'           
                
            if pred_str == "barboras_preds":
                iOC_mean = iOC_means_barbora[iOC_str]
                D_mean = D_means_barbora[iOC_str]
                iOC_std = iOC_stds_barbora[iOC_str]
                D_std = D_stds_barbora[iOC_str]
                
                c = 'tab:red'
                x_str = ' Barbora'           
                
            f = 0.05
            plt.errorbar(iOC_mean,D_mean,xerr=iOC_std/2,yerr=D_std/2,marker=m,
                         markersize=0,fmt='.', color=c,alpha=alpha,elinewidth=1,
                         capsize=2)
            #if iOC_str == "iOC0.0005" and D_str == "D50":
            #    plt.plot(iOC_mean,D_mean,marker=m,color=c,alpha=alpha,markeredgecolor='black',
            #             label = x_str,markersize=ms)#r'{:.0f} $\mu$m$^2/$s'.format(D_exp_val)+x_str)
            #             #label = r'iOC: {:.0f}e-4 $\mu$m, D: {:.0f} $\mu$m$^2/$s'.format(iOC_exp_val,D_exp_val)+x_str)
            #else:
            plt.plot(iOC_mean,D_mean,marker=m,color=c,alpha=alpha,markeredgecolor='black',
                         markersize=ms)

    plt.title('Joint predictions on non-mixed dataset',fontsize=14)   
    #plt.legend(fontsize=9)
    plt.xlabel(r'iOC (1e-4$\mu$m)')
    plt.ylabel(r'D $(\mu$m$^2$/s)')
    plt.xscale('log')
    plt.yscale('log')
    
prev_col_idx = np.copy(colspan)

#%% Precision (D)
ANN = []
BBB = []
Ds = np.array([50,20,10])
for j, iOC_str in enumerate(iOC_strs):
    
    ANN.append(D_means_ours[iOC_str][0])
    BBB.append(D_means_barbora[iOC_str])
    
plt.plot(Ds,abs(ANN-Ds)/Ds,color='tab:blue',label='ANN')
plt.plot(Ds,abs(BBB-Ds)/Ds,color='tab:red',label='Barbora')

plt.xlabel(r'D $(\mu$m$^2$/s)')
plt.title(r'$(\hat D - D_{true})/D_{true}$')#' $(\mu$m$^2$/s)')
plt.legend()

#%% Resolution (D)
ANN = []
BBB = []
Ds = np.array([50,20,10])
for j, iOC_str in enumerate(iOC_strs):
    
    ANN.append(D_stds_ours[iOC_str][0])
    BBB.append(D_stds_barbora[iOC_str])
    
plt.plot(Ds,abs(ANN-Ds)/Ds,color='tab:blue',label='ANN')
plt.plot(Ds,abs(BBB-Ds)/Ds,color='tab:red',label='Barbora')

plt.xlabel(r'D $(\mu$m$^2$/s)')
plt.title(r'std(')#' $(\mu$m$^2$/s)')
plt.legend()

#%% Precision (iOC)
D_means_ANN = []
D_means_B = []
iOCs = np.array([1,2,5])
for j, iOC_str in enumerate(iOC_strs):
    D_means_ANN.append(iOC_means_ours['D10'][j])
    D_means_B.append(iOC_means_barbora[iOC_str])
plot(Ds,abs(D_means_ANN-iOCs)/iOCs,color='tab:blue',label='ANN')
plot(Ds,abs(D_means_B-iOCs)/iOCs,color='tab:red',label='Barbora')
plt.xlabel(r'D $(\mu$m$^2$/s)')
plt.title(r'$(\hat iOC - iOC_{true})/iOC_{true}$')#' $(\mu$m$^2$/s)')
plt.legend()

#%% Resolution (iOC)
D_means_ANN = []
D_means_B = []
iOCs = np.array([1,2,5])
for j, iOC_str in enumerate(iOC_strs):
    D_means_ANN.append(iOC_stds_ours['D10'][j])
    D_means_B.append(iOC_stds_barbora[iOC_str])
plot(iOCs,(D_means_ANN),color='tab:blue',label='ANN')
plot(iOCs,(D_means_B),color='tab:red',label='Barbora')
plt.xlabel(r'iOC $(\mu$m)')
plt.ylabel(r'std(iOC) $(\mu$m$^2$/s)')
plt.title(r'std(iOC) ($\mu$m$^2$/s)')
plt.legend()

plt.tight_layout()

