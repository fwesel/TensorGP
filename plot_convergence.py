import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

font_size = 6
label_size = 9

n_fig_per_row = 4
textwidth_pt = 487.8225
points_per_inch = 72.27
textwidth_inches = textwidth_pt / points_per_inch
image_size = textwidth_inches / n_fig_per_row

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",  # Choose the same font family as in your LaTeX document
# })
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

from matplotlib.ticker import ScalarFormatter, LogFormatterExponent, StrMethodFormatter, NullFormatter
M = 10
for D in [2,4,8,16]:
    data = np.load("convergence_D_"+str(D)+".npz")
    P_CPD = data['P_CPD']
    ks_CPD = data['ks_CPD']
    P_TT = data['P_TT']
    ks_TT = data['ks_TT']

    plt.figure(figsize=(image_size,image_size))
    plt.rc('font', size=font_size)

    plt.grid(True, linestyle='dashed', axis='y', linewidth=0.1, which = 'both', dashes=(10, 5))

    plt.plot(P_CPD,np.mean(ks_CPD,axis=1),label='CPD',linewidth=1,zorder=1)
    plt.plot(P_TT,np.mean(ks_TT,axis=1),label='TT',linewidth=1,zorder=3)
    plt.yscale('log')

    current_ylim = plt.ylim()

    plt.fill_between(P_CPD,np.mean(ks_CPD,axis=1)-np.std(ks_CPD,axis=1),np.mean(ks_CPD,axis=1)+np.std(ks_CPD,axis=1),alpha=0.5,zorder=0)
    plt.fill_between(P_TT,np.mean(ks_TT,axis=1)-np.std(ks_TT,axis=1),np.mean(ks_TT,axis=1)+np.std(ks_TT,axis=1),alpha=0.5,zorder=2)

    if D == 2:
        plt.legend(fontsize=font_size)
        plt.ylabel(r'$W^2$',fontsize=label_size,loc='center')
    plt.xlabel(r'$P$',fontsize=label_size,loc='center')
    plt.ylim(current_ylim)
    plt.title(f'$D={D}$', fontsize=label_size)
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))    
    # plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # plt.gca().yaxis.major.formatter._useMathText = True
    # plt.gca().yaxis.major.formatter.set_powerlimits((0, 0))  # Forces scientific notation for all tick labels
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.1e}'))
    # plt.gca().yaxis.set_minor_formatter(NullFormatter())
    # plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    # plt.tight_layout()
    # plt.savefig("convergence_D_"+str(D)+".pdf",bbox_inches="tight",pad_inches=0)
    plt.show()
    plt.close()