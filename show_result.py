import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import pyplot
plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')

def get_asr(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    asr = data['attack']
    asr = np.array([a['ASR'] for a in asr])
    return asr

if __name__ == '__main__':
    GCG_asr = []
    M_GCG_asr = []
    slices = [
        np.arange(i*4, i*4+4) for i in range(5)
    ]
    for i in range(1,6):
        fname = f'log/vicuna_std_0{i}.json'
        asr = get_asr(fname)
        GCG_asr.append(asr)
        
        fname = f'log/vicuna_mgcg_0{i}.json'
        asr = get_asr(fname)
        M_GCG_asr.append(asr)
    
    GCG_asr = np.array(GCG_asr)
    M_GCG_asr = np.array(M_GCG_asr)
    
    # GCG_plot = [GCG_asr.mean(0)[s].mean() for s in slices]
    # M_GCG_plot = [M_GCG_asr.mean(0)[s].mean() for s in slices]

    # plt.plot(np.arange(5), GCG_plot)
    # plt.plot(np.arange(5), M_GCG_plot)
    res = [
        
    GCG_asr.mean(1),
    GCG_asr.std(1),
    GCG_asr.max(1),
    
    
    M_GCG_asr.mean(1),
    M_GCG_asr.std(1),
    M_GCG_asr.max(1)
    
    ]
    res = np.array(res).T
    # print(res * 100)
    
    print(GCG_asr.mean(), GCG_asr.mean(1).std())
    print(M_GCG_asr.mean(), M_GCG_asr.mean(1).std())
    
    print(GCG_asr.max(1).mean(), GCG_asr.max(1).std())
    print(M_GCG_asr.max(1).mean(), M_GCG_asr.max(1).std())

    # plt.plot(np.arange(20), GCG_asr.mean(0), color=palette(1))
    # plt.plot(np.arange(20), M_GCG_asr.mean(0), color=palette(2))
    # plt.fill_between(np.arange(20), GCG_asr.mean(0)+GCG_asr.std(0), GCG_asr.mean(0)-GCG_asr.std(0), color=palette(1), alpha=0.2)
    # plt.fill_between(np.arange(20), M_GCG_asr.mean(0)+M_GCG_asr.std(0), M_GCG_asr.mean(0)-M_GCG_asr.std(0), color=palette(2), alpha=0.2)
    # plt.legend(['GCG', 'M_GCG'])
    # plt.savefig('compare_asr.png', dpi=200)
    

        