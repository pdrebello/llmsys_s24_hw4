import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import re
import sys

def plot(time_means, time_stds, tokens_means, tokens_stds, time_labels, tokens_labels, fig_name):
    fig, ax_big = plt.subplots(1, 2, figsize=(12, 5))
    ax = ax_big[0]
    ax.bar(np.arange(len(time_means)), time_means, yerr=time_stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_xticks(np.arange(len(time_means)))
    ax.set_xticklabels(time_labels)
    ax.yaxis.grid(True)
    plt.tight_layout()

    ax = ax_big[1]
    ax.bar(np.arange(len(tokens_means)), tokens_means, yerr=tokens_stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Throughput (Tokens per Second')
    ax.set_xticks(np.arange(len(tokens_means)))
    ax.set_xticklabels(tokens_labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    
    plt.savefig(fig_name)
    plt.close(fig)

def extract(filename, search):
    with open(filename) as f:
        for line in f:
            if re.search(search, line):
                dic = {}
                li = line.replace("avg: ","avg:").replace(",","").split()
                dic['avg_time'] = float([x for x in li if ('avg') in x][0].replace("avg:",""))
                dic['std_time'] = float([x for x in li if ('std') in x][0].replace("std:",""))
                dic['avg_tokens'] = float([x for x in li if ('avg') in x][1].replace("avg:",""))
                dic['std_tokens'] = float([x for x in li if ('std') in x][1].replace("std:",""))
                return dic

# Fill the data points here
if __name__ == '__main__':
    rn = extract("../data_parallel_1.txt","raining time")
    mp0 = extract("../data_parallel_2.txt","Rank 0 training time")
    mp1 = extract("../data_parallel_2.txt","Rank 1 training time")
    plot([mp0['avg_time'], mp1['avg_time'], rn['avg_time']],
        [mp0['std_time'], mp1['std_time'], rn['std_time']],
         [mp0['avg_tokens']+ mp1['avg_tokens'], rn['avg_tokens']],
         [mp0['std_tokens']+ mp1['std_tokens'], rn['std_tokens']],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
         ['Data Parallel - 2GPUs', 'Single GPU'],
        'ddp_vs_rn.png')

    pp = extract("../pipeline_parallel.txt","raining time")
    mp = extract("../model_parallel.txt","raining time")
    plot([pp['avg_time'], mp['avg_time']],
        [pp['std_time'], mp['std_time']],
         [pp['avg_tokens'], mp['avg_tokens']],
         [pp['std_tokens'], mp['std_tokens']],
        ['Pipeline Parallel', 'Model Parallel'],
         ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp.png')