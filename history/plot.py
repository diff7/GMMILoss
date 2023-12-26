import os
import matplotlib.pyplot as plt


files = os.listdir('./')
files = [f for f in files if 'acc' in f]
print(files)
data = dict()

for f in files:
    with open(f, 'r') as t:
        lines = t.readlines()
    f = f.replace('_model_acc.txt', '')
    data[f] = [float(l.replace('\n', '')) for l in lines]


plt.figure(figsize=(10, 5))

for k in data:
    if 'KL' in k:
        ls = 'dashed'
    elif 'MI' in k:
        ls = 'solid'
    else:
        ls = 'dotted'

    plt.plot(range(len(data[k])), data[k], label=k, ls=ls)


plt.title('5 runs for each experiment separetly')
plt.xlabel(r"epoch", fontsize=10)
plt.ylabel(f"ACC", fontsize=10)
plt.legend()
plt.grid()
plt.savefig(f"./result_3.pdf")
