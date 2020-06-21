import os
import pickle
from matplotlib import pyplot as plt

cwd = os.getcwd() + os.sep
data_path = cwd + 'data' + os.sep

day = '2020-03-11'
res = pickle.load(open(data_path + 'results_' + day + '.pkl', 'rb'))

# ------------------------------------------------------------------ GRID PLOTS
fig1, axes = plt.subplots(2,4, figsize=(10,7))
for key, ax in zip(sorted(res), axes.flatten()):
    print(key, ax)
    ax.plot(res[key]['df'].M, res[key]['df'].iv, '.')
    ax.plot(res[key]['M'], res[key]['smile'])
    ax.text(0.99, 0.99, r'$\tau$ = ' + str(key),
         horizontalalignment='right',
         verticalalignment='top',
         transform=ax.transAxes)
axes.flatten()[0].set_ylabel('implied volatility')
axes.flatten()[4].set_ylabel('implied volatility')
axes.flatten()[4].set_xlabel('moneyness')
axes.flatten()[5].set_xlabel('moneyness')
axes.flatten()[6].set_xlabel('moneyness')
axes.flatten()[7].set_xlabel('moneyness')
plt.tight_layout()
fig1.savefig(data_path + day + '_smiles.png', transparent=True)


fig2, axes = plt.subplots(2,4, figsize=(10,7))
for key, ax in zip(sorted(res), axes.flatten()):
    print(key, ax)
    ax.plot(res[key]['K'][::-1], res[key]['q'])
    ax.text(0.99, 0.99, r'$\tau$ = ' + str(key),
         horizontalalignment='right',
         verticalalignment='top',
         transform=ax.transAxes)
    ax.set_yticks([])
axes.flatten()[0].set_ylabel('risk neutral density')
axes.flatten()[4].set_ylabel('risk neutral density')
axes.flatten()[4].set_xlabel('spot price')
axes.flatten()[5].set_xlabel('spot price')
axes.flatten()[6].set_xlabel('spot price')
axes.flatten()[7].set_xlabel('spot price')
plt.tight_layout()
fig2.savefig(data_path + day + '_RND.png', transparent=True)


fig3, axes = plt.subplots(2,4, figsize=(10,7))
for key, ax in zip(sorted(res), axes.flatten()):
    print(key, ax)
    ax.plot(res[key]['M'], res[key]['smile'])
    ax.plot(res[key]['M'], res[key]['first'])
    ax.plot(res[key]['M'], res[key]['second'])
    ax.text(0.99, 0.01, r'$\tau$ = ' + str(key),
         horizontalalignment='right',
         verticalalignment='bottom',
         transform=ax.transAxes)
    ax.set_yticks([])
axes.flatten()[0].set_ylabel('implied volatility')
axes.flatten()[4].set_ylabel('implied volatility')
axes.flatten()[4].set_xlabel('moneyness')
axes.flatten()[5].set_xlabel('moneyness')
axes.flatten()[6].set_xlabel('moneyness')
axes.flatten()[7].set_xlabel('moneyness')
plt.tight_layout()
fig3.savefig(data_path + day + '_derivatives.png', transparent=True)

for key in sorted(res):
    print(res[key]['K'])


# ----------------------------------------------------------------- TAU PROCESS
for key in res:
    s = res[key]

    fig4, axes = plt.subplots(1,3, figsize=(10,4))
    ax = axes[0]
    ax.plot(s['df'].M, s['df'].iv, '.', c='r')
    ax.plot(s['M'], s['smile'])
    ax.set_xlabel('moneyness')
    ax.set_ylabel('implied volatility')

    ax = axes[1]
    ax.plot(s['M'], s['smile'])
    ax.plot(s['M'], s['first'])
    ax.plot(s['M'], s['second'])
    ax.set_xlabel('moneyness')
    ax.set_ylabel('implied volatility')

    ax = axes[2]
    ax.plot(s['S'], s['q'])
    ax.set_xlabel('spot price')
    ax.set_ylabel(r'risk neutral density')
    ax.set_yticks([])

    plt.tight_layout()

    fig4.savefig(data_path + day + '_' + str(key) + '.png', transparent=True)

# ------------------------------------------------------------------- HISTOGRAM
a = df[df.tau_day == 9]
Sm = a.S.mean()
Smin = a.S.min()
Smax = a.S.max()

fig5, axes = plt.subplots(1,3, figsize=(10,4))
ax = axes[0]
ax.hist(a.S, bins=20)
ax.set_xlabel(r'spot price $S$')
ax.set_ylabel(r'amount')
ax.axvline(Sm, 0, 400, c='r', ls='--')
ax.axvline(Smin, 0, 400, c='b', ls=':')
ax.axvline(Smax, 0, 400, c='b', ls=':')

Smin = a.S.min()
Smax = a.S.max()
ax = axes[1]
bar = a.K.value_counts().reset_index()
ax.bar(list(bar['index']), list(bar.K), width=250)
ax.set_xlabel(r'strike price $K$')
ax.set_ylabel(r'amount')
ax.axvline(Sm, 0, 400, c='r', ls='--')
ax.axvline(Smin, 0, 400, c='b', ls=':')
ax.axvline(Smax, 0, 400, c='b', ls=':')
plt.tight_layout()

ax = axes[2]
ax.plot(a.M, a.iv, 'o', ms=3)
ax.set_xlabel(r'moneyness $M=S/K$')
ax.set_ylabel(r'implied volatility')

fig5.savefig(data_path + day + '_SKhist_9.png', transparent=True)


a = df
Sm = a.S.mean()
Smin = a.S.min()
Smax = a.S.max()

fig6, axes = plt.subplots(1,2, figsize=(10,4))
ax = axes[0]
ax.hist(a.S, bins=20)
ax.set_xlabel(r'spot price $S$')
ax.set_ylabel(r'amount')
ax.axvline(Sm, 0, 400, c='r', ls='--')
ax.axvline(Smin, 0, 400, c='b', ls=':')
ax.axvline(Smax, 0, 400, c='b', ls=':')

Smin = a.S.min()
Smax = a.S.max()
ax = axes[1]
bar = a.K.value_counts().reset_index()
ax.bar(list(bar['index']), list(bar.K), width=250)
ax.set_xlabel(r'strike price $K$')
ax.set_ylabel(r'amount')
ax.axvline(Sm, 0, 400, c='r', ls='--')
ax.axvline(Smin, 0, 400, c='b', ls=':')
ax.axvline(Smax, 0, 400, c='b', ls=':')
pos   = [5000  , 10000, 15000  , 20000, 25000  , 30000, 35000  , 40000]
label = ['5000', ' '  , '10500', ' '  , '20500', ' '  , '30500', ' ']
plt.xticks(pos, label)

# ax = axes[2]
# ax.plot(a.M, a.iv, 'o', ms=3)
# ax.set_xlabel(r'moneyness $M=S/K$')
# ax.set_ylabel(r'implied volatility')
# plt.tight_layout()

fig6.savefig(data_path + day + '_SKhist_all.png', transparent=True)


