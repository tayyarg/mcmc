import tkinter
import numpy as np
import pandas as pd
import scipy as sp


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.switch_backend('TkAgg')

from scipy.stats import norm

np.random.seed(225)

gozlem = np.random.randn(20)
plt.hist(gozlem, bins='auto') 
plt.xlabel('gozlem')
plt.ylabel('gozlem frekansi')
plt.show()

def sonsal_analitik_hesapla(gozlem, x, mu_ilk, sigma_ilk):
    sigma = 1.
    n = len(gozlem)
    mu_sonsal = (mu_ilk / sigma_ilk**2 + gozlem.sum() / sigma**2) / (1. / sigma_ilk**2 + n / sigma**2)
    sigma_sonsal = (1. / sigma_ilk**2 + n / sigma**2)**-1
    return norm(mu_sonsal, np.sqrt(sigma_sonsal)).pdf(x)

x = np.linspace(-1, 1, 500)
sonsal_analitik = sonsal_analitik_hesapla(gozlem, x, 0., 1.)
plt.plot(x, sonsal_analitik)
plt.xlabel('mu')
plt.title('analitik sonsal dagilim')
plt.show()


N = 100000
mu_guncel = 1.0
mu_oncul_mu= 1.0
mu_oncul_ss= 1.0
oneri_genislik= 0.2
sonsal = [mu_guncel]
nkabul = 0
for i in range(N):
    # yeni konum oner
    mu_oneri = norm(mu_guncel, oneri_genislik).rvs()

    # olabilirlik hesapla (herbir gozlem noktasinin olasiligini carparak)
    olabilirlik_guncel = norm(mu_guncel, 1).pdf(gozlem).prod()
    olabilirlik_oneri = norm(mu_oneri, 1).pdf(gozlem).prod()
    
    # guncel ve onerilen mu icin oncul olasiliklari hesapla       
    oncul_guncel = norm(mu_oncul_mu, mu_oncul_ss).pdf(mu_guncel)
    oncul_oneri = norm(mu_oncul_mu, mu_oncul_ss).pdf(mu_oneri)
    
    p_guncel = olabilirlik_guncel * oncul_guncel
    p_oneri = olabilirlik_oneri * oncul_oneri
    
    u = np.random.uniform()

    # oneriyi kabul olasiligini hesapla
    r = p_oneri / p_guncel
    
    #kabul?
    if u<r:
        # pozisyonu guncelle
        mu_guncel = mu_oneri
        nkabul += 1
    
    sonsal.append(mu_guncel)

    # sonsal dagilimi ve onerileri cizdir
    if i==0 or (i+1)%(N/4)==0:
      fig, (ax1) = plt.subplots(ncols=1, figsize=(4, 4))
      x = np.linspace(-3, 3, 5000)
      color = 'g' if nkabul else 'r'

      # sonsal dagilimi hesapla 
      sonsal_analitik = sonsal_analitik_hesapla(gozlem, x, mu_oncul_mu, mu_oncul_ss)
      ax1.plot(x, sonsal_analitik)
      sonsal_guncel = sonsal_analitik_hesapla(gozlem, mu_guncel, mu_oncul_mu, mu_oncul_ss)
      sonsal_oneri = sonsal_analitik_hesapla(gozlem, mu_oneri, mu_oncul_mu, mu_oncul_ss)
      ax1.plot([mu_guncel] * 2, [0, sonsal_guncel], marker='o', color='b')
      ax1.plot([mu_oneri] * 2, [0, sonsal_oneri], marker='o', color=color)
      ax1.set(title='iterasyon %i\nsonsal(mu=%.2f) = %.5f\nsonsal(mu=%.2f) = %.5f' % (i+1, mu_guncel, sonsal_guncel, mu_oneri, sonsal_oneri))
  
print ("Verimlilik = %", 100*nkabul/N)


import seaborn as sns

ax = plt.subplot()
sns.histplot(np.array(sonsal[500:]), ax=ax, label='sonsal kestirim')
x = np.linspace(-1.0, 1.0, 500)
sons = sonsal_analitik_hesapla(gozlem, x, 0, 1)
ax.plot(x, sons, 'g', label='analitik sonsal')
_ = ax.set(xlabel='mu', ylabel='guvenilirlik (inanc)');
ax.legend()
plt.show()

plt.show()

#maksimum oto-korelasyon zamansal gecikme sayisi
maks_zmnslgecikme = 30
plt.acorr(sonsal, detrend=plt.mlab.detrend_mean, maxlags=maks_zmnslgecikme)
plt.xlim(0, maks_zmnslgecikme)
plt.xlabel('zamansal gecikme')
plt.ylabel('oto-korelasyon')
plt.title('analitik sonsal dagilim oto-korelasyon')
plt.show()
