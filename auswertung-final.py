''' Versuch MCC
 Johannes Brinz & Caterina Vanelli
 Betreuer: Laura Meißner
 Datum: 4.12.2020
 Ort: Online'''

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy import optimize
from scipy import stats
import math
import matplotlib.font_manager as fm
import matplotlib.mlab as mlab
from scipy.stats import norm
from scipy.stats import shapiro
import statsmodels.api as sm
import pylab


#Datenimport
A24C = pd.read_csv("A24C.txt", sep = "\t", header = 1, \
    names = ["Index", "Distance[ym]", "Time[s]", "Velocity[ym/s]"])

B24C = pd.read_csv("B24C.txt", sep = "\t", header = 1, \
    names = ["Index", "Distance[ym]", "Time[s]", "Velocity[ym/s]"])

A34C = pd.read_csv("A34C.txt", sep = "\t", header = 1, \
    names = ["Index", "Distance[ym]", "Time[s]", "Velocity[ym/s]"])

B34C = pd.read_csv("B34C.txt", sep = "\t", header = 1, \
    names = ["Index", "Distance[ym]", "Time[s]", "Velocity[ym/s]"])

C24 = A24C.append(B24C)                     #pooling data

C34 = A34C.append(B34C)

#Fit functions
#Gauß
def gauss(x, sigma, mu):
    return (1 / sigma * np.sqrt(np.pi * 2)) * np.exp(-0.5 * ( (x - mu)/ sigma )**2) * 0.25

#Expo
x0 = 0.6
size = 0.2

def expo(x, k, y):
    return np.exp(-k*x) * y

def cdf(x, k, x_0):
    return 1 - np.exp(-k * (x - x_0))

def cdf_umk(y, k, x_0):
    return

#statistical confidence
alpha = 0.05
#Plots
#24°C runlenght
n, bins, patches = plt.hist(C24["Distance[ym]"][0:399], bins = [x0, x0 + size, x0 + 2*size, x0 + 3*size, x0 + 4*size, x0 + 5*size, x0 + 6*size], density = True)

params, params_cov = optimize.curve_fit(expo, bins[0:6], n[0:6])


y = expo(np.linspace(0.5, 2, 100), params[0], params[1])

plt.plot(np.linspace(0.5, 2, 100)+(size/2), y, "r--", linewidth = 2)           #+size/2 weil ich damit die plot kurve in die mitte der bars packe
plt.title('Histogram 24°C run length', fontsize = 15)
plt.xlabel('run length [$\mu m$]', fontsize = 13)
plt.ylabel('probability distribution [$\mu m ^{-1}$]', fontsize = 13)
plt.legend(['fitted pdf, $k$ = ' + str(round(params[0],2)),  "measured distribution"], fontsize = 13)
plt.xticks(ticks=[0.6, 1.0, 1.4, 1.8])
plt.savefig('Plots/Hist24_run.png', dpi=300)
plt.clf()

print("F24 =", params[1])


#24 C comulative
# sort the data:
data_sorted = np.sort(C24["Distance[ym]"][0:399])
# calculate the proportional values of samples
p = 1. * np.arange(len(C24["Distance[ym]"][0:399])) / (len(C24["Distance[ym]"][0:399]) - 1)


params, params_cov = optimize.curve_fit(cdf, data_sorted[50:399], p[50:399])    #fit
perr = np.sqrt(np.diag(params_cov))


# plot the sorted data:
plt.plot(data_sorted[50:399], cdf(data_sorted[50:399], params[0], params[1]), c = "black")
plt.fill_between(data_sorted, p)
plt.xlabel('runlength [$\mu m$]', fontsize = 13)
plt.ylabel('cummulative probability distribution [$\mu m ^{-1}$]', fontsize = 13)
plt.legend(['fitted cdf, $k$ = ' + str(round(params[0],2)),  "measured distribution"], fontsize = 13)
plt.title('commulative probability 24°C run length', fontsize = 15)
plt.text(0, 0.9, "$x_0$ = (" + str(round(params[1], 3)) + "$\pm$" + str(round(perr[1], 4)) + ")$\mu m$", fontsize = 13 )
plt.savefig('Plots/Cum24.png', dpi=300)
plt.clf()


#bootstrap
Distance_mean = []
for i in range(5000):
    Distance = np.random.choice(C24["Distance[ym]"], size = 399)
    Distance_mean.append(Distance.mean())
plt.hist(Distance_mean)
plt.savefig('Plots/Hist24_run_boot.png', dpi=300)
plt.clf()
#Shapiro-Wilk Normality Test
stat, p00 = shapiro(Distance_mean)
print('Statistics=%.3f, p=%.3f' % (stat, p00))
# interpret
if p00 > alpha:
	print('Sample of 24° C - Run length looks Gaussian (fail to reject H0)')
else:
	print('Sample of 24° C - Run length does not look Gaussian (reject H0)')#


#24°C Velocity
n, bins, patches = plt.hist(C24["Velocity[ym/s]"][0:399], bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], density = True)

params, params_cov = optimize.curve_fit(gauss, bins[0:10]+(size/2), n[0:10])
mu, sigma = norm.fit(C24["Velocity[ym/s]"][0:399])

y = gauss(np.linspace(0, 1.75, 100), sigma, mu)

plt.plot(np.linspace(0, 1.75, 100), y, "r--", linewidth = 2)
plt.title('histogram 24°C velocity', fontsize = 15)
plt.xlabel('velocity [$\mu m/s$]', fontsize = 13)
plt.savefig('Plots/Hist24_run.png', dpi=300)
plt.clf()
plt.ylabel('probability distribution [$s/\mu m$]', fontsize = 13)
plt.text( 0, 1.5, "$\sigma$ = " + str(round(sigma, 2)), fontsize = 13 )
plt.axvline(x=mu, ymin=0, ymax=1, c = "black")
plt.xticks(ticks=[0, 0.4, 0.8, 1.2, 1.6, 2.0])
plt.legend(['fitted function',  "$\mu$ =" + str(round(mu,2)), "measured distribution"], fontsize = 13)
plt.savefig('Plots/Hist24_vel.png', dpi=300)
plt.clf()



# Shapiro-Wilk Normality Test 24°
stat0, p0 = shapiro(C24 ["Velocity[ym/s]"])
print('Statistics=%.3f, p=%.3f' % (stat0, p0))
# interpret
if p0 > alpha:
	print('Sample of 24° C - Velocity looks Gaussian (fail to reject H0)')
else:
	print('Sample of 24° C - Velocity does not look Gaussian (reject H0)')

#qq.plot
sm.qqplot(C24 ["Velocity[ym/s]"], line='s')
plt.savefig('Plots/vel24qq.png', dpi=300)
plt.clf()

#34°C runlenght
x0 = 1.0
n, bins, patches = plt.hist(C34["Distance[ym]"][0:399], bins = [x0, x0 + size, x0 + 2*size, x0 + 3*size, x0 + 4*size, x0 + 5*size, x0 + 6*size], density = True)

params, params_cov = optimize.curve_fit(expo, bins[0:6], n[0:6])


y = expo(np.linspace(1, 2.25, 100), params[0], params[1])

plt.plot(np.linspace(1, 2.25, 100)+(size/2), y, "r--", linewidth = 2)
plt.title('histogram 34°C runlength', fontsize = 15)
plt.xlabel('run length [$\mu m$]', fontsize = 13)
plt.ylabel('probability [$\mu m^{-1}$]', fontsize = 13)
plt.legend(['fitted pdf, $k$ = ' + str(round(params[0],2)),  "measured distribution"], fontsize = 13)
plt.xticks(ticks=[1.0, 1.4, 1.8])
plt.savefig('Plots/Hist34_run.png', dpi=300)
plt.clf()

print("F34 =", params[1])

#34 C comulative
# sort the data:
data_sorted = np.sort(C34["Distance[ym]"][0:399])

# calculate the proportional values of samples
p = 1. * np.arange(len(C34["Distance[ym]"][0:399])) / (len(C34["Distance[ym]"][0:399]) - 1)


params, params_cov = optimize.curve_fit(cdf, data_sorted[35:399], p[35:399])    #fit
perr = np.sqrt(np.diag(params_cov))

# plot the sorted data:
plt.plot(data_sorted[35:399], cdf(data_sorted[35:399], params[0], params[1]), c = "black")
plt.fill_between(data_sorted, p)
plt.xlabel('runlength [$\mu m$]', fontsize = 13)
plt.ylabel('cummulative probability distribution [$\mu m ^{-1}$]', fontsize = 13)
plt.title('commulative probability 34°C run length', fontsize = 15)
plt.legend(['fitted cdf, $k$ = ' + str(round(params[0],2)),  "measured distribution"], fontsize = 13)
plt.text(0, 0.7, "$x_0$ = (" + str(round(params[1], 3)) + "$\pm$" + str(round(perr[1], 4)) + ")$\mu m$", fontsize = 13 )
plt.savefig('Plots/Cum34.png', dpi=300)
plt.clf()

#bootstrap
Distance34_mean = []
for i in range(5000):
    Distance34 = np.random.choice(C34["Distance[ym]"], size = 399)
    Distance34_mean.append(Distance34.mean())
plt.hist(Distance34_mean)
plt.savefig('Plots/Hist34_run_boot.png', dpi=300)
plt.clf()
#Shapiro-Wilk Normality Test
stat3, p3 = shapiro(Distance34_mean)
print('Statistics=%.3f, p=%.3f' % (stat3, p3))
# interpret
if p3 > alpha:
	print('Sample of 34° C - Run length looks Gaussian (fail to reject H0)')
else:
	print('Sample of 34° C - Run length does not look Gaussian (reject H0)')#



#34°C Velocity
n, bins, patches = plt.hist(C34["Velocity[ym/s]"][0:399], bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], density = True)

params, params_cov = optimize.curve_fit(gauss, bins[0:10]+(size/2), n[0:10])
mu, sigma = norm.fit(C34["Velocity[ym/s]"][0:399])

y = gauss(np.linspace(0.4, 2.4, 100), sigma, mu)

plt.plot(np.linspace(0.4, 2.4, 100), y, "r--", linewidth = 2)
plt.title('histogram 34°C velocity', fontsize = 15)
plt.xlabel('velocity [$\mu m/s$]', fontsize = 13)
plt.ylabel('probability [$s/\mu m$]', fontsize = 13)
plt.text( 0, 1.75, "$\sigma$ = " + str(round(sigma, 2)), fontsize = 13 )
plt.axvline(x=mu, ymin=0, ymax=1, c = "black")
plt.xticks(ticks=[0, 0.4, 0.8, 1.2, 1.6, 2.0])
plt.legend(['fitted function',  "$\mu$ =" + str(round(mu,2)), "measured distribution"], fontsize = 13)
plt.savefig('Plots/Hist34_vel.png', dpi=300)
plt.clf()

# normality test
stat1, p1 = shapiro(C34 ["Velocity[ym/s]"])
print('Statistics=%.3f, p=%.3f' % (stat1, p1))
#qqplot
sm.qqplot(C34 ["Velocity[ym/s]"], line='s')
plt.savefig('Plots/vel34qq.png', dpi=300)
plt.clf()
# interpret
if p1 > alpha:
	print('Sample of 34° C - Velocity looks Gaussian (fail to reject H0)')
else:
	print('Sample of 34° C - Velocity does not look Gaussian (reject H0)')#

#Compare run length
stats.ttest_ind(Distance34_mean, Distance_mean , equal_var = False)
t_w, p_w = stats.ttest_ind(Distance34_mean, Distance_mean, equal_var = False)
print('Statistics=%.3f, p=%.3f' % (t_w, p_w))
if p_w > alpha:
	print('There is no statistically significant difference between the means of run length for different temperatures (fail to reject H0)')
else:
	print('There is a statistically significant difference between the means of run length for different temperatures(reject H0)')#

#Compare velocities
stats.ttest_ind(C34 ["Velocity[ym/s]"], C24 ["Velocity[ym/s]"] , equal_var = False)
t_w1, p_w1 = stats.ttest_ind(C34 ["Velocity[ym/s]"],C24 ["Velocity[ym/s]"] , equal_var = False)
print('Statistics=%.3f, p=%.3f' % (t_w1, p_w1))
if p_w1 > alpha:
    print('There is no statistically significant difference between the means of velocity for different temperatures (fail to reject H0)')
else:
    print('There is a statistically significant difference between the means of velocity for different temperatures(reject H0)')#
