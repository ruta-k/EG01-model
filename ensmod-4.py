# Program to make ensslin and gopal-krishna 2001 model in python
# this is modification of ensmod.py to get the calculations without any normalisation.
# this is a modification of ensmod-1.py to further make the calculations easier.
# this is a modificationof ensmod-2.py to automate the parameter space exploration.
# Notes for parameter space exploration
# Ensslin and Gopal-Krishna 2001
# We refer to the phases as 0 - 4 as given in the paper.
# Injection(0) - Expansion(1) - Lurking(2) - Flashing(3) - Fading(4)
# Scenarios are : A = Cocoon at cluster centre
#                 B = Cocoon at cluster periphery
#                 C = Smoking gun
# Injection phase timescale also can be kept the same for all scenarios.
# Tau in phase 1 is 2/3rd of Tau in phase 0.
# In phase 3, delta_t can be in the range 0.03 to 3 Gyr as limited by the 
# practical shock velocities between 300 - 3000 km/s and distance from 0.1 - 1 Mpc
# In Scen A: lurking cannot last beyond 0.1 Gyr.
# In Scen B: phase 1 of expansion is short but the delta_t2 of lurking can be as long as 1 Gyr.
# 
## The b in all the scenarios is the same so that can be fixed.
## Tau0 and Tau1 are related.
## delta_t0 = 0, 
## 


import math
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy import stats
import os
from numpy import *
from pylab import *
from matplotlib import rc, rcParams
rcParams['legend.numpoints'] = 1
from astropy.cosmology import FlatLambdaCDM

####################################################
# Luminosity distance to the source considered
##################################################
z=0.17120 # redshift for A1914
cosmo = FlatLambdaCDM(H0=70, Om0=0.27)   
def getdl(redshift):
	dl = cosmo.luminosity_distance(redshift)
	return dl.value
d = getdl(z)
#print "Luminosity distance to the source = ", d
#quit()
# calculate the redshift dependent quantity
b_cmb=3.250*(1.00+z)**2.00  # equivalent field for CMB
u_c=((b_cmb**2.00)/(8.00*pi))*(0.62460) # energy density for CMB
#############################
bofsrc = 5.0 # micro Gauss
volofsrc = 0.0370#530*265*265 kpc^3 = 0.0370 # Mpc^3
#####################################
###################################################



#reading in the measured spectrum of the relic
def readfile(myspec):
	dat = ascii.read(myspec)
	return dat 
myfile ='A1914_spec.dat'  # col1 =freq col2 = flux col3 = error
spec = readfile(myfile)
#print spec
f_nuobs=np.array(spec['nughz'])
f_err=np.array(spec['fmyerr'])
flx_obs = np.array(spec['fmy'])
#print 'Min nu obs=', min(f_nuobs), 'GHz'
#print 'Max nu obs=', max(f_nuobs), 'GHz'
######################################
# In the parameter space exploration only values at the given frequencies will be calculated.
######################################
# Read in the parameters for the model
# fixed parameters across the scenarios.
b1 =[1.80,1.20,0.0,2.0,0.0]
#############################################

print "======================================="

def getparspace(scen, phase): # usage getparspace(True, False, False, 4) # boolean and integer
# only tau[0] and tau[3] are free for exploration
# tau[0] is nearly equal to typical active timescale for radio galaxies
# tauex0 is the exploration vaviable for tau[0]
# 0.01 < tauex0 < 0.1 Gyr in steps of XXX.
# This range is same across all scenarios.
	if scen =='A':
		scena = True
		scenb = False
		scenc = False
	elif scen =='B':
		scena = False
		scenb = True
		scenc = False
	elif scen =='C':
		scena = False
		scenb = False
		scenc = True
###################################
	if phase == 0:
		lolim = 0.001
		uplim = 0.1
		step = 0.005
		tauex0 = np.arange(lolim, uplim, step)
	else:
#		lolim = 0.015
#		uplim = 0.015
#		step = 0.005
#		tauex0 = np.arange(lolim, uplim, step)		
		tauex0 = [0.005]  # for later phase the sensitivity to the first phase time is low.
#		print tauex0
#		print len(tauex0)
# tauex3 is the exploration variable for tau[3]
	if phase > 2:
		lolim = -0.2
		uplim = -0.01
		step = 0.02
		tauex3 = np.arange(lolim, uplim, step)
	else:
		tauex3=[-0.11]
#	print tauex3
#	print len(tauex3)
############################################
# let deltex be the exploration parameter for del_t.
# deltex1 will be del_t[1]
# deltex2 will be del_t[2]
# deltex3 will be del_t[3]
# deltex4 will be del_t[4]
# In all Scen  0.03 < deltex3 < 3 Gyr
	if phase >2:
		lolim = 0.001
		uplim = abs(max(tauex3)) #0.5  changed to get deltex3 < tau3 to avoid nans in pstar
		step = 0.005
		deltex3 = np.arange(lolim, uplim, step)  # taken only up to 0.5 Gyr here
	else:
		deltex3=[0.1]
#	print "For Scen A/B/C del_t3 is between ", lolim, uplim, step
#	print "Number of params for del_t3 =", len(deltex3)
#deltex3 = np.arange(0.5,3.0,0.5) # the remaining parameter space if needed.
#print deltex3
#print len(deltex3)
# In all Scen  0.001 < deltex1 < 0.1 Gyr
	if phase > 0:
		lolim = 0.001
		uplim = 0.01
		step = 0.001
		deltex1 = np.arange(lolim, uplim, step)  # taken only up to 0.5 Gyr here
	else:
		deltex1 =[0.01]
#	print "For Scen A/B/C del_t1 is between ", lolim, uplim, step
#	print "Number of params for del_t1 =", len(deltex1)
#quit()
	if phase == 4:
		lolim = 0.01
		uplim = 0.1
		step = 0.01
		deltex4 = np.arange(lolim, uplim, step)  # taken only up to 0.5 Gyr here
	else:
		deltex4 = [0.1] # a fixed number
#	print "For Scen A/B/C del_t4 is between ", lolim, uplim, step
#	print "Number of params for del_t4 =", len(deltex4)
# If Scen A then deltex2 < 0.1 Gyr to keep the revived cocoon within detectable range
	if scena == True:
		if phase > 1:
			lolim = 0.001
			uplim = 0.1
			step = 0.01
			deltex2 = np.arange(lolim, uplim, step)   # keep sampling coarse to reduce the number
		else:
			deltex2=[0.1]
#		print "For Scen A del_t2 is between ", lolim, uplim, step
#		print "Number of params for del_t2 =", len(deltex2)
	elif scenb == True: # If Scen B than deltex2 can be upto 1 Gyr 
		if phase > 1:
			lolim = 0.001
			uplim = 1.0
			step = 0.1
			deltex2 = np.arange(lolim, uplim, step) # keep sampling coarse
		else:
			deltex2=[1.0]
#		print "For Scen B del_t2 is between ", lolim, uplim, step
#		print "Number of params for del_t2 =", len(deltex2)
		if phase > 0:
			lolim = 0.01
			uplim = 0.2
			step = 0.05
			deltex1 = np.arange(lolim, uplim, step) # again coarse sampling done
		else:
			deltex1=[0.17]
#		print "For Scen B del_t1 is between ", lolim, uplim, step
#		print "Number of params for del_t1 =", len(deltex1)
	elif scenc ==True:
		if phase >2:
			lolim = 0.01
			uplim = 0.2
			step = 0.03
			deltex2 = np.arange(lolim, uplim, step)
			uplim = abs(max(tauex3))
			deltex3 = np.arange(lolim, uplim, step)	
		else:
			deltex2=[0.1]
#			uplim = abs(max(tauex3))
			deltex3=(0.6)*abs(max(tauex3)) #[0.13]  # just based on the selections in the EG01 paper
#	return tauex0, tauex3, deltex1, deltex2, deltex3, deltex4
	print len(tauex0),len(deltex1),len(deltex2),len(deltex3),len(tauex3),len(deltex4)
	print "Parameter space loops=", len(tauex0)*len(deltex1)*len(deltex2)*len(deltex3)*len(tauex3)*len(deltex4)
	return tauex0, deltex1, deltex2, deltex3, tauex3, deltex4 

########################################################################
# for a finer search around the min chisq parameters

def getfineparspace(scen, phase): # usage getparspace(True, False, False, 4) # boolean and integer
	if scen =='A':
		scena = True
		scenb = False
		scenc = False
	elif scen =='B':
		scena = False
		scenb = True
		scenc = False
	elif scen =='C':
		scena = False
		scenb = False
		scenc = True
# only tau[0] and tau[3] are free for exploration
# tau[0] is nearly equal to typical active timescale for radio galaxies
# tauex0 is the exploration vaviable for tau[0]
# 0.01 < tauex0 < 0.1 Gyr in steps of XXX.
# This range is same across all scenarios.
	if phase == 0:
		lolim = 0.001
		uplim = 0.1
		step = 0.005
		tauex0 = np.arange(lolim, uplim, step)
	else:
		lolim = 0.001
		uplim = 0.15
		step = 0.005
		tauex0 = np.arange(lolim, uplim, step)		
#		tauex0 = [0.015]  # for later phases the sensitivity to the first phase time is low.
#		print tauex0
#		print len(tauex0)
# tauex3 is the exploration variable for tau[3]
	if phase > 2:
		lolim = -0.2
		uplim = -0.01
		step = 0.02
		tauex3 = np.arange(lolim, uplim, step)
	else:   
		tauex3=[-0.11]
#	print tauex3
#	print len(tauex3)
############################################
# let deltex be the exploration parameter for del_t.
# deltex1 will be del_t[1]
# deltex2 will be del_t[2]
# deltex3 will be del_t[3]
# deltex4 will be del_t[4]
# In all Scen  0.03 < deltex3 < 3 Gyr
	if phase >2:
		lolim = 0.001
		uplim = abs(max(tauex3)) #0.5  changed to get deltex3 < tau3 to avoid nans in pstar
		step = 0.002
		deltex3 = np.arange(lolim, uplim, step)  # taken only up to 0.5 Gyr here
	else:   # fixed value removed for finer sampling of parameter space
		lolim = 0.001
		uplim = abs(max(tauex3)) #0.5  changed to get deltex3 < tau3 to avoid nans in pstar
		step = 0.002
		deltex3 = np.arange(lolim, uplim, step)  # taken only up to 0.5 Gyr here
#		deltex3=[0.1]
#	print "For Scen A/B/C del_t3 is between ", lolim, uplim, step
#	print "Number of params for del_t3 =", len(deltex3)
#deltex3 = np.arange(0.5,3.0,0.5) # the remaining parameter space if needed.
#print deltex3
#print len(deltex3)
# In all Scen  0.001 < deltex1 < 0.1 Gyr
	if phase > 0:
#		lolim = 0.001 # for coarse
#		uplim = 0.1   
#		step = 0.01
		lolim = 0.001 # for fine
		uplim = 0.05
		step = 0.001
		deltex1 = np.arange(lolim, uplim, step)  # taken only up to 0.5 Gyr here
	else:  # fixed value removed for finer sampling
#		deltex1 =[0.01]
		lolim = 0.001 # for fine
		uplim = 0.2
		step = 0.001
		deltex1 = np.arange(lolim, uplim, step)  # taken only up to 0.5 Gyr here
#	print "For Scen A/B/C del_t1 is between ", lolim, uplim, step
#	print "Number of params for del_t1 =", len(deltex1)
#quit()
	if phase == 4:
#		lolim = 0.01
#		uplim = 0.1
#		step = 0.01  
		lolim = 0.001
		uplim = 0.1
		step = 0.001  
		deltex4 = np.arange(lolim, uplim, step)  # taken only up to 0.5 Gyr here
	else:
		deltex4 = [0.1] # a fixed number
#	print "For Scen A/B/C del_t4 is between ", lolim, uplim, step
#	print "Number of params for del_t4 =", len(deltex4)
# If Scen A then deltex2 < 0.1 Gyr to keep the revived cocoon within detectable range
	if scena == True:
		if phase > 1:
#			lolim = 0.001 # for coarse search
#			uplim = 0.1
#			step = 0.01
			lolim = 0.001 # for finer search
			uplim = 0.2
			step = 0.005
			deltex2 = np.arange(lolim, uplim, step)   # keep sampling coarse to reduce the number
		else:
			deltex2=[0.1]
#		print "For Scen A del_t2 is between ", lolim, uplim, step
#		print "Number of params for del_t2 =", len(deltex2)
	elif scenb == True: # If Scen B than deltex2 can be upto 1 Gyr 
		if phase > 1:
#			lolim = 0.001   # for coarse search
#			uplim = 1.0
#			step = 0.1
			lolim = 0.001   # for finer search
			uplim = 1.0
			step = 0.01
			deltex2 = np.arange(lolim, uplim, step) # 
#			deltex2 = [0.165] # after finre search found this; now fix this and change tau[0]  for Scen B, phase = 2 
		else:
			deltex2=[1.0]
#		print "For Scen B del_t2 is between ", lolim, uplim, step
#		print "Number of params for del_t2 =", len(deltex2)
		if phase > 0:
			lolim = 0.01
			uplim = 0.2
			step = 0.001
			deltex1 = np.arange(lolim, uplim, step) # again coarse sampling done
		else:
			deltex1=[0.17]
#		print "For Scen B del_t1 is between ", lolim, uplim, step
#		print "Number of params for del_t1 =", len(deltex1)
	elif scenc ==True:
		if phase >2:
			lolim = 0.01
			uplim = 0.2
			step = 0.005
			deltex2 = np.arange(lolim, uplim, step)
			uplim = abs(max(tauex3))
			deltex3 = np.arange(lolim, uplim, step)	
		else:
			deltex2=[0.1]
#			uplim = abs(max(tauex3))
			deltex3=(0.6)*abs(max(tauex3)) #[0.13]  # just based on the selections in the EG01 paper
#	return tauex0, tauex3, deltex1, deltex2, deltex3, deltex4
	print len(tauex0),len(deltex1),len(deltex2),len(deltex3),len(tauex3),len(deltex4)
	print "Parameter space loops=", len(tauex0)*len(deltex1)*len(deltex2)*len(deltex3)*len(tauex3)*len(deltex4)
	return tauex0, deltex1, deltex2, deltex3, tauex3, deltex4 





#quit()

##############################
# For the case considered for fitting, the parameters for the preceding phases need to be back calculated.
# This function back calculates the parameters.

def ph4flashing(nphase, b1, del_t, tau, bsrc, vsrc):
	'''nphase = phase that is being fit '''
#	nphase=4
	pi=math.acos(-1.00)  # pi
	b_mug=[0,0,0,0,0] # 5 elements for total number of phases.
	u_b=[0,0,0,0,0]
	vol=[0,0,0,0,0]
	b_mug[nphase]=bsrc #5.00  # equipartition field for your src in microGauss
	u_b[nphase]=((b_mug[nphase]**2.00)/(8.00*pi))*(0.62460)
	vol[nphase] = vsrc #0.0370 # volume estimated for the src in Mpc^3
	c1=[0,0,0,0,0]
	for j in range(0,5):
		c1[j]=((1.0+(del_t[j]/tau[j]))**(-b1[j]))
#        	print j,c1[j]
	# back calculate volume
	for i in range(0,nphase):
		vol[nphase-i-1]= vol[nphase-i]*c1[nphase-i]
#		print i
#	print vol	
	#get the magnetic energy density in phase 0
	u_b[0]=u_b[nphase]*(vol[nphase]/vol[0])**(4.00/3.00)
	b_mug[0]=sqrt((u_b[0]*8.00*pi)/0.62460)
	return b_mug, u_b, vol, c1

#b_mug, u_b, vol, c1 = ph4flashing(2,b1,del_t,tau)
#print b_mug, u_b, vol, c1
#quit()
###################################
# functions defined for spectal emissivity
def func(x, alpha1, alpha2, p_star, a1):
#	print 'func values=',x,alpha1,alpha2, p_star, a1
#	myexp=math.exp(-a1*(x**(-7.00/4.00)))
#	print 'myexp=', myexp
#	print x**alpha1, ((1.00-(x/p_star))**alpha2), math.exp(-a1*(x**(-7.00/4.00)))
#	print x, p_star, alpha2, ((1.00-(x/p_star))**alpha2)
	myval= (x**alpha1)*((1.00-(x/p_star))**alpha2)*math.exp(-a1*(x**(-7.00/4.00)))
#	print myval
	return myval
# integration over the momenta
def Trapezoidal(f, a, b, n, alpha1, alpha2, p_star, a1):
    h = (b-a)/float(n)
    s = 0.5*(f(a,alpha1,alpha2,p_star,a1) + f(b,alpha1,alpha2,p_star,a1))
    for i in range(1,n,1):
        s = s + f(a + i*h,alpha1,alpha2,p_star,a1)
    return h*s
###############
# define a function for the calculation of the spectrum
# npts = number of points along frequency
#
def mylumint(npts,myconst2,b_mugj,cj,alpha3,volj,myd,myf1,bnorm,b,p_star,alpha1,alpha2):
	pi=math.acos(-1.00)  # pi
	c3=((math.sqrt(3.00)*(4.80**3.0))/(4.00*pi*9.1090*(2.9970**20)))*(1.0e-22)  # const c3 in eqn 19
	c4=((2.00**(2.00/3.00))*((pi/3.00)**(3.00/2.00)))/0.94050  # const in eqn 20 with Gamma function(11/6)
	c51=(4.00*pi*9.1090*2.9970)/(3.00*4.800)  # const in nu_i(p) in eqn 19
	c5=c51**(1.00/3.00)
	c6=c5*10.00**(-8.00/3.00)
	mpc3=(3.0850**3.00)*((1.0e24)**3.00) # megapc cube for volume
	const1=c3*c4*c6*mpc3*0.100  #const in front of integral in eqn 19
	l_nu=[]
	f_nu=[]
	i=0
	for i in range(0,npts):
		nu_ghz=i*0.01
		k=((nu_ghz/b_mugj)**(1.00/3.00))*b_mugj*(cj**alpha3)*volj
		a1=(myconst2)*((nu_ghz/b_mugj)**(7.00/8.00))
		s=0.0
		n=1000
#		print 'function input=', a,b,n,alpha1,alpha2,p_star,a1
		a=10.0# integration limit
		s= Trapezoidal(func,a,b,n,alpha1,alpha2,p_star,a1)
		l_nu.append(const1*s*k)
		f_nu.append((l_nu[i]/(4*pi*myd**2.00))*bnorm)
		mydat=str(nu_ghz)+' '+str(l_nu[i])+' '+str(f_nu[i])+'\n'
		myf1.write(mydat)
	return l_nu, f_nu, k, a1

# calculate the spectrum only where samples exist in the observed spectrum
def mylumintobs(myconst2,b_mugj,cj,alpha3,volj,myd,myspec,b,p_star,alpha1,alpha2):
	pi=math.acos(-1.00)  # pi
	c3=((math.sqrt(3.00)*(4.80**3.0))/(4.00*pi*9.1090*(2.9970**20)))*(1.0e-22)  # const c3 in eqn 19
	c4=((2.00**(2.00/3.00))*((pi/3.00)**(3.00/2.00)))/0.94050  # const in eqn 20 with Gamma function(11/6)
	c51=(4.00*pi*9.1090*2.9970)/(3.00*4.800)  # const in nu_i(p) in eqn 19
	c5=c51**(1.00/3.00)
	c6=c5*10.00**(-8.00/3.00)
	mpc3=(3.0850**3.00)*((1.0e24)**3.00) # megapc cube for volume
	const1=c3*c4*c6*mpc3*0.100  #const in front of integral in eqn 19
#	fact=10**-25 # to get normalisation within some factors
#	const2=(11.00/8.00)*((c51*1e7)**(7.00/8.00)) # const inside the exp in eqn 20
	l_nu=[]
	f_nu=[]
	i=0
	for i in range(0,len(myspec)):
		nu_ghz=myspec[i]   # match this to the observed frequencies
		k=((nu_ghz/b_mugj)**(1.00/3.00))*b_mugj*(cj**alpha3)*volj
		a1=(myconst2)*((nu_ghz/b_mugj)**(7.00/8.00))
		s=0.0
		n=1000
		a=10.0# integration lower limit
		print i, 'function input=', a,b,n,alpha1,alpha2,p_star,a1
		s= Trapezoidal(func,a,b,n,alpha1,alpha2,p_star,a1)
		print i, s
		l_nu.append(const1*s*k)
		f_nu.append((l_nu[i]/(4*pi*myd**2.00)))
		mydat=str(nu_ghz)+' '+str(l_nu[i])+' '+str(f_nu[i])+'\n'
#		print mydat
	return f_nu


def getnorm1(fobs, fobserr, fexp):
	myb1=[]
	myb2=[]
	myb3=[]
	for i in range(0,len(fobs)):
		diff1 = fobs[i]*(1.0/(fobserr[i]**2))
		diff3 = (1.0/fobserr[i])**2.0
		diff2 = fexp[i]#/(fobserr[i]**2)
		myb1.append(diff1)
		myb3.append(diff3)
		myb2.append(diff2)
	mynorm = (sum(myb1)/sum(diff3))/sum(myb2)
	return mynorm


def getnorm(fobs, fobserr, fexp):
	myb1=[]
	myb2=[]
	for i in range(0,len(fobs)):
		diff1 = fobs[i]
		diff2 = fexp[i]
		myb1.append(diff1)
		myb2.append(diff2)
	mynorm = sum(myb1)/sum(myb2)
	return mynorm

def myredchi(fobs,fexp,bnorm,fobserr):
	mychisq=[]
	for i in range(0,len(fobs)):
		mymod = abs(bnorm)*fexp[i]
		diffsq= ((fobs[i] - mymod)**2.0)/(fobserr[i]**2.0)
		mychisq.append(diffsq)
	mytot =sum(mychisq)/8.0 # number of points used is 9 and 1 parameter is fitted
	return mytot



def mychi(fobs,fexp,bnorm):
	mychisq=[]
	for i in range(0,len(fobs)):
		mymod = bnorm*fexp[i]
		diffsq= ((fobs[i] - mymod)**2.0)/mymod
		mychisq.append(diffsq)
	mytot =sum(mychisq)
	return mytot
#=====================================
# Here the main program begins

def myfull(del_t, tau, bofsrc, volofsrc, myph):
################################################
# block of constants used.
	a0=1.643e-3  # a0 in eqn 1
	pi=math.acos(-1.00)  # pi
	c3=((math.sqrt(3.00)*(4.80**3.0))/(4.00*pi*9.1090*(2.9970**20)))*(1.0e-22)  # const c3 in eqn 19
	c4=((2.00**(2.00/3.00))*((pi/3.00)**(3.00/2.00)))/0.94050  # const in eqn 20 with Gamma function(11/6)
	c51=(4.00*pi*9.1090*2.9970)/(3.00*4.800)  # const in nu_i(p) in eqn 19
	c5=c51**(1.00/3.00)
	c6=c5*10.00**(-8.00/3.00)
	mpc3=(3.0850**3.00)*((1.0e24)**3.00) # megapc cube for volume
	const1=c3*c4*c6*mpc3*0.100  #const in front of integral in eqn 19
	const2=(11.00/8.00)*((c51*1e7)**(7.00/8.00)) # const inside the exp in eqn 20
###################################################
# particle distribution spectral index
	alph_e=2.50
# useful numbers in further calculations      
	alpha1= -alph_e - (2.00/3.00)
	alpha2=alph_e-2.00
	alpha3=(alph_e+2.00)/3.00
# source properties
	z=0.17120 # redshift for A1914
	cosmo = FlatLambdaCDM(H0=69.6, Om0=0.286)   
	def getdl(redshift):
		dl = cosmo.luminosity_distance(redshift)
		return dl.value
	d = getdl(z)
#print "Luminosity distance to the source = ", d
#quit()
# calculate the redshift dependent quantity
	b_cmb=3.250*(1.00+z)**2.00  # equivalent field for CMB
	u_c=((b_cmb**2.00)/(8.00*pi))*(0.62460) # energy density for CMB
#	bofsrc = 5.0 # micro Gauss
#	volofsrc = 0.00370 # Mpc^3
##############################################
	b1 =[1.80,1.20,0.0,2.0,0.0]
#	print myph, b1, del_t, tau, bofsrc, volofsrc
	b_mug, u_b, vol, c1 = ph4flashing(myph, b1, del_t, tau, bofsrc, volofsrc)
#	print 'c1=', c1
#	print 'vol=', vol
	l_nu=[]
	f_nu=[]
	t=[0,0,0,0,0]
	p_star1=[0,0,0,0,0]
	c=[0,0,0,0,0]
	fnu_exp=[]	
	ptb = open('pars-phases.dat','w')
	mycols='Phase-i   delta-t   tau   Ci-1i    b    Coi   Vimpc3   Bimug   ubi    pstar0i \n'
	ptb.write(mycols)
	for j in range(0,5):
		if j==0:
			t[j]=del_t[j]+tau[j]
#			t[j]=del_t[j]+abs(tau[j]) ## TRIAL to avoid NAN
#			c1[j]=(1.0+del_t[j]*1.00/tau[j])**(-b1[j])
			p_star=1e5
			p_star1[0]=p_star
# upperlimit for integration b for case j=1       
			b=1e5
#      b=p_star
			c[0]=1.0
# now the actual loop for nu begins: this is for phase 0
#			mylumint(1000,const2,b_mug[j],c[j],alpha3,vol[j],d,f1)
#			print const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star
			if myph == j:
				fnu_exp = mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star,alpha1,alpha2)
			mycols=str("%d" % j)+' '+str("%.4f" % del_t[j])+' '+str("%.3f" % tau[j])+' '+str("%.4f" % b1[j])+' '+\
				str("%.4f" % c1[j])+' '+str("%.4f" % vol[j])+' '+str("%.4f" % b_mug[j])+' '+str("%.4f" % u_b[j])+\
				' '+str("%.4f" % p_star1[j])+'\n'
			ptb.write(mycols)
		else:
			init=[0,0,0,0,0]
			r_t=[0,0,0,0,0]
			s_t=[0,0,0,0,0]
			brac_B=[0,0,0,0,0]
			brac_C=init
#			fact=init
			t[j]=del_t[j]+tau[j]
#			t[j]=del_t[j]+abs(tau[j])## TRIAL to avoid NAN
#			c1[j]=(1.0+del_t[j]*1.00/tau[j])**(-b1[j])
			vol[j]=vol[j-1]/c1[j]
			c[j]=vol[0]/vol[j]      
			u_b[j]=u_b[0]*(vol[j]/vol[0])**(-4.00/3.00)
#c         write(*,*)'c',c(j),'vol',vol(j),'u_b',u_b(1)         
			b_mug[j]=sqrt((u_b[j]*8.00*pi)/0.62460)         
#c          x2= (c_ub(j)*u_b(j))+(c_uc(j)*u_c)
#c the change to u_b(j-1) has given correct value of p_star(2)
#c but not of p_star(3,4,5) !!!
			r_t[j]=del_t[j]/tau[j]
			s_t[j]=1.0+r_t[j]
			ind1=1.00-((5.00*b1[j])/3.00)
			numera = ((s_t[j]**ind1)-1.00)
			denom = ((ind1*s_t[j])*u_b[j-1])
			brac_B[j]=((s_t[j]**ind1)-1.00)/((ind1*s_t[j])*u_b[j-1])
#			print j, ind1, s_t[j], numera, brac_B[j]
			ind2=1.00-(b1[j]/3.00)
			brac_C[j]=(((s_t[j]**ind2)-1.00)/(ind2*s_t[j]))*u_c
			p_star1[j]=(s_t[j]**(-b1[j]/3.00))/(a0*t[j]*(brac_B[j]+brac_C[j]))
			x1=0.00
#			print 'p_star1=',p_star1
#			print 'c=', c
			myx1=[]
			for l in range(1,j+1):
				x1=x1+((c[l-1]**(1.00/3.00))/p_star1[l])
				myx1.append(((c[l-1]**(1.00/3.00))/p_star1[l]))
#		print 'myx1=',myx1
			x1=sum(myx1)
#		print x1
#		do l=2,j
#			x1=x1+((c(l-1)**(1.00/3.00))/p_star1(l))
#		enddo
			p_star=1.00/x1
			print j, p_star, x1
# 			print 'j', j, 'p_star', p_star, 'x1', x1
####         write(*,*) 'j', j, 'p_star', p_star, 'x1', x1
			b=p_star
#######################3
			if j==1:
#				mylumint(1000,const2,b_mug[j],c[j],alpha3,vol[j],d,f2)
				if myph == j:
					fnu_exp = mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star,alpha1,alpha2)
#				mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star)
#				print const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star
				mycols=str("%d" % j)+' '+str("%.4f" % del_t[j])+' '+str("%.3f" % tau[j])+' '+str("%.4f" % b1[j])+' '+\
				str("%.4f" % c1[j])+' '+str("%.4f" % vol[j])+' '+str("%.4f" % b_mug[j])+' '+str("%.4f" % u_b[j])+\
				' '+str("%.4f" % p_star1[j])+'\n'
				ptb.write(mycols)
#				myf2=str(nu_ghz)+' '+str(l_nu[i])+' '+str(f_nu[i])
#				f2.write(myf2)
#				print myf2
			elif j==2:
#				myf3=str(nu_ghz)+' '+str(l_nu[i])+' '+str(f_nu[i])
#				f3.write(myf3)
#				mylumint(1000,const2,b_mug[j],c[j],alpha3,vol[j],d,f3)
				if myph == j:
					fnu_exp = mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star,alpha1,alpha2)
#				mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star)
#				print const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star
				mycols=str("%d" % j)+' '+str("%.4f" % del_t[j])+' '+str("%.3f" % tau[j])+' '+str("%.4f" % b1[j])+' '+\
				str("%.4f" % c1[j])+' '+str("%.4f" % vol[j])+' '+str("%.4f" % b_mug[j])+' '+str("%.4f" % u_b[j])+\
				' '+str("%.4f" % p_star1[j])+'\n'
				ptb.write(mycols)
#				print myf3
			elif j==3:
#				mylumint(1000,const2,b_mug[j],c[j],alpha3,vol[j],d,f4)
				if myph == j:
					fnu_exp = mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star,alpha1,alpha2)
#				print const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star
				mycols=str("%d" % j)+' '+str("%.4f" % del_t[j])+' '+str("%.3f" % tau[j])+' '+str("%.4f" % b1[j])+' '+\
				str("%.4f" % c1[j])+' '+str("%.4f" % vol[j])+' '+str("%.4f" % b_mug[j])+' '+str("%.4f" % u_b[j])+\
				' '+str("%.4f" % p_star1[j])+'\n'
				ptb.write(mycols)
#				myf4=str(nu_ghz)+' '+str(l_nu[i])+' '+str(f_nu[i])
#				f4.write(myf4)
#				print myf4
			elif j==4:
#				mylumint(1000,const2,b_mug[j],c[j],alpha3,vol[j],d,f5)
				if myph == j:
					fnu_exp = mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star,alpha1,alpha2)
#				mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star)
#				print const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star
				mycols=str("%d" % j)+' '+str("%.4f" % del_t[j])+' '+str("%.3f" % tau[j])+' '+str("%.4f" % b1[j])+' '+\
				str("%.4f" % c1[j])+' '+str("%.4f" % vol[j])+' '+str("%.4f" % b_mug[j])+' '+str("%.4f" % u_b[j])+\
				' '+str("%.4f" % p_star1[j])+'\n'
				ptb.write(mycols)
#			myf5=str(nu_ghz)+' '+str(l_nu[i])+' '+str(f_nu[i])
#				f5.write(myf5)
#				print myf5
	ptb.close()
# Now obtain the normalisation
#print flx_obs
#print f_err
#print fnu_exp
#f_nuobs=np.array(spec['nughz'])
#f_err=np.array(spec['fmyerr'])
#flx_obs = np.array(spec['fmy'])
	bestnorm = getnorm(flx_obs, f_err, fnu_exp)
#	if bestnorm =='nan':
#		raise SystemExit
#Now go for minimising chisquare
	bestchi = mychi(flx_obs, fnu_exp, bestnorm)
#	bestchi = myredchi(flx_obs, fnu_exp, bestnorm,f_err)  # reduced chisq
#	print del_t, tau
	print bestnorm, bestchi
	return bestchi, tau[0], del_t[1], del_t[2], del_t[3], tau[3], del_t[4], fnu_exp, bestnorm	

#########################
def myfull_best(del_t, tau, bofsrc, volofsrc, myph, mynorm): # to calculate only at all the frequencies
################################################
# block of constants used.
	a0=1.643e-3  # a0 in eqn 1
	pi=math.acos(-1.00)  # pi
	c3=((math.sqrt(3.00)*(4.80**3.0))/(4.00*pi*9.1090*(2.9970**20)))*(1.0e-22)  # const c3 in eqn 19
	c4=((2.00**(2.00/3.00))*((pi/3.00)**(3.00/2.00)))/0.94050  # const in eqn 20 with Gamma function(11/6)
	c51=(4.00*pi*9.1090*2.9970)/(3.00*4.800)  # const in nu_i(p) in eqn 19
	c5=c51**(1.00/3.00)
	c6=c5*10.00**(-8.00/3.00)
	mpc3=(3.0850**3.00)*((1.0e24)**3.00) # megapc cube for volume
	const1=c3*c4*c6*mpc3*0.100  #const in front of integral in eqn 19
	const2=(11.00/8.00)*((c51*1e7)**(7.00/8.00)) # const inside the exp in eqn 20
###################################################
# particle distribution spectral index
	alph_e=2.50
# useful numbers in further calculations      
	alpha1= -alph_e - (2.00/3.00)
	alpha2=alph_e-2.00
	alpha3=(alph_e+2.00)/3.00
# source properties
	z=0.17120 # redshift for A1914
	cosmo = FlatLambdaCDM(H0=69.6, Om0=0.286)   
	def getdl(redshift):
		dl = cosmo.luminosity_distance(redshift)
		return dl.value
	d = getdl(z)
#print "Luminosity distance to the source = ", d
#quit()
# calculate the redshift dependent quantity
	b_cmb=3.250*(1.00+z)**2.00  # equivalent field for CMB
	u_c=((b_cmb**2.00)/(8.00*pi))*(0.62460) # energy density for CMB
#	bofsrc = 5.0 # micro Gauss
#	volofsrc = 0.00370 # Mpc^3
##############################################
	b1 =[1.80,1.20,0.0,2.0,0.0]
#	print myph, b1, del_t, tau, bofsrc, volofsrc
	b_mug, u_b, vol, c1 = ph4flashing(myph, b1, del_t, tau, bofsrc, volofsrc)
#	print 'c1=', c1
#	print 'vol=', vol
	l_nu=[]
	f_nu=[]
	t=[0,0,0,0,0]
	p_star1=[0,0,0,0,0]
	c=[0,0,0,0,0]
	fnu_exp=[]	
	ptb = open('pars-phases.dat','w')
	mycols='Phase-i   delta-t   tau   Ci-1i    b    Coi   Vimpc3   Bimug   ubi    pstar0i \n'
	ptb.write(mycols)
	f1 = open('phase1.dat'+str(matchphase),'w')
	f2 = open('phase2.dat'+str(matchphase),'w')
	f3 = open('phase3.dat'+str(matchphase),'w')
	f4 = open('phase4.dat'+str(matchphase),'w')
	f5 = open('phase5.dat'+str(matchphase),'w')
	for j in range(0,5):
		if j==0:
			t[j]=del_t[j]+tau[j]
#			t[j]=del_t[j]+abs(tau[j]) ## TRIAL to avoid NAN
#			c1[j]=(1.0+del_t[j]*1.00/tau[j])**(-b1[j])
			p_star=1e5
			p_star1[0]=p_star
# upperlimit for integration b for case j=1       
			b=p_star # 
#      b=p_star
			c[0]=1.0
# now the actual loop for nu begins: this is for phase 0
			mylumint(10000,const2,b_mug[j],c[j],alpha3,vol[j],d,f1,mynorm,b,p_star,alpha1,alpha2)
#			print const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star
#			if myph == j:
#				fnu_exp = mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star,alpha1,alpha2)
			mycols=str("%d" % j)+' '+str("%.4f" % del_t[j])+' '+str("%.3f" % tau[j])+' '+str("%.4f" % b1[j])+' '+\
				str("%.4f" % c1[j])+' '+str("%.4f" % vol[j])+' '+str("%.4f" % b_mug[j])+' '+str("%.4f" % u_b[j])+\
				' '+str("%.4f" % p_star1[j])+'\n'
			ptb.write(mycols)
		else:
			init=[0,0,0,0,0]
			r_t=[0,0,0,0,0]
			s_t=[0,0,0,0,0]
			brac_B=[0,0,0,0,0]
			brac_C=init
#			fact=init
			t[j]=del_t[j]+tau[j]
#			t[j]=del_t[j]+abs(tau[j])## TRIAL to avoid NAN
#			c1[j]=(1.0+del_t[j]*1.00/tau[j])**(-b1[j])
			vol[j]=vol[j-1]/c1[j]
			c[j]=vol[0]/vol[j]      
			u_b[j]=u_b[0]*(vol[j]/vol[0])**(-4.00/3.00)
#c         write(*,*)'c',c(j),'vol',vol(j),'u_b',u_b(1)         
			b_mug[j]=sqrt((u_b[j]*8.00*pi)/0.62460)         
#c          x2= (c_ub(j)*u_b(j))+(c_uc(j)*u_c)
#c the change to u_b(j-1) has given correct value of p_star(2)
#c but not of p_star(3,4,5) !!!
			r_t[j]=del_t[j]/tau[j]
			s_t[j]=1.0+r_t[j]
			ind1=1.00-((5.00*b1[j])/3.00)
			numera = ((s_t[j]**ind1)-1.00)
			denom = ((ind1*s_t[j])*u_b[j-1])
			brac_B[j]=((s_t[j]**ind1)-1.00)/((ind1*s_t[j])*u_b[j-1])
#			print j, ind1, s_t[j], numera, brac_B[j]
			ind2=1.00-(b1[j]/3.00)
			brac_C[j]=(((s_t[j]**ind2)-1.00)/(ind2*s_t[j]))*u_c
			p_star1[j]=(s_t[j]**(-b1[j]/3.00))/(a0*t[j]*(brac_B[j]+brac_C[j]))
			x1=0.00
#			print 'p_star1=',p_star1
#			print 'c=', c
			myx1=[]
			for l in range(1,j+1):
				x1=x1+((c[l-1]**(1.00/3.00))/p_star1[l])
				myx1.append(((c[l-1]**(1.00/3.00))/p_star1[l]))
#		print 'myx1=',myx1
			x1=sum(myx1)
#		print x1
#		do l=2,j
#			x1=x1+((c(l-1)**(1.00/3.00))/p_star1(l))
#		enddo
			p_star=1.00/x1
# 			print 'j', j, 'p_star', p_star, 'x1', x1
####         write(*,*) 'j', j, 'p_star', p_star, 'x1', x1
#		#	if j==1:
#				b=1E5
#			else:
			b=p_star
#######################3
			if j==1:
				mylumint(10000,const2,b_mug[j],c[j],alpha3,vol[j],d,f2,mynorm,b,p_star,alpha1,alpha2)
#				if myph == j:
#					fnu_exp = mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star,alpha1,alpha2)
#				mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star)
#				print const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star
				mycols=str("%d" % j)+' '+str("%.4f" % del_t[j])+' '+str("%.3f" % tau[j])+' '+str("%.4f" % b1[j])+' '+\
				str("%.4f" % c1[j])+' '+str("%.4f" % vol[j])+' '+str("%.4f" % b_mug[j])+' '+str("%.4f" % u_b[j])+\
				' '+str("%.4f" % p_star1[j])+'\n'
				ptb.write(mycols)
#				myf2=str(nu_ghz)+' '+str(l_nu[i])+' '+str(f_nu[i])
#				f2.write(myf2)
#				print myf2
			elif j==2:
#				myf3=str(nu_ghz)+' '+str(l_nu[i])+' '+str(f_nu[i])
#				f3.write(myf3)
				mylumint(10000,const2,b_mug[j],c[j],alpha3,vol[j],d,f3,mynorm,b,p_star,alpha1,alpha2)
#				if myph == j:
#					fnu_exp = mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star,alpha1,alpha2)
#				mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star)
#				print const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star
				mycols=str("%d" % j)+' '+str("%.4f" % del_t[j])+' '+str("%.3f" % tau[j])+' '+str("%.4f" % b1[j])+' '+\
				str("%.4f" % c1[j])+' '+str("%.4f" % vol[j])+' '+str("%.4f" % b_mug[j])+' '+str("%.4f" % u_b[j])+\
				' '+str("%.4f" % p_star1[j])+'\n'
				ptb.write(mycols)
#				print myf3
			elif j==3:
				mylumint(10000,const2,b_mug[j],c[j],alpha3,vol[j],d,f4,mynorm,b,p_star,alpha1,alpha2)
#				if myph == j:
#					fnu_exp = mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star,alpha1,alpha2)
#				print const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star
				mycols=str("%d" % j)+' '+str("%.4f" % del_t[j])+' '+str("%.3f" % tau[j])+' '+str("%.4f" % b1[j])+' '+\
				str("%.4f" % c1[j])+' '+str("%.4f" % vol[j])+' '+str("%.4f" % b_mug[j])+' '+str("%.4f" % u_b[j])+\
				' '+str("%.4f" % p_star1[j])+'\n'
				ptb.write(mycols)
#				myf4=str(nu_ghz)+' '+str(l_nu[i])+' '+str(f_nu[i])
#				f4.write(myf4)
#				print myf4
			elif j==4:
				mylumint(10000,const2,b_mug[j],c[j],alpha3,vol[j],d,f5,mynorm,b,p_star,alpha1,alpha2)
#				if myph == j:
#					fnu_exp = mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star,alpha1,alpha2)
#				mylumintobs(const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star)
#				print const2,b_mug[j],c[j],alpha3,vol[j],d,f_nuobs,b,p_star
				mycols=str("%d" % j)+' '+str("%.4f" % del_t[j])+' '+str("%.3f" % tau[j])+' '+str("%.4f" % b1[j])+' '+\
				str("%.4f" % c1[j])+' '+str("%.4f" % vol[j])+' '+str("%.4f" % b_mug[j])+' '+str("%.4f" % u_b[j])+\
				' '+str("%.4f" % p_star1[j])+'\n'
				ptb.write(mycols)
#			myf5=str(nu_ghz)+' '+str(l_nu[i])+' '+str(f_nu[i])
#				f5.write(myf5)
#				print myf5
	ptb.close()
	f1.close()
	f2.close()
	f3.close()
	f4.close()
	f5.close()
	print b1, vol, b_mug
	return b1, vol, b_mug
# Now obtain the normalisation
#print flx_obs
#print f_err
#print fnu_exp
#f_nuobs=np.array(spec['nughz'])
#f_err=np.array(spec['fmyerr'])
#flx_obs = np.array(spec['fmy'])
#	bestnorm = getnorm(flx_obs, f_err, fnu_exp)
#	if bestnorm =='nan':
#		raise SystemExit
#Now go for minimising chisquare
#	bestchi = mychi(flx_obs, fnu_exp, bestnorm)
#	bestchi = myredchi(flx_obs, fnu_exp, bestnorm,f_err)  # reduced chisq
#	print del_t, tau
#	print bestnorm, bestchi
#s	return tau[0], del_t[1], del_t[2], del_t[3], tau[3], del_t[4]	



######################
# Run the program
#####################
matchphase =2

#myscen = [[True, False, False], [False, True, False],[False, False, True]

# For scen A
#tau_0, deltex_1, deltex_2, deltex_3, tau_3, deltex_4 = getparspace(True,False,False,matchphase)
# For scen B
#tau_0, deltex_1, deltex_2, deltex_3, tau_3, deltex_4 = getparspace(False,True,False,matchphase)
# For scen C
#tau_0, deltex_1, deltex_2, deltex_3, tau_3, deltex_4 = getparspace(False,False,True,matchphase)

#myscenario = 'A'
myscenario = 'B'
#myscenario = 'C'

tau_0, deltex_1, deltex_2, deltex_3, tau_3, deltex_4 = getparspace(myscenario,matchphase)

# for finer parameter search
#tau_0, deltex_1, deltex_2, deltex_3, tau_3, deltex_4 = getfineparspace(myscenario,matchphase)





# write this out to a file
import time

time.sleep(3.0)





#quit()

pex=open('parex.dat','w')
mypex = 'tau0  deltex1 deltex2 deltex3 tau3 deltex4\n'
pex.write(mypex)

for i in range(0,len(tau_0)):
	for jj in range(0,len(deltex_1)):
		for kk in range(0,len(deltex_2)):
			for ll in range(0,len(deltex_3)):
				for mm in range(0,len(tau_3)):
					for nn in range(0,len(deltex_4)):
						mypex = str(tau_0[i])+' '+str(deltex_1[jj])+' '+str(deltex_2[kk])+' '+str(deltex_3[ll])+' '+str(tau_3[mm])+' '+str(deltex_4[nn])+'\n'
#						print mypex
						pex.write(mypex)

pex.close()


# Now read in the parameters from this file and loop over them
timeex = ascii.read('parex.dat')



#print timeex






# first write the exploration space to a file and then run the actual program.

# Now all the exploration variables are in timeex

mychi1=[]
mychi2=[]
mtau0=[]
mdelt1=[]
mdelt2=[]
mdelt3=[]
mtau3=[]
mdelt4=[]
mybestnorm=[]

result=open('myresults.dat', 'w')
#mycols='index, tau0, delt1, delt2, delt3, tau3, delt4, chi2 \n'
#result.write(mycols)
for i in range(0,len(timeex)):
#for i in range(0,min(len(timeex),50)):
#for i in range(0,500):
	del_t=[0.0,0.0,0.0,0.0,0.0]
	tau=[0.0,0.0,9.0,0.0,9.0]
	tau[0] = timeex['tau0'][i]
	tau[1] = (2.0/3.0)*tau[0]
	del_t[1] = timeex['deltex1'][i]
	del_t[2] = timeex['deltex2'][i]
	del_t[3] = timeex['deltex3'][i]
	tau[3] = timeex['tau3'][i]
	del_t[4] = timeex['deltex4'][i]
	print i, del_t[3], tau[3]
#	print i
	mychi1 = myfull(del_t, tau, bofsrc, volofsrc, matchphase)
#	print mychi1[0]
	mychi2.append(mychi1[0])
	mtau0.append(mychi1[1])
	mdelt1.append(mychi1[2])
	mdelt2.append(mychi1[3])
	mdelt3.append(mychi1[4])
	mtau3.append(mychi1[5])
	mdelt4.append(mychi1[6])
	mybestnorm.append(mychi1[8])
# bestchi, tau[0], del_t[1], del_t[2], del_t[3], tau[3], del_t[4], fnu_exp, bestnorm							
	myresult=str(i)+' '+str(mychi1[0])+'\n'
#	myresult= str(myind)+' '+str(mtau0[myind])+' '+\
#		str(mdelt1[myind])+' '+str(mdelt2[myind])+\
#		' '+str(mdelt3[myind])+' '+str(mtau3[myind])+\
#		' '+str(mdelt4[myind])+' '+str(mychi2[myind])+'\n'
	result.write(myresult)
#	print myind, mdelt1[myind],mdelt2[myind],mdelt3[myind],mychi2[myind]

#print "my best norm=", mybestnorm

myind = np.nanargmin(mychi2)  # ignored nan and gives the minimum
mynorm = mybestnorm[myind]
#myind1 = np.nanmin(np.array(mychi2),axis=0)
#print 'mind1', myind1
#myind=int(myind1)
del_t=[0.0,0.0,0.0,0.0,0.0]
tau=[0.0,0.0,9.0,0.0,9.0]
tau[0] = timeex['tau0'][myind]
tau[1] = (2.0/3.0)*tau[0]
tau[3] = timeex['tau3'][myind]
del_t[1] = timeex['deltex1'][myind]
del_t[2] = timeex['deltex2'][myind]
del_t[3] = timeex['deltex3'][myind]
del_t[4] = timeex['deltex4'][myind]

print "For minimum chisq"
print 'Myindex= ', myind
print "tau =", tau
print "delt =", del_t

bestchisq, tau0, dt1, dt2, dt3, tau3, dt4, fnu_exp, bestnorm = myfull(del_t, tau, bofsrc, volofsrc, matchphase)
#myfull_best(del_t, tau, bofsrc, volofsrc, matchphase,mybestnorm[0])


#quit()
# Plot with the source spectrum
fig,ax = plt.subplots()
ax.set_yscale('log')
ax.set_xscale('log')
ax.plot(f_nuobs,flx_obs,'bo',label='obs')
ax.errorbar(f_nuobs,flx_obs,yerr=f_err)
for i in range(0,len(fnu_exp)):
#		print f_nuobs[i],bestnorm*fnu_exp[i]
	ax.plot(f_nuobs[i],bestnorm*fnu_exp[i],'ro')
plt.show()
plt.close()

#parchk =True

#if parchk==True:
#	bestchisq, tau0, dt1, dt2, dt3, tau3, dt4, fnu_exp, bestnorm = myfull_best(del_t, tau, bofsrc, volofsrc, matchphase,mybestnorm[0])


myfull_best(del_t, tau, bofsrc, volofsrc, matchphase,mynorm)


quit()



# to get the max min for finer parameter search:

tau[0] = timeex['tau0'][myind-1]
tau[1] = (2.0/3.0)*tau[0]
tau[3] = timeex['tau3'][myind-1]
del_t[1] = timeex['deltex1'][myind-1]
del_t[2] = timeex['deltex2'][myind-1]
del_t[3] = timeex['deltex3'][myind-1]
del_t[4] = timeex['deltex4'][myind-1]

print "For lower limit"
print "tau =", tau
print "delt =", del_t

tau[0] = timeex['tau0'][myind+1]
tau[1] = (2.0/3.0)*tau[0]
tau[3] = timeex['tau3'][myind+1]
del_t[1] = timeex['deltex1'][myind+1]
del_t[2] = timeex['deltex2'][myind+1]
del_t[3] = timeex['deltex3'][myind+1]
del_t[4] = timeex['deltex4'][myind+1]

print "For upper limit"
print "tau =", tau
print "delt =", del_t




quit()



