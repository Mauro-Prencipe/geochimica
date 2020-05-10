# import os
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from scipy import integrate
from scipy.optimize import minimize

R=8.3145
    
class name_data:
    def __init__(self):
        self.mineral_names=[]
    def add(self,nlist):
        self.mineral_names.extend(nlist)

class ideal:
    def __init__(self):
        self.flag=False
        self.w=0.
    def set_w(self,cw):
        self.w=cw
        self.flag=True
    def reset(self):
        self.flag=False
    
class mineral:
    def __init__(self,name,nick):
        self.name=name
        self.nick=nick
        self.eos='m'
        self.cp=None
        self.al=None
        self.k0=None
        self.kp=None
        self.dkt=None
        self.v0=None
        self.g0=None
        self.s0=None
        
    def info(self):
        print("Mineral: %s\n" % self.name)
        print("K0: %4.2f GPa, Kp: %4.2f, dK0/dT: %4.4f GPa/K, V0: %6.4f J/bar" \
              % (self.k0, self.kp, self.dkt, self.v0))
        print("G0: %8.2f J/mol, S0: %6.2f J/mol K\n" % (self.g0, self.s0))
        print("Cp coefficients and powers:")
        for ci in self.cp:
            print('{:>+8.4e}{:>+6.1f}'.format(ci[0],ci[1]))
        
        print("\nAlpha coefficients and powers:")
        for ai in self.al:
            print('{:>+8.4e}{:>+6.1f}'.format(ai[0],ai[1]))
            
    def load_ref(self,v0,g0,s0):
        self.v0=v0
        self.g0=g0
        self.s0=s0
         
    def load_bulk(self,k0, kp, dkt):
        self.k0=k0
        self.kp=kp
        self.dkt=dkt
        
    def load_cp(self,cpc,cpp):
        cl=list(zip(cpc,cpp))
        item=0
        self.cp=np.array([])
        for ic in cl:
            self.cp=np.append(self.cp,[cl[item][0], cl[item][1]])
            item=item+1
        self.cp=self.cp.reshape(item,2)
        
    def load_alpha(self, alc, alp):
        cl=list(zip(alc,alp))
        item=0
        self.al=np.array([])
        for ic in cl:
            self.al=np.append(self.al,[cl[item][0], cl[item][1]])
            item=item+1
        self.al=self.al.reshape(item,2) 
    
    def cp_t(self,tt):
        cpt=0.
        iterm=0
        for it in self.cp:
            cpt=cpt+self.cp[iterm,0]*(tt**self.cp[iterm,1])
            iterm=iterm+1
        return cpt 
    
    def alpha_t(self,tt):
        alt=0.
        iterm=0
        for it in self.al:
            alt=alt+self.al[iterm,0]*(tt**self.al[iterm,1])
            iterm=iterm+1
        return alt
    
    def kt(self,tt):
        return self.k0+(tt-298.15)*self.dkt
    
    def entropy(self,tt):
        fc=lambda ti: (self.cp_t(ti))/ti
        integ, err=scipy.integrate.quad(fc,298.15,tt)
        return integ+self.s0

    def volume_t(self,tt):
        fc= lambda ti: self.alpha_t(ti)
        integ,err=scipy.integrate.quad(fc,298.15,tt)
#        return (self.v0)*(1.+integ)
        return (self.v0)*np.exp(integ)
    
    def volume_p(self,tt,pp):
        k0t=self.kt(tt)
        vt=self.volume_t(tt)
        if self.eos=='m':
           fact=(1.-(pp*self.kp)/(pp*self.kp+k0t))**(1./self.kp)
           return fact*vt
        elif self.eos=='bm':
           pf=lambda f: (3*k0t*f*(1+2*f)**(5/2)*(1+3*f*(self.kp-4)/3)-pp)**2
           ff=scipy.optimize.minimize(pf,1,tol=0.00001)            
           return vt/((2*ff.x[0]+1)**(3/2))
    
    def volume_fix_t_p(self,tt):
        return lambda pp: self.volume_p(tt,pp) 
    
    def vdp(self,tt,pp):
        fv=self.volume_fix_t_p(tt)
        integ,err=scipy.integrate.quad(fv,0.0001, pp)
        return integ*1e4
    
    def g_t(self,tt):
        integ,err=scipy.integrate.quad(self.entropy, 298.15, tt)
        return integ
    
    def g_tp(self,tt,pp):
        return self.g0+self.vdp(tt,pp)-self.g_t(tt) 
    
    def alpha_p(self, tt, pp):
        v=self.volume_p(tt,pp)
        t_list=np.linspace(tt-10, tt+10, 5)
        vt_list=np.array([])
        for ti in t_list:
            vi=self.volume_p(ti,pp)
            vt_list=np.append(vt_list,vi)
        fitpv=np.polyfit(t_list,vt_list,2)
        fitder1=np.polyder(fitpv,1)
        altp=np.polyval(fitder1,tt)
        return 1*altp/v   
    
    def s_tp(self,tt,pp):
        gtp=lambda tf: self.g_tp(tf,pp)
        t_list=np.linspace(tt-5, tt+5, 5)
        g_list=np.array([])
        for ti in t_list:
            gi=self.g_tp(ti,pp)
            g_list=np.append(g_list,gi)
        fit=np.polyfit(t_list,g_list,2)
        fitder=np.polyder(fit,1)
        return -1*np.polyval(fitder,tt)
    
    def h_tp(self,tt,pp):
        g=self.g_tp(tt,pp)
        s=self.s_tp(tt,pp)
        return g+tt*s

name_list=name_data()

ens=mineral("enstatite", "ens")
fs=mineral("ferrosilite","fs")
di=mineral("diopside","di")
hed=mineral("hedembergite","hed")

def import_database():
    
    name_l=[]
    
    al_power_dic={
            'b1':  0,
            'b2':  1,
            'b3': -1,
            'b4': -2,
            'b5': -0.5,         
              }
    
    cp_power_dic={
            'c1':  0,
            'c2':  1,
            'c3': -2, 
            'c4':  2,
            'c5': -0.5,
            'c6': -1,
            'c7': -3,
            'c8': 3
                 }  
      
    list_cpc=[]
    list_cp=[]
    for ki,vi in cp_power_dic.items():
        list_cpc.append(ki)
        list_cp.append(vi)
        
    list_cal=[]
    list_al=[]
    for ki,vi in al_power_dic.items():
        list_cal.append(ki)
        list_al.append(vi)    
        
    line=''
    with open('pyrox.dat') as f:
         jc=0
         l0=['']
         while True:
            line=f.readline().rstrip()
            if line=='': continue
            if  line == 'END': break 
            jc=jc+1
            line_s=line.split()
            if line_s != []:
               l0=line_s[0].rstrip()
               if l0=='#': continue
               name=l0
               name_l.append(l0)
               l1=f.readline()
               l2=f.readline().rstrip()
               l2_s=l2.split()
               g0=float(l2_s[2])
               s0=float(l2_s[5])
               v0=float(l2_s[8])
               
               l3=f.readline().rstrip()
               l3_s=l3.split()
               l3n=len(l3_s)
               coeff_cp=l3_s[2:l3n:3]
               coeff_cp=[float(ci) for ci in coeff_cp]
               power_ccp=l3_s[0:l3n:3]
               power=[]
               for cci in power_ccp:
                   power.append(cp_power_dic.get(cci))
                                
               l4=f.readline().rstrip()
               l4_s=l4.split()
               l4n=len(l4_s)
               l4n_alpha=l4n-9
               coeff_alpha=l4_s[2:l4n_alpha:3]
               coeff_alpha=[float(ai) for ai in coeff_alpha]
               power_ac=l4_s[0:l4n_alpha:3]
               power_a=[]
               for ai in power_ac:
                   power_a.append(al_power_dic.get(ai))
                   
               k0=float(l4_s[-7])/1.e4
               dkt=float(l4_s[-4])/1.e4
               kp=float(l4_s[-1])
               
               eos='m'
               if kp < 0.:
                   eos='bm'
               
               eval(name+'.load_ref(v0,g0,s0)')
               eval(name+'.load_bulk(k0,kp,dkt)')
               eval(name+'.load_cp(coeff_cp,power)')
               eval(name+'.load_alpha(coeff_alpha,power_a)')
               name+'.eos='+eos
                  
               f.readline()
            line=f.readline().rstrip()
                              
         f.close()
    name_list.add(name_l)

import_database()

def AB(it,ip):
    
    mg_o=ens.g_tp(it,ip)
    mg_c=di.g_tp(it,ip)
    
    fe_o=fs.g_tp(it,ip)
    fe_c=hed.g_tp(it,ip)
    
    Ae=0.5*mg_c-mg_o
    Be=0.5*fe_c-fe_o
    
    A=np.exp(-1*Ae/(R*it))
    B=np.exp(-1*Be/(R*it))
    
    return A, B 
        

