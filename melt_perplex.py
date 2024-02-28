# Calcolo del diagramma di stato TX del sistema forsterite-fayalite
# alla pressione voluta.

# Modello ideale per la soluzione solida, oppure modello regolare,
# con parametro di Margules W

# Il calcolo sfrutta la minimizzazione dell'energia libera 
# del sistema globale solido + liquido 

# import os
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from scipy import integrate
from scipy.optimize import minimize
import warnings

database='perplex2_db.dat'

R=8.3145

class name_data:
    def __init__(self):
        self.mineral_names=[]
        self.database=''
    def add(self,nlist):
        self.mineral_names.extend(nlist)
    def data_file(self,file):
        self.database=file
    
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
name_list.data_file(database)

name_l=[]
name_ext=[]
list_flag=False
with open(database) as f:
    while True:
          line=f.readline().rstrip()
          if line=='': continue
          if line == 'end_list': break
          line_s=line.split()
          if line_s != []:
             l0=line_s[0].rstrip()
             if l0=='#': continue
             if l0=='begin_list':
                 list_flag=True
                 continue
             if list_flag: 
                name=l0
                l1=line_s[1].rstrip()
                name_l.append(l0)
                name_ext.append(l1)
f.close()   
print("\nDatabase %s" % name_list.database)
print("Number of imported phases: %4i" % len(name_l))
print("Phases: %s " % name_l)

for ip in list(range(len(name_l))):
    vars()[name_l[ip]]=mineral(name_ext[ip],name_l[ip])
            
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
    input_flag=False
    with open(database) as f:
         jc=0
         l0=['']
         while True:
            line=f.readline().rstrip()
            if line=='': continue
            if line=='start':
                input_flag=True
                continue
            
            if not input_flag: continue
        
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

# T fusione forsterite/fayalite

def deltaG(phase,phaseL,it,ip):
        gs=eval(phase+'.g_tp(it,ip)')
        gl=eval(phaseL+'.g_tp(it,ip)')
        dg=gl-gs
        return dg**2
   
def fusion(phase,phaseL,ip,t_ini=1800.,prt=False):
    """
    Determina la temperature di fusione di una fase 
    
    Input:
        phase, phaseL - sigle delle fasi solida e liquida
        ip            - pressione GPa
        t_ini         - temperatura iniziale da cui partire
                        per la ricerca (default: 1800 K)
        prt:          - se True, stampa la temperatura di fusione
                        (default: False)
    """

    tf=lambda it: deltaG(phase,phaseL,it,ip)
    t_fusion=scipy.optimize.minimize(tf,t_ini,tol=0.001)
    if prt:
        print("Temperatura di fusione: %5.2f K" % t_fusion.x[0])
    else:
        return t_fusion


def melt(ip=0,nt=10,tfmax=0.,W=8400., ideal=False,nt_prt=0):
    """
    Calcola il diagramma di stato TX del sistema fayalite-forsterite
    
    Input:
        ip    - pressione (GPa)
        nt    - numero di punti in temperatura
        tfmax - se non 0, fissa il massimo di temperatura per il grafico
        W     - Parametro di Margules per la soluzione solida
                (default: 8400 J/mole)
        ideal - calcola un diagramma di riferimento ideale (default: False)
        nt_ptr - se > 0 fissa il numero di valori di temperatura per
                 la stampa della tabella T(X) (default: 0; stampa tutti
                 gli nt valori calcolati)
    """
    
    W_val=W
    
    if W_val > 0.:
        print("\nModello simmetrico di soluzione per il solido:\nW*Xa*Xb; W= %5.1f J/mole\n" \
              % W_val)
    else:
        print("\nModello ideale di soluzione per il solido\n")
    
    tf_flag=False
    if tfmax > 0.:
        tf_flag=True
    
    tf_fo=fusion("fo","foL",ip)
    tf_fa=fusion("fa","faL",ip)
    
    t_fo=tf_fo.x[0]
    t_fa=tf_fa.x[0]
    
    print("Temperatura di fusione della forsterite: %5.2f K" % t_fo)
    print("Temperatura di fusione della fayalite: %5.2f K" % t_fa)
    
    t_list=np.linspace(t_fa,t_fo,nt)
    
    solid=np.array([])
    liquid=np.array([])
    solid_ideal=np.array([])
    liquid_ideal=np.array([])
            
    for it in t_list:
        
        comp,g_t=composition(it,ip,W_val,prt=False)
        comp_s=comp[0]
        comp_L=comp[1] 

        solid=np.append(solid,comp_s)
        liquid=np.append(liquid,comp_L)   
        
        if ideal:
            comp,g_t=composition(it,ip,0,prt=False)
            comp_s=comp[0]
            comp_L=comp[1] 

            solid_ideal=np.append(solid_ideal,comp_s)
            liquid_ideal=np.append(liquid_ideal,comp_L) 
         
    plt.figure()
    plt.plot(solid,t_list,"k-",label="solido")
    plt.plot(liquid,t_list,"b-",label="liquido")
    
    if ideal:
        plt.plot(solid_ideal,t_list, "k--",label="solido ideale")
        plt.plot(liquid_ideal,t_list, "b--", label="liquido ideale")
        
    plt.xlabel("X(Mg)")
    plt.ylabel("T (K)")
    if tf_flag:
        plt.ylim(1000.,tfmax)
    else:
        plt.ylim((t_fa-500,t_fo+100))
    plt.xlim((0,1.))
    title="Diagramma TX alla pressione di " + str(ip) + " GPa"
    plt.title(title)
    plt.legend(frameon=False,loc='lower right')
    plt.show()

    
    step=1
    if nt_prt > 0:
        step=int(nt/nt_prt)
        if step < 1:
            step=1
    
    xs_mg=solid[::step]
    xs_fe=1.-xs_mg
    xL_mg=liquid[::step]
    xL_fe=1-xL_mg
    t_list=t_list[::step]
    serie=(t_list.round(2),xs_mg.round(2),xs_fe.round(2),xL_mg.round(2),xL_fe.round(2))      
    pd.set_option('colheader_justify', 'center')
    df=pd.DataFrame(serie, index=['T (K)','X(Mg)sol','X(Fe)sol','X(Mg)liq','X(Fe)liq'])
    df=df.T
    print("")
    print(df.to_string(index=False))         
    
      
def g(x,it,gv,W):
    """  
    Calcola l'energia libera del sistema complessivo solido + liquido
    dalle rispettive composizioni e dalle quantità relative di solido
    e liquido.
    
    Input:
        x  - array contenente le composizioni e le quantità di solido e
             liquido
        it - temperatura
        gv - energie libere dei termini puri
    """
    
    gas=gv[0]
    gaL=gv[1]
    gbs=gv[2]
    gbL=gv[3]
    
    x1a=x[0]
    x1b=x[1]
    ra=x[2]
    rb=x[3]
    
    x2a=1-x1a
    x2b=1-x1b
         
    gab=x1a*gas+x2a*gbs+R*it*(x1a*np.log(x1a)+x2a*np.log(x2a)) + W*x1a*x2a
    gcd=x1b*gaL+x2b*gbL+R*it*(x1b*np.log(x1b)+x2b*np.log(x2b))
    
    return ra*gab+rb*gcd

def dg(x,it,gv,W):
    gas=gv[0]
    gaL=gv[1]
    gbs=gv[2]
    gbL=gv[3]
    
    x1a=x[0]
    x1b=x[1]
    ra=x[2]
    rb=x[3]
    
    x2a=1-x1a
    x2b=1-x1b
    
    gab=gas-gbs+R*it*(np.log(x1a)-np.log(x2a)) + W*x2a - W*x1a
    gcd=gaL-gbL+R*it*(np.log(x1a)-np.log(x2a))
    
    d1=ra*(gas-gbs+R*it*(np.log(x1a)-np.log(1-x1a))+W*(1-x1a)-W*x1a)
    d2=rb*(gaL-gbL+R*it*(np.log(x1b)-np.log(1-x1b)))
    d3=x1a*gas+(1-x1a)*gbs+R*it*(x1a*np.log(x1a)+(1-x1a)*np.log(1-x1a)) + W*x1a*(1-x1a)
    d4=x1b*gaL+(1-x1b)*gbL+R*it*(x1b*np.log(x1b)+(1-x1b)*np.log(1-x1b))
   
    return d1,d2,d3,d4
    
       
def refine(x,temp,gv,W):
    """
    Funzione utilizzata per determinare l'intersezione delle curve
    di energia libera delle fasi solida e liquida
    
    Input:
        x    - composizione iniziale
        temp - temperatura
        gv   - energie libere dei termini puri
    """

    gas=gv[0]
    gaL=gv[1]
    gbs=gv[2]
    gbL=gv[3]
        
    if x < 0.2:
        xmin=0.01
    else:
        xmin=x-0.2 
    if x > 0.80:
        xmax=0.99 
    else:
        xmax=x+0.20
        
    x_list=np.linspace(xmin,xmax,5)
    
    g2_list=np.array([])
    for ix in x_list:
        gs=ix*gas+(1-ix)*gbs+R*temp*(ix*np.log(ix)+(1-ix)*np.log(1-ix))+W*ix*(1-ix)
        gL=ix*gaL+(1-ix)*gbL+R*temp*(ix*np.log(ix)+(1-ix)*np.log(1-ix))
        ig2=(gs-gL)**2
        g2_list=np.append(g2_list,ig2)
        
    reg=np.polyfit(x_list,g2_list,2)
    der=np.polyder(reg,1)
    x_ref=-1*der[1]/der[0]

    return x_ref

def composition(it,ip,W_val,prt=True,xval=-1.):  
        
        """
        Determina la composizione delle fasi liquida e solida all'equilibrio
        e le quantità di liquido e solido data una composizione globale
    
        Input:
          it    -  temperatura (K)
          ip    -  pressione (GPa)
          W_val -  parametro di Margules 
          prt   -  stampa i risultati (default: True)
          xval  -  se diverso da -1 calcola le quantita' di solido e liquido
                   per la composizione globale xval; se xval=-1 il calcolo
                   viene fatto alla composizione per la quale le due curve
                   di energia libera si intersecano (default xval = -1) 
        """
    
        nx=30
        
        x_list=np.linspace(0.01,0.99,nx)
        gas=fo.g_tp(it,ip)
        gaL=foL.g_tp(it,ip)
        gbs=fa.g_tp(it,ip)
        gbL=faL.g_tp(it,ip)
        
        g_t=(gas,gaL,gbs,gbL)
        
        xe_list=np.array([])
        dg2l=np.array([])
        for ix in x_list:
            gs=ix*gas+(1-ix)*gbs+R*it*(ix*np.log(ix)+(1-ix)*np.log(1-ix))+W_val*ix*(1-ix)
            gL=ix*gaL+(1-ix)*gbL+R*it*(ix*np.log(ix)+(1-ix)*np.log(1-ix))
            dg2=(gs-gL)**2
            dg2l=np.append(dg2l,dg2)
        
        inx=np.argmin(dg2l)
        ix_int=x_list[inx]
        
        ix_int=refine(ix_int,it,g_t,W_val)
        
        if ix_int > 0.999:
           ix_int=0.999
        if ix_int < 0.001:
           ix_int=0.001
            
        x0=(ix_int, ix_int, 0.5, 0.5)
    
        con1={'type': 'eq', 
                           'fun': lambda x: x[2]+x[3]-1,
                           'jac': lambda x: np.array([0., 0., 1., 1.])}
        con2={'type': 'eq', 
                           'fun': lambda x: x[0]*x[2]+x[1]*x[3]-ix_int,
                           'jac': lambda x: np.array([x[2],x[3],x[0],x[1]])}
    
        eq_cons=[con1,con2]
        warnings.filterwarnings("ignore")
        bounds=((0.000001,0.999999),(0.000001,0.999999),(0.00001,0.99999),(0.00001,0.99999))
         
        res = minimize(g, x0, args=(it,g_t,W_val), bounds=bounds, method='slsqp',\
                     jac=dg, constraints=eq_cons, options={'ftol':1e-7, 'maxiter':200})
        
        if prt:
            xx = ix_int
            sq=res.x[2]
            lq=res.x[3]
            if xval > -1.:
                if xval < res.x[1] or xval > res.x[0]:
                    print("Warning: valore di X(Mg) fuori dal range\n")
                    return
                xx=xval
                dqsl=res.x[1]-res.x[0]
                lq=(xx-res.x[0])/dqsl
                sq=1.-lq
                
            print("Pressione %3.1f GPa,  Temperatura %4.1f K\n" % (ip,it))
            print("Per una composizione X(Mg) globale pari a %4.2f: " % xx)
            print("X(Mg) fase solida %4.2f, quantità fase solida %4.2f" \
                  % (res.x[0],sq))
            print("X(Mg) fase liquida %4.2f, quantità fase liquida %4.2f" \
                  % (res.x[1],lq))    
        
        if not prt:
            return res.x, g_t


def g_phase(tt,ip=0.,save=False,dp=600.,yl=False, W=0.):
    """
    Curve di energia libera della fase solida e della fase
    liquida.
    
    Input: 
        tt   - temperatura (K)
        ip   - pressione (GPa)  default: 0.
        save - salva una figura in un file (default: False)
        dp   - risoluzione della figura se salvata (default 600)
        yl   - limiti dell'asse G: se True, usa i limiti imposti
               nelle variabili ymn e ymx (default: False)
    """
    
    ymn=-2800000
    ymx=-1600000
    
    tf_fo=fusion("fo","foL",ip)
    tf_fa=fusion("fa","faL",ip)
    t_fo=tf_fo.x[0]
    t_fa=tf_fa.x[0]
    
    x=np.linspace(0.001, 0.999,50)
    g_fo=fo.g_tp(tt,ip)
    g_fa=fa.g_tp(tt,ip)
    g_foL=foL.g_tp(tt,ip)
    g_faL=faL.g_tp(tt,ip)
    
    gs=np.array([])
    gL=np.array([])
    
    if (tt > t_fa) and (tt < t_fo): 
        for ix in x:
            igs=ix*g_fo+(1-ix)*g_fa+R*tt*(ix*np.log(ix)+(1-ix)*np.log(1-ix))+W*ix*(1-ix)
            igL=ix*g_foL+(1-ix)*g_faL+R*tt*(ix*np.log(ix)+(1-ix)*np.log(1-ix))
            gs=np.append(gs,igs)
            gL=np.append(gL,igL)
        
        comp,_=composition(tt,ip,W,prt=False) 
        cs=comp[0]
        cL=comp[1]
        
        gts=cs*g_fo+(1-cs)*g_fa+R*tt*(cs*np.log(cs)+(1-cs)*np.log(1-cs))+W*cs*(1-cs)
        gtL=cL*g_foL+(1-cL)*g_faL+R*tt*(cL*np.log(cL)+(1-cL)*np.log(1-cL))
        
        min_g=np.min([gL,gs])
        max_g=np.max([gL,gs])
        
        xs=(cs,cL)
        ys=(gts,gtL)
        
        xr=(cs,cL)
        res=np.polyfit(xs,ys,1)
        yr=np.polyval(res,xr)
        
        ymin=min_g+0.01*min_g
        ymax=max_g-0.01*max_g
        
        line1x=(cs,cs)
        line2x=(cL,cL)
        if not yl:
           line1y=(ymin,gts)
           line2y=(ymin,gtL)
        else: 
           line1y=(ymn,gts)
           line2y=(ymn,gtL)
    
        plt.figure()
        plt.plot(x,gs,label="Solido")
        plt.plot(x,gL,label="Liquido")
        plt.plot(xr,yr,"k--",label="Solido+liquido")
        plt.plot(line1x,line1y,"g--")
        plt.plot(line2x,line2y,"g--")
        plt.xlim(0., 1)
        if not yl:
           plt.ylim(ymin,ymax)
        else:
           plt.ylim(ymn,ymx)
        plt.xlabel("X (Mg)")
        plt.ylabel("G")
        tlt="Temperatura: "+str(tt)+" K"
        plt.title(tlt)
        plt.yticks([])
        plt.legend(frameon=False)
        if save:
           name="G_fig_"+str(tt)+".png"
           plt.savefig(name,dpi=dp)
        plt.show()
        
        print("Composizione delle fasi: frazione molare X(Mg):")
        print("Fase liquida: %5.2f" % cL)
        print("Fase solida:  %5.2f" % cs)
    else:
        for ix in x:
            igs=ix*g_fo+(1-ix)*g_fa+R*tt*(ix*np.log(ix)+(1-ix)*np.log(1-ix))+W*ix*(1-ix)
            igL=ix*g_foL+(1-ix)*g_faL+R*tt*(ix*np.log(ix)+(1-ix)*np.log(1-ix))
            gs=np.append(gs,igs)
            gL=np.append(gL,igL)
            
        min_g=np.min([gL,gs])
        max_g=np.max([gL,gs])
        ymin=min_g+0.01*min_g
        ymax=max_g-0.01*max_g
        
        plt.figure()
        plt.plot(x,gs,label="Solido")
        plt.plot(x,gL,label="Liquido")
        plt.xlim(0., 1)
        if not yl:
           plt.ylim(ymin,ymax)
        else:
           plt.ylim(ymn,ymx)
        plt.xlabel("X (Mg)")
        plt.ylabel("G")
        plt.yticks([])
        tlt="Temperatura: "+str(tt)+" K"
        plt.title(tlt)
        plt.legend(frameon=False)
        if save:
           name="G_fig_"+str(tt)+".png"
           plt.savefig(name,dpi=dp)
        plt.show()

# --- Sezione Perplex like ----

        
def perplex(it,ip,inat,W,prt=False,nx=20):
    '''
    Calcola la composizione della fasi solida e liquida
    all'equilibrio, data una certa temperatura e pressione, e
    per una composizione globale del componente "a" (Mg) nel 
    sistema.
    
    Il calcolo è Perplex-like, nel senso vengono generati
    pseudocomposti dei quali viene valutata l'energia libera;
    gli pseudocomposti a minore energia libera sono quelli
    che corrispondono alle fasi effettivamente in equilibrio.
    
    Input:
        it   - temperatura (K)
        ip   - pressione   (GPa)
        inat - composizione globale componente Mg
        W    - parametro di Margules
        prt  - se True, stampa le composizioni degli pseudo-composti
               considerati (default: False)    
        nx   - densità della griglia X/Q per la ricerca numerica
               di X e Q (composizioni e quantità relative di S e L;
               default: 20)       
    '''
    xas_l, xal_l,qs_l,ql_l,chk_l=perplex_comp(inat,nx)
    mu_l=perplex_g(it,ip,xas_l,xal_l,qs_l,ql_l,W)
    mu_min=np.min(mu_l)
    ipos=np.argmin(mu_l)
    xas_min=xas_l[ipos]
    qs_min=qs_l[ipos]
    xal_min=xal_l[ipos]
    ql_min=ql_l[ipos]
   
    if prt:     
       serie=(xas_l.round(3), xal_l.round(3),qs_l.round(3),ql_l.round(3),chk_l.round(3),mu_l.round(5))
       pd.set_option('colheader_justify', 'center')
       df=pd.DataFrame(serie, index=['X(Mg)s','X(Mg)l','qs','ql','Check','G'])
       df=df.T
       print("")
       print(inat)
       print(df.to_string(index=False)) 
    
    print("Calcolo Perplex-like\nTemperatura: %4.1f K,  Pressione: %4.1f GPa" %
          (it, ip))
    print("Composizione globale X(Mg) pari a %4.2f" % inat)
    print("X(Mg) fase solida %4.2f, quantità fase solida %4.2f" % 
          (xas_min, qs_min))
    print("X(Mg) fase liquida %4.2f, quantità fase liquida %4.2f" %
          (xal_min, ql_min))
    
    plt.figure()
    plt.scatter(xas_l,qs_l,s=10)
    plt.scatter(xas_min, qs_min, s=30, c='red')
    tlt="Griglia X/Q di campionamento per X(Mg)totale pari a " + str(inat) + "\n"
    plt.title(tlt)
    plt.xlabel("X(Mg)s")
    plt.ylabel("qs")    
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
  
def perplex_comp(inat,nx):
    '''
    Calcola le possibili composizioni degli pseudocomposti
    e le abbondanze relative delle due fasi in equilibrio, a 
    partire da una fissata composizione globale del componente "a" (Mg)
    
    Input: 
        inat - composizione globale del componente "a"
        nx   - densità della griglia X/Q per la ricerca numerica
               di X e Q (composizioni e quantità relative di S e L)
    '''
    
    xas=np.linspace(0.001,0.999,nx)
    qs=np.linspace(0.001,0.999,nx)
    
    xas_l=np.array([])
    xal_l=np.array([])
    qs_l=np.array([])
    ql_l=np.array([])
    chk_l=np.array([])
    
    for ixas in xas:        # loop sulle composizioni X(Mg) in S
        for iqs in qs:      # loop sulle moli di fase S
            nas=iqs*ixas    # moli di Mg in S
            nal=inat-nas    # moli di Mg in L
            iql=1-iqs       # moli di fase L
            ixal=nal/iql    # composizione X(Mg) in L
            
            # Controlli:
            # 0.001 < X(Mg)L < 0.999 e
            # chk = inat
            if ixal >= 0.999 or ixal <= 0.001: continue
            chk=ixas*iqs+ixal*iql  
            
            xas_l=np.append(xas_l,ixas)
            xal_l=np.append(xal_l,ixal)
            qs_l=np.append(qs_l,iqs)
            ql_l=np.append(ql_l,iql)
            chk_l=np.append(chk_l,chk)
            
    return xas_l,xal_l,qs_l,ql_l,chk_l

def perplex_g(it,ip,xas_l,xal_l,qs_l,ql_l,W):
    
    mu_l=np.array([])
    for ixs,ixl,qs,ql in zip(xas_l,xal_l,qs_l,ql_l):        
        gs=ixs*fo.g_tp(it,ip)+(1-ixs)*fa.g_tp(it,ip)+R*it*(ixs*np.log(ixs)+(1-ixs)*np.log(1-ixs))
        gs=gs+W*ixs*(1-ixs)
        gl=ixl*foL.g_tp(it,ip)+(1-ixl)*faL.g_tp(it,ip)+R*it*(ixl*np.log(ixl)+(1-ixl)*np.log(1-ixl))
        mu=qs*gs+ql*gl
        mu_l=np.append(mu_l,mu)
        
    return mu_l