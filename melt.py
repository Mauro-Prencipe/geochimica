# Calcolo del diagramma di stato TX del sistema forsterite-fayalite
# alla pressione voluta.

# Modello ideale per la soluzione solida
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

R=8.3145
    
class name_data:
    def __init__(self):
        self.mineral_names=[]
    def add(self,nlist):
        self.mineral_names.extend(nlist)
    
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

fo=mineral("forsterite", "fo")
fa=mineral("fayalite","fa")
foL=mineral("fo_liquid","foL")
faL=mineral("fa_liquid","faL")


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
    with open('melt_db.dat') as f:
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


def melt(ip=0,nt=10,tfmax=0.):
    """
    Calcola il diagramma di stato TX del sistema fayalite-forsterite
    
    Input:
        ip    - pressione (GPa)
        nt    - numero di punti in temperatura
        tfmax - se non 0, fissa il massimo di temperatura per il grafico 
    """
    
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
            
    for it in t_list:
        
        comp,g_t=composition(it,ip,prt=False)
        comp_s=comp[0]
        comp_L=comp[1] 

        solid=np.append(solid,comp_s)
        liquid=np.append(liquid,comp_L)          
         
   
    plt.figure()
    plt.plot(solid,t_list,"k-",label="solido")
    plt.plot(liquid,t_list,"b-",label="liquido")
    plt.xlabel("X(Mg)")
    plt.ylabel("T (K)")
    if tf_flag:
        plt.ylim(1000.,tfmax)
    else:
        plt.ylim((t_fa-500,t_fo+100))
    plt.xlim((0,1.))
    title="Diagramma TX alla pressione di " + str(ip) + " GPa"
    plt.title(title)
    plt.legend(frameon=False)
    plt.show()

    xs_mg=solid
    xs_fe=1.-solid
    xL_mg=liquid
    xL_fe=1-liquid
    serie=(t_list.round(2),xs_mg.round(2),xs_fe.round(2),xL_mg.round(2),xL_fe.round(2))      
    pd.set_option('colheader_justify', 'center')
    df=pd.DataFrame(serie, index=['T (K)','X(Mg)sol','X(Fe)sol','X(Mg)liq','X(Fe)liq'])
    df=df.T
    print("")
    print(df.to_string(index=False))         
    
      
def g(x,it,gv):
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
         
    gab=x1a*gas+x2a*gbs+R*it*(x1a*np.log(x1a)+x2a*np.log(x2a))
    gcd=x1b*gaL+x2b*gbL+R*it*(x1b*np.log(x1b)+x2b*np.log(x2b))
    
    return ra*gab+rb*gcd
       
def refine(x,temp,gv):
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
        
    if x < 0.25:
        xmin=0.001
    else:
        xmin=x-0.25
    if x > 0.75:
        xmax=0.999
    else:
        xmax=x+0.25
        
    x_list=np.linspace(xmin,xmax,5)
    
    g2_list=np.array([])
    for ix in x_list:
        gs=ix*gas+(1-ix)*gbs+R*temp*(ix*np.log(ix)+(1-ix)*np.log(1-ix))
        gL=ix*gaL+(1-ix)*gbL+R*temp*(ix*np.log(ix)+(1-ix)*np.log(1-ix))
        ig2=(gs-gL)**2
        g2_list=np.append(g2_list,ig2)
        
    reg=np.polyfit(x_list,g2_list,2)
    der=np.polyder(reg,1)
    x_ref=-1*der[1]/der[0]

    return x_ref

def composition(it,ip,prt=True,xval=-1.):   
        """
        Determina la composizione delle fasi liquida e solida all'equilibrio
        e le quantità di liquido e solido data una composizione globale
    
        Input:
          it   -  temperatura (K)
          ip   -  pressione (GPa)
          prt  -  stampa i risultati (default: True)
          xval -  se diverso da -1 calcola le quantita' di solido e liquido
                  per la composizione globale xval; se xval=-1 il calcolo
                  viene fatto alla composizione per la quale le due curve
                  di energia libera si intersecano (default xval = -1) 
        """
    
        nx=30
        x_list=np.linspace(0.00001,0.99999,nx)
        gas=fo.g_tp(it,ip)
        gaL=foL.g_tp(it,ip)
        gbs=fa.g_tp(it,ip)
        gbL=faL.g_tp(it,ip)
        
        g_t=(gas,gaL,gbs,gbL)
        
        xe_list=np.array([])
        dg2l=np.array([])
        for ix in x_list:
            gs=ix*gas+(1-ix)*gbs+R*it*(ix*np.log(ix)+(1-ix)*np.log(1-ix))
            gL=ix*gaL+(1-ix)*gbL+R*it*(ix*np.log(ix)+(1-ix)*np.log(1-ix))
            dg2=(gs-gL)**2
            dg2l=np.append(dg2l,dg2)
        
        inx=np.argmin(dg2l)
        ix_int=x_list[inx]
        
        ix_int=refine(ix_int,it,g_t)
        
        x0=(ix_int,ix_int,0.5,0.5)
    
        con1={'type': 'eq', 'fun': lambda x: x[2]+x[3]-1}
        con2={'type': 'eq', 'fun': lambda x: x[0]*x[2]+x[1]*x[3]-ix_int}
    
        eq_cons=[con1,con2]
    
        bounds=((0.000001,0.999999),(0.000001,0.999999),(0.00001,0.99999),(0.00001,0.99999))
  
        res = minimize(g, x0, args=(it,g_t), bounds=bounds, method='slsqp',
                          constraints=eq_cons, options={'ftol':1e-8})
        
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


def g_phase(tt,ip=0.,save=False,dp=600.,yl=False):
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
            igs=ix*g_fo+(1-ix)*g_fa+R*tt*(ix*np.log(ix)+(1-ix)*np.log(1-ix))
            igL=ix*g_foL+(1-ix)*g_faL+R*tt*(ix*np.log(ix)+(1-ix)*np.log(1-ix))
            gs=np.append(gs,igs)
            gL=np.append(gL,igL)
        
        comp,_=composition(tt,ip,prt=False) 
        cs=comp[0]
        cL=comp[1]
        
        gts=cs*g_fo+(1-cs)*g_fa+R*tt*(cs*np.log(cs)+(1-cs)*np.log(1-cs))
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
            igs=ix*g_fo+(1-ix)*g_fa+R*tt*(ix*np.log(ix)+(1-ix)*np.log(1-ix))
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
    
        
            
    
        
    
        
            
            
 


            
