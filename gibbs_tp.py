# import os
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from scipy import integrate


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
ens=mineral("enstatite","en")
cor=mineral("corindone","cor")
py=mineral("pyrope","py")
coe=mineral("coesite", "coe")
q=mineral("quartz","q")
fo=mineral("forsterite", "fo")
ky=mineral("kyanite","ky")
sill=mineral("sillimanite","sill")
andal=mineral("andalusite","and")
per=mineral("periclase","per")
sp=mineral("spinel","sp")
fa=mineral("fayalite","fa")
foL=mineral("Fo_liquid","foL")
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
    with open('perplex_db.dat') as f:
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

# ----------- Reactions ------------------

def equilib(tini,tfin,npoint,pini=1,prod=['py',1], rea=['ens',1.5,'cor', 1]):
    """
    Computes the equilibrium pressure for a reaction involving a
    given set of minerals, in a range of temperatures.
    
    Args:
        tini: minimum temperature in the range
        tfin: maximum temperature in the range
        npoint: number of points in the T range
        pini (optional): initial guess for the pressure
        prod: list of products of the reaction in the form 
              [name_1, c_name_1, name_2, c_name_2, ...]
              where name_i is the name of the i^th mineral, as stored
              in the database, and c_name_i is the corresponding
              stoichiometric coefficient
        rea:  list of reactants; same syntax as the "prod" list.
        
    Example:
        equilib(300, 500, 12, prod=['py',1], rea=['ens', 1.5, 'cor', 1])
    """
    
    lprod=len(prod)
    lrea=len(rea)
    prod_spec=prod[0:lprod:2]
    prod_coef=prod[1:lprod:2]
    rea_spec=rea[0:lrea:2]
    rea_coef=rea[1:lrea:2]
    
    lastr=rea_spec[-1]
    lastp=prod_spec[-1]
    
    prod_string=''
    for pri in prod_spec:
        prod_string=prod_string + pri
        if pri != lastp:
            prod_string=prod_string+' + '
        
    rea_string=''
    for ri in rea_spec:
        rea_string = rea_string + ri
        if ri != lastr:
            rea_string=rea_string+' + '
       
    t_list=np.linspace(tini,tfin,npoint)
    p_list=np.array([])
    h_list=np.array([])
    s_list=np.array([])
    v_list=np.array([])
    cs_list=np.array([])
    
    for ti in t_list:
        pi=pressure_react(ti,pini, prod_spec, prod_coef, rea_spec, rea_coef)
       
        hprod=0.
        sprod=0.
        vprod=0.
        for pri, pci in zip(prod_spec, prod_coef):
            hprod=hprod+(eval(pri+'.h_tp(ti,pi)'))*pci
            sprod=sprod+(eval(pri+'.s_tp(ti,pi)'))*pci
            vprod=vprod+(eval(pri+'.volume_p(ti,pi)'))*pci 
        
        hrea=0.    
        srea=0.
        vrea=0.
        for ri,rci in zip(rea_spec, rea_coef):
            hrea=hrea+(eval(ri+'.h_tp(ti,pi)'))*rci   
            srea=srea+(eval(ri+'.s_tp(ti,pi)'))*rci
            vrea=vrea+(eval(ri+'.volume_p(ti,pi)'))*rci             
        
        hi=hprod-hrea
        si=sprod-srea
        vi=vprod-vrea
        dsdv_i=si/vi
    
        p_list=np.append(p_list,pi)
        h_list=np.append(h_list,hi)
        s_list=np.append(s_list,si)
        v_list=np.append(v_list,vi)
        cs_list=np.append(cs_list, dsdv_i)
                
    serie=(t_list.round(1),p_list.round(2),h_list.round(3), s_list.round(3), \
           v_list.round(4), cs_list.round(2))
    pd.set_option('colheader_justify', 'center')
    df=pd.DataFrame(serie, index=['T (K)','P (GPa)','DH(J/mol)', \
        'DS (J/mol K)', 'DV (J/bar)','Slope (bar/K)'])
    df=df.T
    df2=df.round(3)
    print("")
    print(df2.to_string(index=False))
    
    ymax=max(p_list)+0.1*(max(p_list)-min(p_list))
    ymin=min(p_list)-0.1*(max(p_list)-min(p_list))
    
    xloc_py, yloc_py, xloc_en, yloc_en=field(tini,tfin, ymin, ymax, \
                                prod_spec, prod_coef, rea_spec, rea_coef)
    
    print("\n")
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.title.set_text("Reaction "+ rea_string + " <--> " + prod_string + "\n" )
    ax.text(xloc_en, yloc_en, rea_string)
    ax.text(xloc_py,yloc_py, prod_string)
    ax.plot(t_list,p_list,"k-")
    ax.axis([tini,tfin,ymin,ymax])
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (GPa)")
    plt.show()
    
    clap=np.polyfit(t_list,p_list,1)
    cl_s=clap[0]*1.e4
    print("\nAverage Clapeyron Slope (from Delta S/Delta V): %6.2f bar/K" \
           % cs_list.mean())
    print("Clapeyron slope (from a linear fit of the P/T curve): %6.2f bar/K"\
          % cl_s)
    
    
def reaction(tt,pp, prod_spec, prod_coef, rea_spec, rea_coef):
    """
    Computes the Gibbs free energy of reaction at given 
    temperature (tt) and pressure (pp), involving specified
    minerals.
    """
    
    gprod=0.
    for pri, pci in zip(prod_spec, prod_coef):
        gprod=gprod+(eval(pri+'.g_tp(tt,pp)'))*pci
        
    grea=0.    
    for ri,rci in zip(rea_spec, rea_coef):
        grea=grea+(eval(ri+'.g_tp(tt,pp)'))*rci
        
    return gprod-grea
    
def pressure_react(tt,pini, prod_spec, prod_coef, rea_spec, rea_coef):
    """
    Computes the pressure at which a given set of minerals
    is at the equilibrium for a specified temperature.
    "pini" is a initial guess for the pressure. 
    Output in GPa.
    
    "pressure_react" calls "reactions, and it is invoked  
    by "equilib".
    """
    
    fpr=lambda pp: (reaction(tt,pp, prod_spec, prod_coef, rea_spec, rea_coef))**2      
    pres=scipy.optimize.minimize(fpr,pini,tol=1)

    return pres.x

def field(tmin,tmax,pmin,pmax,\
          prod_spec, prod_coef, rea_spec, rea_coef, nx=6, ny=6):
    
    t_range=np.linspace(tmin,tmax,nx)
    p_range=np.linspace(pmin,pmax,ny)
    
    fld=np.array([])
    for ti in t_range:
        for pi in p_range:
            de=reaction(ti,pi,prod_spec, prod_coef, rea_spec, rea_coef)
            fld=np.append(fld,[ti,pi,de])
            
    fld=fld.reshape(nx*ny,3)
    
    prodx=np.array([])
    prody=np.array([])    
    reax=np.array([])
    reay=np.array([])
    
    for fi in fld:
        if fi[2]>0:
            reax=np.append(reax,fi[0])
            reay=np.append(reay,fi[1])
        else:
            prodx=np.append(prodx,fi[0])
            prody=np.append(prody,fi[1])
            
    return prodx.mean(), prody.mean(), reax.mean(), reay.mean()

def plot_g_t(tmin,tmax,npoint,pres,ret=False):
    t_list=np.linspace(tmin,tmax,npoint)
      
    g_1=np.array([])
    g_2=np.array([])
    g_3=np.array([])
    for it in t_list:
        ig=py.g_tp(it,pres)
        ig2=1.5*ens.g_tp(it,pres)+cor.g_tp(it,pres)
        ig3=3*per.g_tp(it,pres)+cor.g_tp(it,pres)+3*q.g_tp(it,pres)
        g_1=np.append(g_1,ig)
        g_2=np.append(g_2,ig2)
        g_3=np.append(g_3,ig3)
        
    fig=plt.figure()
    ax=fig.add_subplot(111) 
    ax.title.set_text("G in funzione di T, a P costante = "+str(pres)+" GPa")
    ax.plot(t_list,g_1,"k-",label="py")
    ax.plot(t_list,g_2,"r-",label="ens+cor")
    ax.plot(t_list,g_3,"b-",label="per+cor+q")
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%4.2e'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.set_xlabel("Temperatura (K)")
    ax.set_ylabel("G (J/mole)")
    ax.legend(frameon=False)
    plt.show()
    
    if ret:
        return t_list, g_1, g_2, g_3
    
def plot_g_p(pmin,pmax,npoint,temp,ret=False):
    p_list=np.linspace(pmin,pmax,npoint)
      
    g_1=np.array([])
    g_2=np.array([])
    for ip in p_list:
        ig=py.g_tp(temp,ip)
        ig2=1.5*ens.g_tp(temp,ip)+cor.g_tp(temp,ip)
        g_1=np.append(g_1,ig)
        g_2=np.append(g_2,ig2)
        
    fig=plt.figure()
    ax=fig.add_subplot(111) 
    ax.title.set_text("G in funzione di P, a T costante = "+str(temp)+" K")
    ax.plot(p_list,g_1,"k-",label="py")
    ax.plot(p_list,g_2,"r-",label="ens+cor")
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%4.2e'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.set_xlabel("Pressione (GPa)")
    ax.set_ylabel("G (J/mole)")
    ax.legend(frameon=False)
    plt.show()  
    
    if ret:
         return p_list, g_1, g_2
     
        
def adiabat(tini,tfin,nt,pini,pfin,npp,env=0.,nsamp=0, grd=False):
    """
    Computes adiabats on a P/T grid
    
    Input:
        tini, tfin, nt: minimum, maximum and number of point 
                        along the temperature axis
        pini, pfin, npp: minimum, maximum and number of point
                         along the pressure axis
        phase: string of the mineral phase (default = 'py')
        env: if env > 0., value of the entropy correspondent to wanted
             P/T path (default = 0.)  
        nsamp: if > 0, restricts the number of entries of the printed
               P/T/V list to nsamp values (default = 0.). Relevant if
               env > 0.
    """
    t_list=np.linspace(tini,tfin,nt)
    p_list=np.linspace(pini,pfin,npp)
    tg, pg=np.meshgrid(t_list,p_list)
    
    ntp=nt*npp
    
    tgl=tg.reshape(ntp)
    pgl=pg.reshape(ntp)
    
    ent=np.array([])
    index=0
    for it in tgl:
        ip=pgl[index]
        g_py=py.g_tp(it,ip)
        g_ens=ens.g_tp(it,ip)
        g_cor=cor.g_tp(it,ip)
        g_rea=1.5*g_ens+g_cor
        if g_py < g_rea:
           ient=py.s_tp(it,ip)
        else:
           ient=1.5*ens.s_tp(it,ip)+cor.s_tp(it,ip)
        ent=np.append(ent,ient)
        index=index+1  
    
    ent=ent.reshape(npp,nt)
    
    if grd:
       plt.figure()
       plt.scatter(tg,pg,s=20,color='k')
       plt.xlabel("T (K)")
       plt.ylabel("P GPa")
       plt.title("Grid")
       plt.show()
    
    plt.figure()
    if env > 0.:
       con=plt.contour(tg,pg,ent, [env])
       p1=con.collections[0].get_paths()[0]
       path=p1.vertices
    else:
       con=plt.contour(tg,pg,ent)
    
    if env > 0.:
        plt.close()
    else:
        plt.clabel(con, inline=1, fontsize=10)
        plt.xlabel("T (K)")
        plt.ylabel("P (GPa)")
        plt.title("Entropy (J/K mol)")
        plt.show()
    
    if env > 0.:
        t_val=path[:,0]
        p_val=path[:,1]
        plt.figure()
        plt.plot(p_val,t_val)
        plt.ylabel("T (K)")
        plt.xlabel("P (GPa)")
        title="P/T adiabat for an entropy of " + str(env) \
             + " J/(K mol)"
        plt.title(title)
        plt.show()
        
        ipos=p_val.argsort()
        t_val=t_val[ipos]
        p_val=p_val[ipos]
        
        ism=1
        if nsamp > 0:
           lt=len(t_val)
           if lt > nsamp:
              ism=int(lt/nsamp)
              
        index=0
        t_val=t_val[0:-1:ism]
        p_val=p_val[0:-1:ism]
        v_val=np.array([])
        for it in t_val:
            ip=p_val[index]
            dg=py.g_tp(it,ip)-(1.5*ens.g_tp(it,ip)+cor.g_tp(it,ip))
            if dg < 0.:
                iv=py.volume_p(it,ip)
            else:
                iv=1.5*ens.volume_p(it,ip)+cor.volume_p(it,ip)
            v_val=np.append(v_val,iv)
            index=index+1
            
        serie=(p_val.round(2),t_val.round(1),v_val.round(3))
        pd.set_option('colheader_justify', 'center')
        df=pd.DataFrame(serie, index=['P (GPa)','T (K)','Vol (J/bar)'])
        df=df.T
        print("")
        print(df.to_string(index=False))
    
    
    
    
    
    
    
    