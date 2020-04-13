# Calcolo dell'energia libera di Gibbs del piropo in funzione della
# temperatura, a pressione costante

# Il calcolo e' fatto a partire dai dati sperimentali del calore specifico
# misurato (a P=1 atm) a determinati valori di temperatura superiori a 298 K

# Strategia del calcolo
# Ci si determina la funzione Cp(T) per "best fit" sui dati sperimentali;
# si integra la funzione Cp(T)/T rispetto a T, tra 0 e 298 K; il valore
# dell'integrale e' l'entropia di stato standard (S0).
# S0 e' salvata nella variabile reg.s0 dove "reg" è un'istanza della 
# classe fpar. Nella stessa classe e' salvata anche l'entalpia di stato
# standard (H0) del piropo, nella variabile reg.h0, e l'energia libera 
# di stato standard (G0) del minerale (variabile reg.g0) 
# G0 = H0 - 298.15*S0
#
# A una diversa temperatura T, la funzione di Gibbs e' calcolata
# integrando Cp tra i limiti 298 e T; questo integrale restituisce
# Delta H = H(T) - H(298.15) = H(T)- H0 --> H(T) = Delta H + H0  
# Ci si calcola poi l'entropia S(T) alla temperatura T integrando
# Cp/T tra i limiti 10K e T
# Ci si calcola poi G(T) = H(T) - T*S(T)
# Il plot che viene dato in funzione di T e' quello di G(T) - G0

# Librerie da importare
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import curve_fit

# La classe fpar e' costruita per contenere i valori dei coefficienti
# del polinomio che esprime Cp in funzione di T; questi parametri
# sono usati da varie funzioni del programma e devono essere
# determinati dalla funzione fit. 
# Sono predisposte delle funzioni "set" per caricare i diversi 
# parametri, e una funzione "out" per visualizzarli
# La lista "ex" contiene gli esponenti delle potenze di T che 
# definisce il polinomio Cp(T)
# La variabile minT definisce il limite minimo per il calcolo
# dell'integrale sull'entropia
class fpar():
    def __init__(self):
        self.flag=False       
        self.ex=(0,1,-1,2, -2,0.5,-0.5)
        self.size=len(self.ex)
        self.par=np.ones(self.size)
        self.h0=-6281969.575
        self.s0=269.5
        self.g0=self.h0-298.15*self.s0
        self.minT=10.
    def set(self,par):
        self.par=par
    def set_h0(self,h):
        self.h0=h
    def set_s0(self,s):
        self.s0=s
    def set_g0(self):
        self.g0=self.h0-298.15*self.s0
    def out(self):
        print("Entalpia di formazione %12.2f J/mole" % self.h0)
        print("Entropia di stato standard %9.2f J/mole K" % self.s0)
        print("Energia libera di Gibbs di stato standard %9.2f J/mole" %\
               self.g0)
        print("\nCoefficienti del polinomio che esprime Cp in funzione")
        print("di T, determinati per 'best fit' dai dati sperimentali")
        index=0
        for ip in self.par:
            print("parameter %i, value: %6.4e" % (index, self.par[index]))
            index=index+1

# reg e' un'istanza della classe fpar
# reg.par contiene i parametri determinati dalla funzione fit
# reg e' inizializzata con fpar() 
reg=fpar()
    
#  ------ Input dei dati sperimentali -------------

# lista dei valori di temperatura a cui il Cp è stato misurato (in K)
T_list=np.array([20., 40., 60., 80., 100., 150., 200., 250., 298.15, 
                 350., 400., 500.])

# lista dei valori del Cp misurato (in J/mole K)
Cp_list=np.array([0.862, 11.054, 33.631, 62.668, 94.270, 171.540,
                 235.850, 286.480, 325.310, 359.030, 385.800, 422.800])

# ------- Fine della sezione di input ------------

# funzione per l'interpolazione (best fit) del Cp(T)
# Si tratta di una funzione del tipo
# Cp(T) = a + bT + cT^-1 + eT^-2 + fT^0.5 + gT^-0.5
# i valori dei coefficienti a, b, c, d, e, f, g sono contenuti
# nella variabile par.  
# Gli esponenti delle potenze di T sono conservati nella lista "reg.ex"
# "ex"
def Cp(T,*par):
    cp=0.
    ix=0
    for ip in reg.ex:
        cp=cp+par[ix]*(T**(reg.ex[ix]))
        ix=ix+1
    return cp

# Best fit della funzione Cp(T)
# La lista reg.par e' inizializzata con valori tutti pari ad 1 al momento
# della sua costruzione (vedi classe fpar, sopra) usando la funzione 
# "ones" di numpy
# La funzione restituisce i parametri ottimizzati a, b, c... che vengono
# salvati nuovamente in reg.par usando la funzione reg.set
def fit(prt=True):
    par=reg.par
    opt,err=curve_fit(Cp, T_list, Cp_list, p0=par)
    reg.set(opt)
    if prt:
       reg.out()

# Controllo grafico della qualita' del fit, con un plot della funzione
# Cp(T) (linea continua) con sovrapposti i dati misurati (asterischi)
def check_cp():
    if not reg.flag:
        fit()
        reg.flag=True
    delta=1.
    Tmin=min(T_list)-delta
    Tmax=max(T_list)+delta
    npoint=100
    if Tmin < 0.5:
        Tmin=0.5
        
    T_plot=np.linspace(Tmin,Tmax,npoint)
    Cp_plot=Cp(T_plot,*reg.par)
    
    plt.figure()
    plt.plot(T_plot,Cp_plot,"k-",label="Cp fit")
    plt.plot(T_list, Cp_list,"k*",label="Cp exp")
    plt.xlabel("T (K)")
    plt.ylabel("Cp (J/mol K)")
    plt.legend(frameon=False)
    plt.title("Calore specifico a pressione costante")
    plt.show()
    
    Cp_fit=np.array([])
    for it in T_list:
        icp=Cp(it,*reg.par)
        Cp_fit=np.append(Cp_fit,icp)
    
    delta=Cp_list-Cp_fit
# Stampa di una tabella di valori T, Cp_exp, Cpfit e delta,
# usando le funzioni della libreria Pandas
    serie=(T_list,Cp_list,Cp_fit,delta)
    df=pd.DataFrame(serie, index=['T','Cp_exp','Cp_fit','delta'])
    df=df.T
    df2=df.round(3)
    print("")
    print(df2.to_string(index=False))   

    
# definizione della funzione integranda Cp(T)/T a cui devono essere
# passati i parametri a, b, c... 
# la funzione è sfruttata per il calcolo dell'entropia
def integrand_s(T,par):
    return Cp(T,*par)/T

# definizione della funzione integranda Cp(T) per il calcolo 
# dell'entalpia
def integrand_h(T,par):
    return Cp(T,*par)

# Calcolo dell'entropia attraverso l'integrazione numerica della 
# funzione Cp/T (funzione "integrand").
# Il calcolo usa la funzione "quad" (importata da SciPy), tra i limiti
# reg.MinT e T; nella chiamata di quad si specifica anche la lista di 
# parametri a, b, c... di Cp(T) determinati tramite la funzione fit
# La funzione "quad" esegue l'integrale per via numerica
def entropia(T):
    ent=quad(integrand_s, reg.minT, T, args=reg.par)
    return ent[0]

# calcolo del Delta H = H(T) - H0
def entalpia(T):
    enth=quad(integrand_h, 298.15, T, args=reg.par)
    return enth[0]

# Plot della funzione entropia da 10K fino alla temperatura T
def plot_entropy(T):
    T_plot=np.linspace(reg.minT,T,100)
    E_plot=np.array([])
    for it in T_plot:
        ie=entropia(it)
        E_plot=np.append(E_plot,ie)
        
    plt.figure()
    plt.plot(T_plot,E_plot)
    plt.xlabel("T (K)")
    plt.ylabel("S (J/mole K)")
    plt.title("Entropia in funzione di T")
    plt.show()

# La funzione start() fa partire il calcolo effettivo, eseguendo
# il fit, costruendo i plot Cp ed S  fino alla temperatura di 500K,
# e scrivendo il valore dell'entropia a 298 K nella variabile reg.s0
def start():
    reg.flag=True
    T=500
    fit(prt=False)
    check_cp()
    plot_entropy(T)
    ent_st=entropia(298.15)
    reg.set_s0(ent_st)
    reg.set_g0()
    print("\nEntropia di stato standard: %6.2f J/mole K" % ent_st)
    print("Energia libera di stato standard: %9.2f J/mole" % reg.g0)
 
# Calcolo del Delta G = G(T) - G0 = H(T) + H0 - T*S(T) 
def gibbs(T,prt=False):
    if not reg.flag:
        print("Eseguo preventivamente la funzione start")
        start()
    enth_T=entalpia(T)+reg.h0
    ent_T=entropia(T)
    gibbs_T=enth_T-T*ent_T
    DG=gibbs_T-reg.g0
    if prt:
        print("Delta G(T): %8.2f J/mole; T: %5.1f K" % (DG,T))
        return
    return DG

# Plot della funzione Delta G = G(T) - G0
def gibbs_plot(T):
    if T < 298.15:
        print("La temperatura deve essere superiore a 298.15 K")
        return
    T_list=np.linspace(298.15, T, 50)
    gibbs_list=np.array([])
    
    for it in T_list:
        ig=gibbs(it)
        gibbs_list=np.append(gibbs_list,ig)
        
    plt.figure()
    plt.plot(T_list,gibbs_list)
    plt.xlabel("T (K)")
    plt.ylabel("Delta G (J/mole)")
    plt.title("Delta G = G(T) - G(298.15)")
    plt.show()

    
        

    

