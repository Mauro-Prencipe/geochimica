# Calcolo dell'entropia di stato standard (P=1 atm, T=298 K) del piropo
# a partire dai dati sperimentali del calore specifico misurato (a P=1 atm) 
# a determinati valori di temperatura tra 0 K e 500 K

# Strategia del calcolo
# Ci si determina la funzione Cp(T) per "best fit" sui dati sperimentali
# si integra la funzione Cp(T)/T rispetto a T, tra 0 e 298 K; il valore
# dell'integrale e' l'entropia di stato standard

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
# Sono predisposte due funzioni ("set" e "out") per caricare e scrivere i
# parametri del fit 
class fpar():
    def __init__(self,size):
        self.par=np.ones(size)
    def set(self,par):
        self.par=par
    def out(self):
        print("Stored parameters of the Cp function")
        index=0
        for ip in self.par:
            print("parameter %i, value: %6.4e" % (index, self.par[index]))
            index=index+1

# reg e' un'istanza della classe fpar
# reg.par contiene i parametri determinati dalla funzione fit
# reg e' inizializzata con fpar(7) dove 7 e' il numero di coefficienti
# usati nella costruzione della funzione Cp; se questa funzione
# viene modificata con una variazione nel numero di coefficienti,
# la variabile reg deve essere inizializzata opportunamente.
reg=fpar(7)
    
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
# Cp(T) = a + bT + cT^-1 + dT^2 + eT^-2 + fT^0.5 + gT^-0.5
# i valori dei coefficienti a, b, c, d, e, f, g sono contenuti
# nella variabile par.  
def Cp(T,*par):
    cp=par[0]+par[1]*T + par[2]*T**(-1) + par[3]*T**2 + \
       par[4]*T**(-2) + par[5]*T**0.5 + par[6]*T**(-0.5)
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


# ---  Calcolo dell'entropia
    
# definizione della funzione integranda Cp(T)/T a cui devono essere
# passati i parametri a, b, c...   
def integrand(T,par):
    return Cp(T,*par)/T

# Calcolo dell'entropia attraverso l'integrazione numerica della 
# funzione Cp/T (funzione "integrand").
# Il calcolo usa la funzione "quad" (importata da SciPy), tra i limiti
# 10 K e T; nella chiamata di quad si specifica anche la lista di 
# parametri a, b, c... di Cp(T) determinati tramite la funzione fit
def entropia(T):
    ent=quad(integrand, 10, T, args=reg.par)
    return ent[0]

# Plot della funzione entropia da 10K fino alla temperatura T
def plot_entropy(T):
    T_plot=np.linspace(10,T,100)
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

# La funzione start(T) fa partire il calcolo effettivo, eseguendo
# il fit, costruendo i plot Cp ed S(T) fino alla temperatura T,
# e scrivendo il valore dell'entropia a 298 K
def start(T):
    fit(prt=False)
    check_cp()
    plot_entropy(T)
    ent_st=entropia(298)
    print("\nEntropia di stato standard: %6.2f J/mole K" % ent_st)
    
        

    

