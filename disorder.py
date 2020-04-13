# Configurazioni ordinate e disordinate

import numpy as np
import matplotlib.pyplot as plt

ll=np.log(2)   # fattore di conversione per il passaggio da logaritmo 
               # in base "e" a logaritmo in base 2

# Con riferimento alla lezione termodinamica_11, si costruisce qui 
# la funzione N(nT,nC) della slide 9
def myf(nt,nc):
    n=nt+nc
    num=np.math.factorial(n)
    d1=np.math.factorial(nt)
    d2=np.math.factorial(nc)
    return num/(d1*d2)

def myf_stirling(nt,nc):
    if nt==0 or nc==0:
       return 0.
    else:
       n=nt+nc
       t1=n*np.log(n)
       t2=nt*np.log(nt)
       t3=nc*np.log(nc)
       return t1-t2-t3

# Funzione per il calcolo effettivo della probabilità di 'disordine'
# la funzione richiede come argomenti
#   n: numero di elementi
#   or_lim: soglia per definire una configurazione come ordinata
#
# Il grado d'ordine è calcolato dalla funzione order definita più
# avanti, per ogni valore di nT (o, equivalentementem della frazione
# molare di Mg in X1 data da nT/n)
def disorder(n,or_lim,prt=False):
    st_flag=False
    if n>1000:
        print("n = %i troppo alto per il calcolo del fattoriale; si userà" % n)
        print("l'approssimazione di Stirling e verrà calcolata solo l'entropia\n")
        st_flag=True
        
    case=np.arange(0,n+1)
    ss=2**n                 # Numero totale di configurazioni
    prob_ord=0.
    
    iv_list=np.array([])
    entro_list=np.array([])
    
    for ic in case:
        od=order(ic/n)      # ic/n è la frazione molare, "od" è il parametro 
                            # d'ordine
        if not st_flag:
           iv=myf(ic,n-ic)  
           prob=100*iv/ss
           if od >= or_lim:
              prob_ord=prob_ord+prob       
        else:
           iv=myf_stirling(ic,n-ic)
           
        if not st_flag:
           entro=np.log(iv)/ll
        else:
           entro=iv/ll
           
        entro_list=np.append(entro_list,entro)
        
        if not st_flag:
           iv_list=np.append(iv_list,iv)            
           if prt:
              print("nMg = %3i, prob: %5.2f %%, entropia %5.2f, ordinamento %5.2f" \
                   % (ic, prob, entro, od))
                  
    if not st_flag:     
       prob_dis=100.-prob_ord
    
       print("\nSoglia per definire una configurazione ordinata %4.2f" % or_lim)
       print("Probabilità di avere una configurazione ordinata: %5.2f %%" \
            % prob_ord)
       print("Probabilità di avere una configurazione disordinata: %5.2f %%" \
          % prob_dis)
    
    if not st_flag:
       plt.figure()
       plt.plot(case,iv_list,"k-")
       plt.ylabel("Numero di configurazioni")
       plt.xlabel("numero di atomi Mg nel sito X1")
       plt.show()
    
    plt.figure()
    plt.plot(case,entro_list)
    plt.xlabel("Numero di atomi Mg in X1")
    plt.ylabel("Entropia statistica")
    plt.show()

# calcolo del parametro d'ordine in funzione della frazione molare x
# di Mg in X1
def order(x):
    return abs(1-2*x)

def fraction(od):
    if od > 1. or od < 0.:
       print("Parametro d'ordine fuori dai limiti")
       return
    fr=(1.+od)/2
    print("Frazione molare corrispondente al grado d'ordine %3.2f" % fr)


    
    