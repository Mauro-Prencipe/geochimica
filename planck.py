# Programma Python per il calcolo dello spettro
# di emissione di un corpo che si trova alla 
# temperatura T (spettro di "corpo nero")

# librerie da importare nell'ambiente Python
import numpy as np
import matplotlib.pyplot as plt

# Costanti "a" e "b" della legge di Planck
a=3.7469543e+20
b=14397733.98663405
c=299000000

# Costante di Stefan
sigma=5.6225e-8

#-------- Sezione di input ---------

# Temperatura (in K) per la quale effettuare il calcolo
Temp=5777.

# Valori minimo e massimo (in nm) dell'intervallo di lunghezze 
# d'onda su cui effettuare il calcolo
lam_min=50.
lam_max=2500.


# "Risoluzione": res=5. significa che il calcolo viene
# effettuato per lunghezze d'onda intervallate di 5 nm
# a partire da lam_min, fino a lam_max 
res=5.

#---- Fine della sezione di input -------

# Numero di valori di lunghezze 
npoint=int((lam_max-lam_min)/res)

# Generazione della lista di valori delle 
# lunghezze d'onda
lam_list=np.linspace(lam_min, lam_max, npoint)

# Funzione di Planck
# La funzione riceve come argomenti un valore di lunghezza
# d'onda (lam) e un valore di temperatura (tt)
def planck(lam,tt):
    f1=np.exp(b/(lam*tt)) - 1
    denom=f1*lam**5
    return a/denom

# Applicazione della funzione planck su tutti i valori
# di lunghezza d'onda contenuti nella lista lam_list:
    
# Inizializzazione di una lista che conterra' i risultati
power_list=np.array([])

# Ciclo su tutti i valori contenuti in lam_list
for il in lam_list:
    p_l=planck(il, Temp)
    power_list=np.append(power_list, p_l) 
    
    
Int_tot=sigma*Temp**4


print("\nFlusso (dalla legge di Stefan): %5.4e W/m^2" % Int_tot)  

# Parte grafica
plt.figure()
plt.plot(lam_list, power_list,"k-")
plt.xlabel("Lunghezza d'onda (nm)")
plt.ylabel("Power (W/nm m^2)")
plt.title("T=" + str(Temp))
plt.ylim(0)
plt.xlim(0)
# 
# plt.savefig('filename.png', dpi=600)
plt.show()

       
        
    

    

    

