# Trasformata di Legendre

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

# Definizione della funzione di cui fare la trasformata di Legendre
# La funzione e' definita all'interno di una classe ("function")
# allo scopo di poter essere modificata nel corso dell'utilizzo
# del programma, senza dover modificare il programma e rilanciarlo
# La funzione con cui è inizializzata qualunque istanza della classe
# e' "e^(-x)". La funzione puo' essere modificata usando il metodo
# "set" che accetta come argomento una nuova funzione sotto forma di
# stringa. Il metodo "print" stampa la funzione
class function():
    def __init__(self):
        self.f_str='np.e**(-1*x)'
        self.func=lambda x: np.e**(-1*x)
    def set(self,ff):
        self.f_str=ff
        self.func=lambda x: eval(ff)
    def print(self):
        print(self.f_str)

# Istanza della classe function
# Per modificare la funzione su cui fare la trasformata, per esempio
# la funzione x^2, usare il metodo "set":
# f.set('x**2')        
f=function()

# retta y=mx+q: la funzione calcola il valore di y dato x, il coeff. angolare
# m e l'intercetta q
def retta(x,m,q):
    return m*x+q

# Costruzione grafica dell'inviluppo di rette tangenti alla funzione "func"
# La funzione "legendre" accetta come argomenti i valori minimo e massimo di x
# e il numero di punti in cui tracciare la retta tangente
# Il parametro opzionale "plot" se "True", consente il disegno del grafico
# Se legendre e' chiamata ponendo plot=False, il grafico non e'
# effettuato (cio' serve quando legendre sia chiamata dalla funzione
# legendre_plot); in tal caso legendre restituisce i valori dei coeff.
# angolari e delle intercette di tutte le rette tangenti alla curva func
def legendre(minx,maxx,npoint,plot=True):

  x_list=np.linspace(minx,maxx,npoint)
  y_list=f.func(x_list)

  der_list=np.array([])
  for ix in x_list:
      der=derivative(f.func, ix, dx=1e-6,order=3)
      der_list=np.append(der_list,der)
    
  q_list=y_list-der_list*x_list

  if plot:
     x_range=(maxx-minx)/1
     npr=2
     plt.figure()
     index=0
     for ix in x_list:
        xmin=ix-x_range/2
        xmax=ix+x_range/2
        xr=np.linspace(xmin,xmax,npr)
        yr=retta(xr,der_list[index],q_list[index])
        plt.plot(xr,yr,"b-",linewidth=0.6)
        index=index+1

     x_func=np.linspace(minx,maxx,100)
     y_func=f.func(x_func)
     plt.plot(x_func,y_func,"k-", linewidth=3)
     factor=0.1
     shiftx=factor*max(x_func)
     shifty=factor*max(y_func)
     plt.ylim(min(y_func)-shifty,max(y_func)+shifty)
     plt.xlim(minx-shiftx,maxx+shiftx)
     plt.xlabel("x")
     plt.ylabel("y")
     plt.title("Funzione e sue tangenti")
     plt.show()
  if not plot:
      return der_list,q_list

# Grafico della trasformata di Legendre q=y-m*x  
def legendre_plot(minx,maxx,npoint):
    m,q=legendre(minx,maxx,npoint,plot=False)
    im=np.argsort(m)
    m_list=m[im]
    q_list=q[im]
    plt.figure()
    plt.plot(m_list,q_list,"k-")
    plt.xlabel("m (coefficiente angolare)")
    plt.ylabel("q (intercetta)")
    plt.title("Trasformata di Legendre")
    plt.show()
    
