Utilizzo
========

.. |nbsp| unicode:: 0xA0 


Il file di input
----------------

Il nome di default del file di input è *best_fit.txt*. File con nomi diversi possono comunque essere utilizzati (si vedano le funzioni
*fit* e *load_data*).

Il fit
------ 

Se il file di input *best_fit.txt* è presente (obbligatoriamente nello stesso folder del programma), l'interpolazione può essere fatta 
partire con il solo comando

.. code::

   fit()

La funzione *fit* carica in tal caso il file *best_fit.txt* ed esegue l'interpolazione con i parametri di default (il grado del polinomio
interpolante è posto a 2)

La funzione *fit* accetta comunque alcuni parametri e, precisamente:

.. code::

   fit(file='mio_input.txt')

carica i dati da un file di input diverso dal default.

.. code:

   fit(deg=3)

esegue il fit con un polinomio di grado diverso dal default (nell'esempio, 3)

.. code::

   fit(reload=True)

esegue il fit forzando il ricaricamento del file di dati.

.. Note::

   La funzione *load_data* può anche essere utilizzata per caricare i dati dal file di input scelto. Esempio:

   .. code::

      load_data('mio_file.txt')
      fit(deg=3)


La classe drv
-------------

La variabile *drv* (istanza della classe *driver_class*) contiene un *metodo* (*set_xlimit*) per variare l'intervallo di definizione
della variabile indipendente sul quale l'iterpolazione è effettuata:

.. code::

   drv.set_xlimit(xmin, xmax)

dove *xmin* e *xmax* sono i due valori scelti come estremo dell'intervallo. 

Ad esempio:

.. code::

   >>> drv.set_xlimit(0., 5.)
   >>> fit()

L'intervallo originale può essere ripristinato con il metodo *drv.xlimit_reset*:

.. code::

   drv.reset_xlimit()


