#
# Este codigo tiene como fin dado un csv reemplazar los overruns
#
import time
import numpy as np
import matplotlib.pyplot as plt

####### Parametros editables
nombrearchivo = 'nombredearchivo' 
guardar = True
##############
ti = time.time()
data = np.loadtxt(nombrearchivo + '.csv', skiprows = 2, dtype = '<U20')
print('Importado')
t1 = time.time()
overrunsChecking =  np.char.find(data,'Overrun')
print('Overruns chequeados')
overrunsPositions = np.where(overrunsChecking == 0)
t2 = time.time()

print(len(overrunsPositions[0]))
print('Tiempo parcial', t1-ti)
counter       = 0
overruns      = 0
for pos in overrunsPositions[0]:
	val 	  = int(data[pos + counter - overruns][9:])
	data 	  = np.insert(data, pos+counter+1 - overruns, float(data[pos + counter - overruns-1])*np.ones(val))
	data 	  = np.delete(data,pos+counter - overruns)
	counter  += val
	overruns += 1

tf = time.time()

print('tiempo para carga de archivo: '		, t1-ti)
print('tiempo de busqueda de overruns: '	, t2-t1)
print('tiempo de reemplazo de overruns: '  	, tf-t2)


if guardar:
	np.save(nombrearchivo + '.csv', data)

