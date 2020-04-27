import time
import numpy as np
import matplotlib.pyplot as plt


nombreArchivo = 'TEST_20000000_000702.csv' #Aca va el nombre del archivo
ti = time.time()
data = np.loadtxt(nombreArchivo, skiprows = 2, dtype = 'U')
print('Importado')
t1 = time.time()
overrunsChecking =  np.char.find(data,'Overruns')
print('Overruns chequeados')
overrunsPositions = np.where(overrunsChecking == 0)
t2 = time.time()

print(len(overrunsPositions))
print('Tiempo parcial', t1-ti)
counter = 0
overruns = 0
for pos in overrunsPositions[0]:
	val = int(data[pos + counter - overruns][9:])
	data = np.insert(data, pos+counter+1 - overruns, float(data[pos + counter - overruns-1])*np.ones(val))
	data = np.delete(data,pos+counter - overruns)
	counter += val
	overruns += 1

tf = time.time()

print('tardo en cargar el archivo: ', t1-ti)
print('tardo en ver overruns: ', t2-t1)
print('tardo en reemplazar: '  , tf-t2)





