#Este codigo tiene el unico proposito de convertir un archivo .bin en uno .csv

from subprocess import call
from os import listdir

'''
########################################################################
#Esto es para hacer la convertir un solo archivo

pop = '2' #Nombre del archivo .bin (sin la extension) que quiero transformar a .csv
call(['./read', pop+'.bin', pop+'.csv']) #Esta linea es la que crea el archivo .csv

#Listo, el archivo .csv se crea y se guarda
########################################################################
'''

########################################################################
#Esto es por si quiero hacer todo el proceso para todos los archivos de una misma carpeta sin tener que hacerlo para cada uno de los archivos por separado

carpeta = 'Mas_mediciones_con_distintas_frecuencias/27/D/' #CUIDADO: Siempre el último caracter tiene que ser un /. Sino, no va a tirar error pero no va a hacer la conversión.

for item in listdir(carpeta):
    
    call(['./read', carpeta+item[:-4]+'.bin', carpeta+item[:-4]+'.csv'])

########################################################################


