#
# Este codigo tiene como fin tranformar archivos de bin a csv (y guardarlos)
#
from subprocess import call
from os import listdir

####### Parametros Editables ###############
convertir_archivo = False
convertir_carpeta = True
nombre_archivo 	  = 'archivo'
nombre_carpeta	  = 'nombre_carpeta'
############################################

if convertir_archivo:
	pop = 'nombre_archivo' 	
	call(['./read', pop+'.bin', pop+'.csv']) 	

if convertir_carpeta:
	for item in listdir(carpeta):
	    call(['./read', carpeta+item[:-4]+'.bin', carpeta+item[:-4]+'.csv'])



