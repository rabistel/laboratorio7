#Esto es el mismo codigo que sist_y_errores_6mics_version_3 pero ahora agrego un for que vaya variando la posicion del pajaro

import numpy as np 
import matplotlib.pyplot as plt 

#Defino algunas cosas para despues poder graficar
it = np.array([]) #Aca voy a guardar las iteraciones
err_rel_u = np.array([]) #Aca voy a guardar los errores relativos de u en cada iteracion
err_rel_v = np.array([]) #Aca voy a guardar los errores relativos de v en cada iteracion
err_rel_w = np.array([]) #Aca voy a guardar los errores relativos de w en cada iteracion
err_rel_c2 = np.array([]) #Aca voy a guardar los errores relativos de c**2 en cada iteracion
err_rel_cd = np.array([]) #Aca voy a guardar los errores relativos de c*d en cada iteracion

val_c2 = np.array([]) #Guardo los valores de c**2

#En cada iteracion del primer for vario aleatoriamente las posiciones de los mics
#En cada iteracion del segundo for vario aleatoriamente la posicion del pajaro
for k in range(100): #Vario mics
    nro_mics = 6
    mics = np.random.rand(6,3) #En metros #Posiciones random de los mics dentro de un metro cubico (si aparece solo np.random.rand(6,3))
    
    for m in range(10): #Vario pajaro
        pajaro = 100*np.random.rand(1,3) #En metros
        c = 350 #m/s #Velocidad de sonido
        tiempos  = np.zeros(nro_mics)
        for i in range(nro_mics):
            tiempos[i] = np.linalg.norm(pajaro-mics[i])/c #Distancia entre cada mic y el pajaro, dividido por la velocidad del sonido
        
        sorting = np.argsort(tiempos)
        tiempos = tiempos[sorting] #Ordeno los tiempos de menor a mayor
        mics = mics[sorting] #Ordeno los microfonos de acuerdo al orden en el cual les llega el sonido
        #Armo la matriz A
        A = (mics[0]-mics)[1:,:] #Esta es la parte de las restas entre las posiciones de los mics
        A = 2*np.column_stack([A, (tiempos[0] - tiempos[1:]), (-1/2)*(tiempos[0] - tiempos[1:])**2])
        #Armo el vector b
        b = np.zeros(nro_mics-1)
        for i in range(nro_mics-1):
            b[i] = np.linalg.norm(mics[0])**2 - np.linalg.norm(mics[i+1])**2
        
        #Ahora directamente voy a llamar A a la matriz A original ampliada con la amtriz b
        A = np.column_stack((A,b))
        
        #Resuelvo mi sistema Ax = b usando directamente numpy por si despues quiero chequar que da bien
        solucion = np.linalg.solve(A[:,:-1],A[:,-1])
        
        #Armo la matriz de errores de A ampliada con b
        er_A = np.zeros((5,6)) #Matriz de errores
        ###Errores que puedo cambiar (en realidad salen de la configuracion de microfonos)
        #'d' significa error
        dt = 2/22000 #En segundos #Error de la medicion del tiempo
        dxi = 0.01;dyi = 0.01;dzi = 0.001;dxj = 0.01;dyj = 0.01;dzj = 0.001 #En metros #Error de la posicion de los microfonos 'i' y 'j' en los tres ejes (x,y,z)
        DT = (tiempos[0] - tiempos[len(tiempos) - 2]) #En segundos #Diferencia temporal tipica entre que el sonido llega a un microfono y luego a otro #Si se pone 1/343, se esta tomando el tiempo que tarda el sonido en recorrer 1 metro
        xi = mics[0,0];yi = mics[0,1];zi = mics[0,2];xj = mics[4,0];yj = mics[4,1];zj = mics[4,2] #En metros #Posiciones de los microfonos 'i' y 'j' en los tres ejes (x,y,z)
        ###Propago los errores
        #Esto es delta(xi-xj) #Esta en metros
        er_A[:,:3] = np.sqrt((dxi)**2 + (dxj)**2)
        #Esto es delta(t2-t1) (donde 1 y 2 indica el numero de microfono) #Esta en segundos #dt1 = dt2 = dt porque se asume que todos los microfonos tienen la misma resolucion temporal
        er_A[:,3] = np.sqrt((dt)**2 + (dt)**2)
        #Esto es delta((t2-t1)^2) #Esta en segundos^2
        er_A[:,4] = 2*np.abs(DT)*np.sqrt((dt)**2 + (dt)**2)
        #Esto es delta (||xi||^2 -||xj||^2) #Esta en metros^2
        er_A[:,5] = 2*np.sqrt((xi*dxi)**2 + (yi*dyi)**2 + (zi*dzi)**2 + (xj*dxj)**2 + (yj*dyj)**2 + (zj*dzj)**2)
        '''
        er_A = np.zeros((5,6))
        er_A[:,:3] = 0.057 #Error de las restas entre las posiciones (metros)
        er_A[:,3] = 0.00013 #Error de -delta T (segundos)
        er_A[:,4] = 1.5E-7 #Error de delta T cuadrado (segundos^2)
        er_A[:,5] = 0.00012 #Error de b (metros^2)
        '''
        
        #Ahora quiero ir propagando los errores a medida que triangulo
        
        #Esto es la propagacion del error de (x1 + (x2*x2)/x4). Esto es una operacion generica que se hace para triangular la matriz A
        def deltaf(x1,x2,x3,x4,err_x1,err_x2,err_x3,err_x4):
            return np.sqrt(err_x1**2 + (x3/x4)**2 * err_x2**2 + (x2/x4)**2 * err_x3**2 + (x2*x3/x4**2)**2 * err_x4**2)
        
        #Armo algo asi como mi sistema A|b triangulado pero todo con unos
        L = np.diag(np.ones(5)) + np.diag(np.ones(4),1) + np.diag(np.ones(3),2) + np.diag(np.ones(2),3) + np.diag(np.ones(1),4)
        L = np.column_stack((L,np.ones(5)))
        
        #Propago los errores
        for i in range(len(A)-1):
            for j in range(i+1,len(A)):
                #print(i,j)
                er_A[j,:] = deltaf(A[j,:], A[j,i], A[i,:], A[i,i], er_A[j,:], er_A[j,i], er_A[i,:], er_A[i,i])
        
            E = np.diag(np.ones(5))
            E[i+1:,i] = A[i+1:,i]*(-1/A[i,i])
            A = np.tensordot(E,A,1)
            #Agrego el pivote para intentar reducir los errores (tratar de no dividir por numeros tan chicos)
            sort = np.argsort(np.abs(A[i+1:,i+1]))
            sort = np.flip(sort) #CHEQUEAR ESTO. ESTO VA SI TENGO QUE ORDENAR DE MAYOR A MENOR. SI TENGO QUE ORDENAR DE MENOR A MAYOR, ENTONCES ESTO NO VA.
            sort = np.concatenate((np.linspace(0,i,i + 1),sort + i + 1))
            sort = [int(x) for x in sort]
            A = A[sort]
            
        er_A = L*er_A #Errores del sistema triangulado
        
        #Ahora resuelve mi sistema Ax = b una vez que ya esta triangulado
        def resolver(A):
            b = A[:,-1]
            A = A[:,:-1]
            x = np.zeros(len(b))
        
            x[-1] = b[-1]/A[-1,-1]
            for i in range(len(b)-1,-1,-1):
                x[i] = (b[i] - np.dot(x[i+1:len(x)],A[i,i+1:len(x)]))*(1/A[i,i])
        
            return x
        
        x = resolver(A) #Solucion de mi sistema Ax = b
        
        #Creo un diccionario con mis resultados
        resultados = {}
        resultados['u'] = x[0]
        resultados['v'] = x[1]
        resultados['w'] = x[2]
        resultados['c*d'] = x[3]
        resultados['c**2'] = x[4]
        
        #Creo un diccionario con mis errores
        #Tengo que propagar las cuentas que hago para resolver el sistema ya triangulado
        errores = {}
        errores['c**2'] = np.sqrt((1/A[4,4])**2 * er_A[4,5]**2 + 
                                (A[4,5]/A[4,4]**2)**2 * er_A[4,4]**2)
        errores['c*d'] = np.sqrt((1/A[3,3])**2 * er_A[3,5]**2 +
                                ((A[3,5]-A[3,4]*resultados['c**2'])/A[3,3]**2)**2 * er_A[3,3]**2 +
                                (resultados['c**2']/A[3,3])**2 * er_A[3,4]**2 + 
                                (A[3,4]/A[3,3])**2 * errores['c**2']**2)
        errores['w'] = np.sqrt((1/A[2,2])**2 * er_A[2,5]**2 + 
                                (resultados['c**2']/A[2,2])**2 * er_A[2,4]**2 + 
                                (A[2,4]/A[2,2])**2 * errores['c**2']**2 + 
                                (A[2,3]/A[2,2])**2 * errores['c*d']**2 + 
                                (resultados['c*d']/A[2,2])**2 * er_A[2,3]**2 +
                                ((A[2,5]-A[2,4]*resultados['c**2'] - A[2,3]*resultados['c*d'])/A[2,2]**2)**2 * er_A[2,2]**2)
        errores['v'] = np.sqrt((1/A[1,1])**2 * er_A[1,5]**2 + 
                                (A[1,2]/A[1,1])**2 * errores['w']**2 + 
                                (resultados['w']/A[1,1])**2 * er_A[1,2]**2 + 
                                (resultados['c*d']/A[1,1])**2 * er_A[1,3]**2 + 
                                (A[1,3]/A[1,1])**2 * errores['c*d']**2 + 
                                (A[1,4]/A[1,1])**2 * errores['c**2']**2 + 
                                (resultados['c**2']/A[1,1])**2 * er_A[1,4] + 
                                ((A[1,5] - resultados['w']*A[1,2] - resultados['c*d']*A[1,3] - resultados['c**2']*A[1,4])/A[1,1]**2)**2 * er_A[1,1]**2)
        errores['u'] = np.sqrt((1/A[0,0])**2 * er_A[0,5]**2 + 
                                (resultados['v']/A[0,0])**2 * er_A[0,1]**2 +
                                (A[0,1]/A[0,0])**2 * errores['v']**2 + 
                                (A[0,2]/A[0,0])**2 * errores['w']**2 + 
                                (resultados['w']/A[0,0])**2 * er_A[0,2] + 
                                (A[0,3]/A[0,0])**2 * errores['c*d']**2 + 
                                (resultados['c*d']/A[0,0])**2 * er_A[0,3] + 
                                (resultados['c**2']/A[0,0])**2 * er_A[0,4] + 
                                (A[0,4]/A[0,0])**2 * errores['c**2']**2 + 
                                ((A[0,5] - resultados['v'] * A[0,1] - resultados['w'] * A[0,2] - resultados['c*d'] * A[0,3] - resultados['c**2']*A[0,4])/A[0,0]**2)**2 * er_A[0,0]**2)
        
        print('Iteración mic:')
        print(k)
        print('Micrófonos:')
        print(mics)
        print('Pájaro:')
        print(pajaro)
        print('Resultados:')
        print(resultados)
        print('Errores:')
        print(errores)
        
        it = np.append(it,k)
        err_rel_u = np.append(err_rel_u,np.abs(errores['u']/resultados['u']))
        err_rel_v = np.append(err_rel_v,np.abs(errores['v']/resultados['v']))
        err_rel_w = np.append(err_rel_w,np.abs(errores['w']/resultados['w']))
        err_rel_c2 = np.append(err_rel_c2,np.abs(errores['c**2']/resultados['c**2']))
        err_rel_cd = np.append(err_rel_cd,np.abs(errores['c*d']/resultados['c*d']))
        
        val_c2 = np.append(val_c2,resultados['c**2'])
        

f1 = plt.figure()
plt.plot(it,err_rel_u,'.')
plt.xlabel('Número de iteración')
plt.ylabel('Error relativo de u')

f2 = plt.figure()
plt.plot(it,err_rel_v,'.')
plt.xlabel('Número de iteración')
plt.ylabel('Error relativo de v')

f3 = plt.figure()
plt.plot(it,err_rel_w,'.')
plt.xlabel('Número de iteración')
plt.ylabel('Error relativo de w')

f4 = plt.figure()
plt.plot(it,err_rel_c2,'.')
plt.xlabel('Número de iteración')
plt.ylabel('Error relativo de c**2')

f5 = plt.figure()
plt.plot(it,err_rel_cd,'.')
plt.xlabel('Número de iteración')
plt.ylabel('Error relativo de c*d')

f5 = plt.figure()
plt.plot(it,val_c2,'.')
plt.xlabel('Número de iteración')
plt.ylabel('Valor de c**2')

#Todos los errores para una misma configuracion de mics se van a ver sobre la mimsa columna (osea mismo numero de iteracion)
#Osea que todos los datos que esten graficados con el mismo nro de iteracion, van a ser datos de una misma configuracion pero con distintas posiciones del pajaro
