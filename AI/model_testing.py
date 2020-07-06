#Cargo el modelo ya entrenado y lo testeo

import numpy as np
import matplotlib.pyplot as plt
import torch



device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")

#Cargo los datos de testeo
tst_delay = np.load('delays_medidos_test.npy')
tst_tgt = np.load('fuentes_pos_test.npy')

tst_data = np.zeros((len(tst_delay), len(tst_delay[0]) + 3)) 
tst_data[:, :len(tst_delay[0])] = tst_delay
tst_data[:, len(tst_delay[0]):] = tst_tgt

#Uso el data loader para tener todo bien puesto 
BB = 500 #Batch size del dataset de testeo
tst_load = torch.utils.data.DataLoader(dataset = tst_data, batch_size = BB, shuffle = True)

N = len(tst_delay[0])
C = 3 #Cantidad de unidades de salida son 3 porque es la salida x, y, z

#Declaro la clase de mlp
class MLP(torch.nn.Module):
    def __init__(self,sizes):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(sizes)-1):
            self.layers.append(torch.nn.Linear(sizes[i],sizes[i+1]))
    def forward(self,x):
        h = x
        for hidden in self.layers[:-1]:
            h = torch.tanh(hidden(h))
        output = self.layers[-1]
        y = output(h)
        return y

arq = [64,64,64,64] ##############Elijo que arquitectura quiero

arq.insert(0,N)
arq.append(C)
model = MLP(arq).to(device)

###############################################################################

###CARGO MI MODELO YA ENTRENADO
model.load_state_dict(torch.load('0'))

###TESTEO EL MODELO
model.eval() #Empiezo a evaluar el modelo


#Funcion de costo que uso
costf = torch.nn.MSELoss(reduce= False)

e = np.array([]) #Cada elemento de e va a ser el error de cada batch del tst_load
counter = 0
with torch.no_grad():
    for row in tst_load: #Esto es un minibatch
        counter += 1
        if counter%50000 == 0:
            print(counter)
        x = row[:, :len(tst_delay[0])].float().to(device)
        z = row[:, len(tst_delay[0]):].float().to(device)
        y = model(x)
        error = costf(y,z).sum(dim=1).sqrt()
        e = np.append(e,error.detach().numpy())
        
error_promedio = np.mean(e) #El error en cada epoca es el promedio del error que obtuve en cada batch, en esa epoca
desviacion_error = np.std(e) #Desviacion estandar de los errores
media_error = np.median(e)
    
print('El error promedio de la arquitectura es ' + str(error_promedio) + 'm')
print('El desvio estandar de la arquitectura es ' + str(desviacion_error)+ 'm')
print('La media de la arquitectura es ' + str(media_error)+ 'm')

######################
'''
#Ploteo los errores de entrenamiento
err  = np.load('10.npy')

f1 = plt.figure()
plt.plot(err,'.-')
plt.title('Errores de entrenamiento')
'''

'''
#Para ver los pesos de una capa:
f = plt.plot()
plt.imshow(np.abs(model.layers[0].weight.detach().numpy()), cmap = 'plasma')
plt.colorbar(label = 'Módulo de las activaciones')
plt.xlabel('Diferencias temporales')
plt.ylabel('Neuronas de la primera capa oculta')
'''
