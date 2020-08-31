#
# Este codigo tiene como fin entrenar la arquitectura del perceptron multicapa
#
import numpy as np
import matplotlib.pyplot as plt
import torch

####### Parametros Editables ###############
B   = 250              #batch_size      
arq = [64,64,64,64]    #arquitectura
nro_epocas = 1500      #Nro de epocas
reentrenamiento = False    
modelo_reentrenamiento = 'modelo'
epoca0 = 0             #epocas preentrenadas
############################################

trn_delay   = np.load('delays_medidos.npy')  
trn_tgt     = np.load('fuentes_pos.npy')     

trn_data                        = np.zeros( (len(trn_delay), len(trn_delay[0]) + 3) ) 
trn_data[:, :len(trn_delay[0])] = trn_delay
trn_data[:, len(trn_delay[0]):] = trn_tgt

device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")

print('Tamaño del dataset:', len(trn_delay))

###
P = len(trn_data)           
N = len(trn_delay[0])       
C = 3                       
###

trn_load = torch.utils.data.DataLoader(dataset = trn_data, batch_size = B, shuffle = True)

class MLP(torch.nn.Module):
    def __init__(_,sizes):
        super().__init__()
        _.layers = torch.nn.ModuleList()
        for i in range(len(sizes)-1):
            _.layers.append(torch.nn.Linear(sizes[i],sizes[i+1]))
    def forward(_,x):
        h = x
        for hidden in _.layers[:-1]: 
            h = torch.tanh(hidden(h))
        output = _.layers[-1] 
        y = output(h)
        return y

arq.insert(0,N)
arq.append(C)

model = MLP(arq).to(device)

if reentrenamiento:
    model.load_state_dict(torch.load(modelo_reentrenamiento))

costf = torch.nn.MSELoss(reduce = False)
optim = torch.optim.RMSprop(model.parameters(),lr = 1e-4)

t    = 0
errs = []

model.train() 

while (t + epoca0) < nro_epocas: 
    e = np.array([])
    for row in trn_load:
        optim.zero_grad()
        x =  row[:, :len(trn_delay[0])].float().to(device)
        z =  row[:, len(trn_delay[0]):].float().to(device)
        y = model(x)
        cost = costf(y,z)
        error = cost.mean()
        error.backward()
        optim.step()
        e = np.append(e, cost.sum(dim = 1).sqrt().mean().item())
        
    t += 1 
    E = np.mean(e) 
    errs.append(E)
    
    if t%10 == 0:
        print('-------------- Entrenando ', str(((t+epoca0)/nro_epocas)*100) + '%','--------------------')
        print('Numero de epocas: ' , t)
        print('Error promedio: ', E)
        print('------------------------------------------')
    
    if t%25 == 0:
        np.save('errores.npy' , errs)
        torch.save(model.state_dict(), 'modelo_' + str(t+epoca0) +'_epocas')

model.eval() 




