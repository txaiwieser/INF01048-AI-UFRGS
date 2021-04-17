import sys
import numpy as np

# CONSTANTS 
DEBUG = False

# HELPERS
def debugPrint(str):
    if DEBUG: print("DEBUG: " + str)

# MAIN

rawCSVData = np.genfromtxt('/Users/txaiwieser/GitHub/INF01048-AI-UFRGS/t1/p3/house_prices_train.csv', delimiter=',')

#Extrair colunas para análise
indexOfGrLivArea = 46 #columnLabels.index("GrLivArea")
indexOfSalePrice = 80 #columnLabels.index("SalePrice")
grLivAreas = np.array(rawCSVData[1:,indexOfGrLivArea])
salePrices = np.array(rawCSVData[1:,indexOfSalePrice])

# print(grLivAreas)
# print(salePrices)

minGrLivAreas = grLivAreas.min()
maxGrLivAreas = grLivAreas.max()
def normalizeGrLivAreasDivider(x):
  return (x - minGrLivAreas) / (maxGrLivAreas - minGrLivAreas)
normalizedGrLivAreas = normalizeGrLivAreasDivider(grLivAreas)

data = np.array([normalizedGrLivAreas, salePrices])

# #Gráfico dos dados
# plt.figure(figsize=(10, 6))
# plt.scatter(x,y)
# plt.xlabel('Horas de estudo')
# plt.ylabel('Nota final')
# plt.title('Dados')
# plt.show()

#### Definição da função de custo

def compute_cost(theta_0, theta_1, data):
    """
    Calcula o erro quadratico medio
    
    Args:
        theta_0 (float): intercepto da reta 
        theta_1 (float): inclinacao da reta
        data (np.array): matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    
    Retorna:
        float: o erro quadratico medio
    """
    total_cost = 0
    ### SEU CODIGO AQUI

    def h(x):
      return theta_0 + theta_1 * x

    x = np.array(data[:,0])
    y = np.array(data[:,1])
    n = len(x)
    for i in range(n):
      total_cost += pow(h(x[i]) - y[i], 2)

    total_cost /= n
    return total_cost

#### Define as funções de Gradiente Descendente

def step_gradient(theta_0_current, theta_1_current, data, alpha):
    """Calcula um passo em direção ao EQM mínimo
    
    Args:
        theta_0_current (float): valor atual de theta_0
        theta_1_current (float): valor atual de theta_1
        data (np.array): vetor com dados de treinamento (x,y)
        alpha (float): taxa de aprendizado / tamanho do passo 
    
    Retorna:
        tupla: (theta_0, theta_1) os novos valores de theta_0, theta_1
    """
    
    theta_0_updated = 0
    theta_1_updated = 0
    
    ### SEU CODIGO AQUI
    def h(x):
      return theta_0_current + theta_1_current * x

    x = np.array(data[:,0])
    y = np.array(data[:,1])
    n = len(x)
    for i in range(n):
      theta_0_updated += h(x[i]) - y[i]
      theta_1_updated += (h(x[i]) - y[i]) * x[i]

    theta_0_updated *= 2/n
    theta_1_updated *= 2/n

    return theta_0_current - alpha * theta_0_updated, theta_1_current - alpha * theta_1_updated

def gradient_descent(data, starting_theta_0, starting_theta_1, learning_rate, num_iterations):
    """executa a descida do gradiente
    
    Args:
        data (np.array): dados de treinamento, x na coluna 0 e y na coluna 1
        starting_theta_0 (float): valor inicial de theta0 
        starting_theta_1 (float): valor inicial de theta1
        learning_rate (float): hyperparâmetro para ajustar o tamanho do passo durante a descida do gradiente
        num_iterations (int): hyperparâmetro que decide o número de iterações que cada descida de gradiente irá executar
    
    Retorna:
        list : os primeiros dois parâmetros são o Theta0 e Theta1, que armazena o melhor ajuste da curva. O terceiro e quarto parâmetro, são vetores com o histórico dos valores para Theta0 e Theta1.
    """

    # valores iniciais
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1

    
    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []
    
    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
    theta_0_progress = []
    theta_1_progress = []
    
    # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
    num_iterations = 10
    for i in range(num_iterations):
        cost_graph.append(compute_cost(theta_0, theta_1, data))
        theta_0, theta_1 = step_gradient(theta_0, theta_1, data, alpha=0.0001)
        theta_0_progress.append(theta_0)
        theta_1_progress.append(theta_1)
        
    return [theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress]

#### Executa a função gradient_descent() para obter os parâmetros otimizados, Theta0 e Theta1.

num_iterations = 100
theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress = gradient_descent(data, starting_theta_0=0, starting_theta_1=0, learning_rate=0, num_iterations=num_iterations)

#Imprimir parâmetros otimizados
print ('Theta_0: ', theta_0)
print ('Theta_1: ', theta_1)

#Imprimir erro com os parâmetros otimizados
print ('Erro quadratico medio: ', compute_cost(theta_0, theta_1, data))

#### Gráfico do custo por iteração

# # plt.figure(figsize=(10, 6))
# # plt.plot(cost_graph)
# # plt.xlabel('No. de interações')
# # plt.ylabel('Custo')
# # plt.title('Custo por iteração')
# # plt.show()

# #### Gráfico de linha com melhor ajuste

# # #Gráfico de dispersão do conjunto de dados
# # plt.figure(figsize=(10, 6))
# # plt.scatter(x, y)
# # #Valores preditos de y
# # pred = theta_1 * x + theta_0
# # #Gráfico de linha do melhor ajuste
# # plt.plot(x, pred, c='r')
# # plt.xlabel('Horas de estudo')
# # plt.ylabel('Nota final')
# # plt.title('Melhor ajuste')
# # plt.show()