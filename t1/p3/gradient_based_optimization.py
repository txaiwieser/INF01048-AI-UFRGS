import sys
import numpy as np
from pathlib import Path

# CONSTANTS 
DEBUG = False
CONVERGENCE_RANGE = 100

# HELPERS
def debugPrint(*strs):
    if DEBUG: print("DEBUG: ", strs)

def normalizeArray(arr):
    min = arr.min()
    max = arr.max()
    def normalize(x):
        return (x - min) / (max - min)
    return normalize(arr)

def hasConverged(*arrs):
    for arr in arrs:
      is_long_enough = len(arr) > CONVERGENCE_RANGE
      all_last_elements_are_the_same = all(t == arr[-CONVERGENCE_RANGE:][0] for t in arr[-CONVERGENCE_RANGE:]) 
      if not is_long_enough or not all_last_elements_are_the_same:
        return False
    return True

# MAIN
path = Path(__file__).parent / 'house_prices_train.csv'
rawCSVData = np.genfromtxt(path, delimiter=',')

#Extrair colunas para análise
indexOfGrLivArea = 46
indexOfSalePrice = 80
indexOfOverallQual = 17
indexOfOverallCond = 18
indexOfGarageArea = 62
indexOfYearBuilt = 19

grLivAreas = np.array(rawCSVData[1:,indexOfGrLivArea])
salePrices = np.array(rawCSVData[1:,indexOfSalePrice])
overallQuals = np.array(rawCSVData[1:,indexOfOverallQual])
overallConds = np.array(rawCSVData[1:,indexOfOverallCond])
garageAreas = np.array(rawCSVData[1:,indexOfGarageArea])
yearsBuilt = np.array(rawCSVData[1:,indexOfYearBuilt])

normalizedGrLivAreas = normalizeArray(grLivAreas)
normalizedOverallQuals = normalizeArray(overallQuals)
normalizedOverallConds = normalizeArray(overallConds)
normalizedGarageAreas = normalizeArray(garageAreas)
normalizedYearsBuilt = normalizeArray(yearsBuilt)

oneParamData = np.array([normalizedGrLivAreas, salePrices]).T
twoParamData = np.array([normalizedGrLivAreas, normalizedOverallQuals, salePrices]).T
fiveParamData = np.array([normalizedGrLivAreas, normalizedOverallQuals, normalizedOverallConds, normalizedGarageAreas, normalizedYearsBuilt, salePrices]).T

#### Definição da função de custo

def one_param_compute_cost(theta_0, theta_1, data):
    total_cost = 0

    def h(x1):
      return theta_0 + theta_1 * x1

    x1 = np.array(data[:,0])
    y = np.array(data[:,1])
    n = len(y)
    
    for i in range(n):
      total_cost += pow(h(x1[i]) - y[i], 2)

    total_cost /= n
    return total_cost

def two_param_compute_cost(theta_0, theta_1, theta_2, data):
    total_cost = 0

    def h(x1, x2):
      return theta_0 + theta_1 * x1 + theta_2 * x2

    x1 = np.array(data[:,0])
    x2 = np.array(data[:,1])
    y = np.array(data[:,2])
    n = len(y)

    for i in range(n):
      total_cost += pow(h(x1[i], x2[i]) - y[i], 2)

    total_cost /= n
    return total_cost

def five_param_compute_cost(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, data):
    total_cost = 0

    def h(x1, x2, x3, x4, x5):
      return theta_0 + theta_1 * x1 + theta_2 * x2 + theta_3 * x3 + theta_4 * x4 + theta_5 * x5

    x1 = np.array(data[:,0])
    x2 = np.array(data[:,1])
    x3 = np.array(data[:,2])
    x4 = np.array(data[:,3])
    x5 = np.array(data[:,4])
    y = np.array(data[:,5])
    n = len(y)

    for i in range(n):
      total_cost += pow(h(x1[i], x2[i], x3[i], x4[i], x5[i]) - y[i], 2)

    total_cost /= n
    return total_cost

#### Define as funções de Gradiente Descendente

def one_param_step_gradient(theta_0_current, theta_1_current, data, alpha):
    theta_0_updated = 0
    theta_1_updated = 0
    
    def h(x1):
      return theta_0_current + theta_1_current * x1

    x1 = np.array(data[:,0])
    y = np.array(data[:,1])
    n = len(y)
    for i in range(n):
      # 2 (a + b x1 + c x2 - y)
      theta_0_updated += (h(x1[i]) - y[i])
      # 2 x1 (a + b x1 + c x2 - y)
      theta_1_updated += (h(x1[i]) - y[i]) * x1[i]

    theta_0_updated *= 2/n
    theta_1_updated *= 2/n
    
    theta_0_final = theta_0_current - alpha * theta_0_updated
    theta_1_final = theta_1_current - alpha * theta_1_updated
    
    return theta_0_final, theta_1_final

def two_param_step_gradient(theta_0_current, theta_1_current, theta_2_current, data, alpha):
    theta_0_updated = 0
    theta_1_updated = 0
    theta_2_updated = 0
    
    def h(x1, x2):
      return theta_0_current + theta_1_current * x1 + theta_2_current * x2

    x1 = np.array(data[:,0])
    x2 = np.array(data[:,1])
    y = np.array(data[:,2])
    n = len(y)

    for i in range(n):
      # 2 (a + b x1 + c x2 - y)
      theta_0_updated += (h(x1[i], x2[i]) - y[i])
      # 2 x1 (a + b x1 + c x2 - y)
      theta_1_updated += (h(x1[i], x2[i]) - y[i]) * x1[i]
      # 2 x2 (a + b x1 + c x2 - y)
      theta_2_updated += (h(x1[i], x2[i]) - y[i]) * x2[i]

    theta_0_updated *= 2/n
    theta_1_updated *= 2/n
    theta_2_updated *= 2/n
    
    theta_0_final = theta_0_current - alpha * theta_0_updated
    theta_1_final = theta_1_current - alpha * theta_1_updated
    theta_2_final = theta_2_current - alpha * theta_2_updated
    
    return theta_0_final, theta_1_final, theta_2_final

def five_param_step_gradient(theta_0_current, theta_1_current, theta_2_current, theta_3_current, theta_4_current, theta_5_current, data, alpha):
    theta_0_updated = 0
    theta_1_updated = 0
    theta_2_updated = 0
    theta_3_updated = 0
    theta_4_updated = 0
    theta_5_updated = 0
    
    def h(x1, x2, x3, x4, x5):
      return theta_0_current + theta_1_current * x1 + theta_2_current * x2 + theta_3_current * x3 + theta_4_current * x4 + theta_5_current * x5

    x1 = np.array(data[:,0])
    x2 = np.array(data[:,1])
    x3 = np.array(data[:,2])
    x4 = np.array(data[:,3])
    x5 = np.array(data[:,4])
    y = np.array(data[:,5])
    n = len(y)

    for i in range(n):
      # 2 (a + b x1 + c x2 + d x3 + e x4 + f x5 - y)
      theta_0_updated += (h(x1[i], x2[i], x3[i], x4[i], x5[i]) - y[i])
      # 2 x1 (a + b x1 + c x2 + d x3 + e x4 + f x5 - y)
      theta_1_updated += (h(x1[i], x2[i], x3[i], x4[i], x5[i]) - y[i]) * x1[i]
      # 2 x2 (a + b x1 + c x2 + d x3 + e x4 + f x5 - y)
      theta_2_updated += (h(x1[i], x2[i], x3[i], x4[i], x5[i]) - y[i]) * x2[i]
      # 2 x3 (a + b x1 + c x2 + d x3 + e x4 + f x5 - y)
      theta_3_updated += (h(x1[i], x2[i], x3[i], x4[i], x5[i]) - y[i]) * x3[i]
      # 2 x4 (a + b x1 + c x2 + d x3 + e x4 + f x5 - y)
      theta_4_updated += (h(x1[i], x2[i], x3[i], x4[i], x5[i]) - y[i]) * x4[i]
      # 2 x5 (a + b x1 + c x2 + d x3 + e x4 + f x5 - y)
      theta_5_updated += (h(x1[i], x2[i], x3[i], x4[i], x5[i]) - y[i]) * x5[i]

    theta_0_updated *= 2/n
    theta_1_updated *= 2/n
    theta_2_updated *= 2/n
    theta_3_updated *= 2/n
    theta_4_updated *= 2/n
    theta_5_updated *= 2/n

    theta_0_final = theta_0_current - alpha * theta_0_updated
    theta_1_final = theta_1_current - alpha * theta_1_updated
    theta_2_final = theta_2_current - alpha * theta_2_updated
    theta_3_final = theta_3_current - alpha * theta_3_updated
    theta_4_final = theta_4_current - alpha * theta_4_updated
    theta_5_final = theta_5_current - alpha * theta_5_updated
    
    return theta_0_final, theta_1_final, theta_2_final, theta_3_final, theta_4_final, theta_5_final

def one_param_gradient_descent(data, starting_theta_0, starting_theta_1, learning_rate, num_iterations):
    # valores iniciais
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1
    
    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []
    
    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
    theta_0_progress = []
    theta_1_progress = []
    
    # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
    has_converged = False
    for i in range(num_iterations):
        cost_graph.append(one_param_compute_cost(theta_0, theta_1, data))
        theta_0, theta_1 = one_param_step_gradient(theta_0, theta_1, data, learning_rate)
        debugPrint(theta_0, theta_1)
        theta_0_progress.append(theta_0)
        theta_1_progress.append(theta_1)
        if hasConverged(theta_0_progress, theta_1_progress):
          has_converged = True
          print('Converged after', i, 'iterations')
          break

    if not has_converged:
      print('Did NOT converge after', num_iterations, 'iterations')

    return [theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress]

def two_param_gradient_descent(data, starting_theta_0, starting_theta_1, starting_theta_2, learning_rate, num_iterations):
    # valores iniciais
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1
    theta_2 = starting_theta_2

    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []
    
    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
    theta_0_progress = []
    theta_1_progress = []
    theta_2_progress = []
    
    # Para cada iteração, obtem novos (Theta0, Theta1) e calcula o custo (EQM)
    has_converged = False
    for i in range(num_iterations):
        cost_graph.append(two_param_compute_cost(theta_0, theta_1, theta_2, data))
        theta_0, theta_1, theta_2 = two_param_step_gradient(theta_0, theta_1, theta_2, data, learning_rate)
        debugPrint(theta_0, theta_1, theta_2)
        theta_0_progress.append(theta_0)
        theta_1_progress.append(theta_1)
        theta_2_progress.append(theta_2)
        if hasConverged(theta_0_progress, theta_1_progress, theta_2_progress):
          has_converged = True
          print('Converged after', i, 'iterations')
          break

    if not has_converged:
      print('Did NOT converge after', num_iterations, 'iterations')
        
    return [theta_0, theta_1, theta_2, cost_graph, theta_0_progress, theta_1_progress, theta_2_progress]

def five_param_gradient_descent(data, starting_theta_0, starting_theta_1, starting_theta_2, starting_theta_3, starting_theta_4, starting_theta_5, learning_rate, num_iterations):
    # valores iniciais
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1
    theta_2 = starting_theta_2
    theta_3 = starting_theta_3
    theta_4 = starting_theta_4
    theta_5 = starting_theta_5

    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []
    
    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
    theta_0_progress = []
    theta_1_progress = []
    theta_2_progress = []
    theta_3_progress = []
    theta_4_progress = []
    theta_5_progress = []
    
    # Para cada iteração, obtem novos (Theta0, Theta1) e calcula o custo (EQM)
    has_converged = False
    for i in range(num_iterations):
        cost_graph.append(five_param_compute_cost(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, data))
        theta_0, theta_1, theta_2, theta_3, theta_4, theta_5 = five_param_step_gradient(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, data, learning_rate)
        debugPrint(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5)
        theta_0_progress.append(theta_0)
        theta_1_progress.append(theta_1)
        theta_2_progress.append(theta_2)
        theta_3_progress.append(theta_3)
        theta_4_progress.append(theta_4)
        theta_5_progress.append(theta_5)
        if hasConverged(theta_0_progress, theta_1_progress, theta_2_progress, theta_3_progress, theta_4_progress, theta_5_progress):
          has_converged = True
          print('Converged after', i, 'iterations')
          break
        
    if not has_converged:
      print('Did NOT converge after', num_iterations, 'iterations')

    return [theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, cost_graph, theta_0_progress, theta_1_progress, theta_2_progress, theta_3_progress, theta_4_progress, theta_5_progress]

#### Executa a função gradient_descent() para obter os parâmetros otimizados, Theta0 e Theta1.

num_iterations = 100

def one_param_compute():
    theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress = one_param_gradient_descent(oneParamData, starting_theta_0=0, starting_theta_1=0, learning_rate=0.666, num_iterations=num_iterations)

    #Imprimir parâmetros otimizados
    print ('Theta_0: ', theta_0)
    print ('Theta_1: ', theta_1)

    #Imprimir erro com os parâmetros otimizados
    print ('Erro quadratico medio: ', one_param_compute_cost(theta_0, theta_1, oneParamData))

def two_param_compute():
    theta_0, theta_1, theta_2, cost_graph, theta_0_progress, theta_1_progress, theta_2_progress = two_param_gradient_descent(twoParamData, starting_theta_0=0, starting_theta_1=0, starting_theta_2=0, learning_rate=0.666, num_iterations=num_iterations)

    #Imprimir parâmetros otimizados
    print ('Theta_0: ', theta_0)
    print ('Theta_1: ', theta_1)
    print ('Theta_2: ', theta_2)

    #Imprimir erro com os parâmetros otimizados
    print ('Erro quadratico medio: ', two_param_compute_cost(theta_0, theta_1, theta_2, twoParamData))

def five_param_compute():
    theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, cost_graph, theta_0_progress, theta_1_progress, theta_2_progress, theta_3_progress, theta_4_progress, theta_5_progress = five_param_gradient_descent(fiveParamData, starting_theta_0=0, starting_theta_1=0, starting_theta_2=0, starting_theta_3=0, starting_theta_4=0, starting_theta_5=0, learning_rate=0.42, num_iterations=num_iterations)

    #Imprimir parâmetros otimizados
    print ('Theta_0: ', theta_0)
    print ('Theta_1: ', theta_1)
    print ('Theta_2: ', theta_2)
    print ('Theta_3: ', theta_3)
    print ('Theta_4: ', theta_4)
    print ('Theta_5: ', theta_5)

    #Imprimir erro com os parâmetros otimizados
    print ('Erro quadratico medio: ', five_param_compute_cost(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, fiveParamData))

print ('ONE PARAM SOLUTION: ')
one_param_compute()
print ('TWO PARAM SOLUTION: ')
two_param_compute()
print ('FIVE PARAM SOLUTION: ')
five_param_compute()