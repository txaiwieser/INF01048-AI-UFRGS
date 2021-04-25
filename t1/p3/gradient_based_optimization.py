import sys
import numpy as np
from pathlib import Path

# CONSTANTS 
DEBUG = False

# HELPERS
def debugPrint(*strs):
    if DEBUG: print("DEBUG: ", strs)

def normalizeArray(arr):
    min = arr.min()
    max = arr.max()
    def normalize(x):
        return (x - min) / (max - min)
    return normalize(arr)

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
      hx = h(x1[i])
      # 2 (a + b x1 + c x2 - y)
      theta_0_updated += hx - y[i]
      # 2 x1 (a + b x1 + c x2 - y)
      theta_1_updated += (hx - y[i]) * x1[i]

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
      hx = h(x1[i], x2[i])
      # 2 (a + b x1 + c x2 - y)
      theta_0_updated += hx - y[i]
      # 2 x1 (a + b x1 + c x2 - y)
      theta_1_updated += (hx - y[i]) * x1[i]
      # 2 x2 (a + b x1 + c x2 - y)
      theta_2_updated += (hx - y[i]) * x2[i]

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
      hx = h(x1[i], x2[i], x3[i], x4[i], x5[i])
      # 2 (a + b x1 + c x2 + d x3 + e x4 + f x5 - y)
      theta_0_updated += hx - y[i]
      # 2 x1 (a + b x1 + c x2 + d x3 + e x4 + f x5 - y)
      theta_1_updated += (hx - y[i]) * x1[i]
      # 2 x2 (a + b x1 + c x2 + d x3 + e x4 + f x5 - y)
      theta_2_updated += (hx - y[i]) * x2[i]
      # 2 x3 (a + b x1 + c x2 + d x3 + e x4 + f x5 - y)
      theta_3_updated += (hx - y[i]) * x3[i]
      # 2 x4 (a + b x1 + c x2 + d x3 + e x4 + f x5 - y)
      theta_4_updated += (hx - y[i]) * x4[i]
      # 2 x5 (a + b x1 + c x2 + d x3 + e x4 + f x5 - y)
      theta_5_updated += (hx - y[i]) * x5[i]

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

def one_param_gradient_descent(data, learning_rate, num_iterations):
    # valores iniciais
    theta_0, theta_1 = 0, 0
    
    # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
    has_converged = False
    for i in range(num_iterations):
        new_theta_0, new_theta_1 = one_param_step_gradient(theta_0, theta_1, data, learning_rate)
        debugPrint(new_theta_0, new_theta_1)
        if (new_theta_0, new_theta_1) == (theta_0, theta_1):
          has_converged = True
          debugPrint('Converged after', i, 'iterations')
          break
        theta_0, theta_1 = new_theta_0, new_theta_1

    if not has_converged:
      debugPrint('Did NOT converge after', num_iterations, 'iterations')

    return [theta_0, theta_1]

def two_param_gradient_descent(data, learning_rate, num_iterations):
    # valores iniciais
    theta_0, theta_1, theta_2 = 0, 0, 0
    
    # Para cada iteração, obtem novos (Theta0, Theta1) e calcula o custo (EQM)
    has_converged = False
    for i in range(num_iterations):
        new_theta_0, new_theta_1, new_theta_2 = two_param_step_gradient(theta_0, theta_1, theta_2, data, learning_rate)
        debugPrint(new_theta_0, new_theta_1, new_theta_2)
        if (new_theta_0, new_theta_1, new_theta_2) == (theta_0, theta_1, theta_2):
          has_converged = True
          debugPrint('Converged after', i, 'iterations')
          break
        theta_0, theta_1, theta_2 = new_theta_0, new_theta_1, new_theta_2

    if not has_converged:
      debugPrint('Did NOT converge after', num_iterations, 'iterations')
        
    return [theta_0, theta_1, theta_2]

def five_param_gradient_descent(data, learning_rate, num_iterations):
    # valores iniciais
    theta_0, theta_1, theta_2, theta_3, theta_4, theta_5 = 0, 0, 0, 0, 0 ,0
    
    # Para cada iteração, obtem novos (Theta0, Theta1) e calcula o custo (EQM)
    has_converged = False
    for i in range(num_iterations):
        new_theta_0, new_theta_1, new_theta_2, new_theta_3, new_theta_4, new_theta_5 = five_param_step_gradient(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, data, learning_rate)
        debugPrint(new_theta_0, new_theta_1, new_theta_2, new_theta_3, new_theta_4, new_theta_5)
        if (new_theta_0, new_theta_1, new_theta_2, new_theta_3, new_theta_4, new_theta_5) == (theta_0, theta_1, theta_2, theta_3, theta_4, theta_5):
          has_converged = True
          debugPrint('Converged after', i, 'iterations')
          break
        theta_0, theta_1, theta_2, theta_3, theta_4, theta_5 = new_theta_0, new_theta_1, new_theta_2, new_theta_3, new_theta_4, new_theta_5
        
    if not has_converged:
      debugPrint('Did NOT converge after', num_iterations, 'iterations')

    return [theta_0, theta_1, theta_2, theta_3, theta_4, theta_5]

#### Executa a função gradient_descent() para obter os parâmetros otimizados, Theta0 e Theta1.

def one_param_compute(num_iterations):
    theta_0, theta_1 = one_param_gradient_descent(oneParamData, learning_rate=0.666, num_iterations=num_iterations)

    #Imprimir parâmetros otimizados
    print ('theta_0:', theta_0)
    print ('theta_1:', theta_1)

    #Imprimir erro com os parâmetros otimizados
    print ('Erro quadratico medio:', one_param_compute_cost(theta_0, theta_1, oneParamData))

def two_param_compute(num_iterations):
    theta_0, theta_1, theta_2 = two_param_gradient_descent(twoParamData, learning_rate=0.666, num_iterations=num_iterations)

    #Imprimir parâmetros otimizados
    print ('theta_0:', theta_0)
    print ('theta_1:', theta_1)
    print ('theta_2:', theta_2)

    #Imprimir erro com os parâmetros otimizados
    print ('Erro quadratico medio:', two_param_compute_cost(theta_0, theta_1, theta_2, twoParamData))

def five_param_compute(num_iterations):
    theta_0, theta_1, theta_2, theta_3, theta_4, theta_5 = five_param_gradient_descent(fiveParamData, learning_rate=0.42, num_iterations=num_iterations)

    #Imprimir parâmetros otimizados
    print ('theta_0:', theta_0)
    print ('theta_1:', theta_1)
    print ('theta_2:', theta_2)
    print ('theta_3:', theta_3)
    print ('theta_4:', theta_4)
    print ('theta_5:', theta_5)

    #Imprimir erro com os parâmetros otimizados
    print ('Erro quadratico medio:', five_param_compute_cost(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, fiveParamData))

### Execução para testes

num_iterations = 10000

print ('ONE PARAM SOLUTION: ')
one_param_compute(num_iterations)
print ('TWO PARAM SOLUTION: ')
two_param_compute(num_iterations)
print ('FIVE PARAM SOLUTION: ')
five_param_compute(num_iterations)