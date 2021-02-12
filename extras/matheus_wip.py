def avalia_sucessor(inputState):
    state = parseInputState(inputState)
    return successor(state)

def avalia_expande(inputState, cost):
    state = parseInputState(inputState)
    state['cost'] = cost
    return expand(state)

def avalia_bfs(inputState):
    state = parseInputState(inputState)
    state['cost'] = 0
    return searchGraph(state, 'BFS')

def avalia_dfs(inputState):
    state = parseInputState(inputState)
    state['cost'] = 0
    return searchGraph(state, 'DFS')

def avalia_astar_h1(inputState):
    state = parseInputState(inputState)
    state['cost'] = 0
    return searchGraph(state, 'ASTARH1')

def parseInputState(inputState):
    state = {
        'matrix': [['1','2','3'],['4','5','6'],['7','8','_']],
        'emptySpaceX': 2,
        'emptySpaceY': 2
    }
    for index, char in enumerate(inputState):
        charX = index // 3
        charY = index % 3
        state['matrix'][charX][charY] = char
        if char == '_':
            state['emptySpaceX'], state['emptySpaceY'] = charX, charY
    print('Input state: ', state)
    return state

def expand(state):
    successors = successor(state)
    for s in successors:
        linkStates(state, s)
    #print('Expandido: ', successors)  
    return successors

def linkStates(parentState, childState):
    childState.update({
        'parent': parentState, 
        'cost': parentState['cost'] + 1 }
    )

def successor(state):
    successors = []
    emptySpaceX, emptySpaceY = state['emptySpaceX'], state['emptySpaceY']
    if emptySpaceX > 0:
        successors.append({
            'action':'acima', 
            'matrix': applyAction(state, 'acima')['matrix'],
            'emptySpaceX': applyAction(state, 'acima')['emptySpaceX'],
            'emptySpaceY': applyAction(state, 'acima')['emptySpaceY']
        })
    if emptySpaceX < 2:
        successors.append({
            'action':'abaixo', 
            'matrix': applyAction(state, 'abaixo')['matrix'],
            'emptySpaceX': applyAction(state, 'abaixo')['emptySpaceX'],
            'emptySpaceY': applyAction(state, 'abaixo')['emptySpaceY']
        })
    if emptySpaceY > 0:
        successors.append({
            'action': 'esquerda', 
            'matrix': applyAction(state, 'esquerda')['matrix'],
            'emptySpaceX': applyAction(state, 'esquerda')['emptySpaceX'],
            'emptySpaceY': applyAction(state, 'esquerda')['emptySpaceY']
        })
    if emptySpaceY < 2:
        successors.append({
            'action': 'direita',
            'matrix': applyAction(state, 'direita')['matrix'],
            'emptySpaceX': applyAction(state, 'direita')['emptySpaceX'],
            'emptySpaceY': applyAction(state, 'direita')['emptySpaceY']
        })    
    print('Sucessores: ', successors)    
    return successors    

def applyAction(state, action):
    stateAfterAction = copyState(state)
    emptySpaceX, emptySpaceY = state['emptySpaceX'], state['emptySpaceY']
    if(action == 'acima'):        
        valueAbove = stateAfterAction['matrix'][emptySpaceX - 1][emptySpaceY]
        stateAfterAction['matrix'][emptySpaceX][emptySpaceY] = valueAbove
        stateAfterAction['matrix'][emptySpaceX - 1][emptySpaceY] = '_'
        stateAfterAction['emptySpaceX'] -= 1
    elif(action == 'abaixo'):        
        valueBelow = stateAfterAction['matrix'][emptySpaceX + 1][emptySpaceY]
        stateAfterAction['matrix'][emptySpaceX][emptySpaceY] = valueBelow
        stateAfterAction['matrix'][emptySpaceX + 1][emptySpaceY] = '_'
        stateAfterAction['emptySpaceX'] += 1
    elif(action == 'esquerda'):        
        valueLeft = stateAfterAction['matrix'][emptySpaceX][emptySpaceY - 1]
        stateAfterAction['matrix'][emptySpaceX][emptySpaceY] = valueLeft
        stateAfterAction['matrix'][emptySpaceX][emptySpaceY - 1] = '_'
        stateAfterAction['emptySpaceY'] -= 1
    elif(action == 'direita'):        
        valueRight = stateAfterAction['matrix'][emptySpaceX][emptySpaceY + 1]
        stateAfterAction['matrix'][emptySpaceX][emptySpaceY] = valueRight
        stateAfterAction['matrix'][emptySpaceX][emptySpaceY + 1] = '_'
        stateAfterAction['emptySpaceY'] += 1
    return stateAfterAction

def copyState(state):
    newState = {
        'matrix': [['1','2','3'],['4','5','6'],['7','8','_']],
        'emptySpaceX': 2,
        'emptySpaceY': 2
    }
    for indexX, row in enumerate(state['matrix']):
        for indexY, element in enumerate(row):
            newState['matrix'][indexX][indexY] = element
    newState['emptySpaceX'] = state['emptySpaceX']
    newState['emptySpaceY'] = state['emptySpaceY']
    return newState

def searchGraph(state, algorithm):
    explored = []
    frontier = []
    frontier.append(state)
    foundPath = False
    while not foundPath:
        if len(frontier) == 0: 
            print('Empty frontier!')
            break
        if(algorithm == 'BFS'):
            v = frontier.pop(0)
        elif(algorithm == 'DFS'):
            v = frontier.pop()
        elif(algorithm == 'ASTARH1'):
            bestCandidate = { 'index': 0, 'distance': hammingDistance(frontier[0]) }
            for index, s in enumerate(frontier):
                dist = hammingDistance(s)
                if dist < bestCandidate['distance']:
                    bestCandidate['index'] = index
                    bestCandidate['distance'] = dist
            v = frontier.pop(bestCandidate['index'])
        if isObjective(v):
            foundPath = True
            print('Path: ', path(v, state))
            break
        if v['matrix'] not in explored:
            explored.append(v['matrix'])
            sucessors = expand(v)
            frontier.extend(sucessors)

def isObjective(state):
    return state['matrix'] == [['1','2','3'],['4','5','6'],['7','8','_']] 

def path(endState, startState):
    return endState

def hammingDistance(state):
    objectiveState = [['1','2','3'],['4','5','6'],['7','8','_']] 
    misplacedPieces = 0
    for indexX, row in enumerate(state['matrix']):
        for indexY, element in enumerate(row):
            if element != objectiveState[indexX][indexY]:
                misplacedPieces += 1
    return misplacedPieces 

print('_23541786')

avalia_sucessor('123_56478')
avalia_expande('123_56478', 0)
avalia_bfs('123_56478')
avalia_dfs('123_56478')
avalia_astar_h1('123_56478')
