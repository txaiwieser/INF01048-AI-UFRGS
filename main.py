from enum import Enum

DEBUG = True

def debugPrint(str):
    if DEBUG: print("DEBUG: " + str)

class Direction(Enum):
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4
    
    def description(self):
        if self == Direction.LEFT: return "esquerda"
        if self == Direction.RIGHT: return "direita"
        if self == Direction.TOP: return "acima"
        if self == Direction.BOTTOM: return "abaixo"
    
class Puzzle:
    def __init__(self, initialState, action = None, parent = None, cost = 0):
        debugPrint(f"New Puzzle initialized \"{ initialState }\" \"{ action }\" \"{ parent.currentState if parent else None }\" \"{ cost }\"")
        self.currentState = initialState
        self.action = action
        self.parent = parent
        self.cost = cost

    def isValid(self):
        result = True
        if len(self.currentState) != 9:
            result = False
        else:
            for char in "12345678_":
                if char not in self.currentState:
                    result = False
        
        debugPrint(f"Called isValid with result = { result }")
        return result

    def isFinished(self):
        result = self.currentState == "12345678_"
        debugPrint(f"Called isFinished with result = { result }")
        return result
    
    def emptyIndex(self):
        result = self.currentState.index("_")
        debugPrint(f"Called emptyIndex with result = { result }")
        return result

    def availableDirections(self):
        result = []
        index = self.emptyIndex()
        
        if index > 2:
            result.append(Direction.TOP)
        
        if index < 6:
            result.append(Direction.BOTTOM)
        
        if index not in [0, 3, 6]:
            result.append(Direction.LEFT)

        if index not in [2, 5, 8]:
            result.append(Direction.RIGHT)
        
        debugPrint(f"Called availableDirections with result = { result }")
        return result

    def applyDirection(self, direction):
        if direction not in self.availableDirections():
            raise Exception("ERROR: Direction can't be applied to current state.")

        if direction == Direction.TOP:
            index = self.emptyIndex()
            newIndex = index - 3
        elif direction == Direction.BOTTOM:
            index = self.emptyIndex()
            newIndex = index + 3
        elif direction == Direction.LEFT:
            index = self.emptyIndex()
            newIndex = index - 1
        elif direction == Direction.RIGHT:
            index = self.emptyIndex()
            newIndex = index + 1
        
        newState = list(self.currentState)
        newState[index], newState[newIndex] = newState[newIndex] + newState[index]
        result = "".join(newState)
        debugPrint(f"Called applyDirection with new state = { result }")
        self.action = direction
        self.currentState = result

    def successors(self):
        directions = self.availableDirections()
        
        def fnc(direction):
            puzzle = Puzzle(self.currentState, self.action, self.parent, self.cost)
            puzzle.applyDirection(direction)
            return puzzle
        
        return list(map(fnc, directions))
            
    def expand(self):
        successors = self.successors()
        
        def fnc(successor):
            successor.cost += 1
            successor.parent = self
            return successor
        
        return list(map(fnc, successors))

    def path(self):
        if self.parent is None:
            return []
        parentPath = self.parent.path()
        parentPath.append(self.action)
        return parentPath

    def breadthFirstSearch(self):
        debugPrint('Starting BFS')
        def fnc(frontier):
            return frontier.pop(0)
        return self.__graphSearch__(fnc)
    
    def depthFirstSearch(self):
        debugPrint('Starting DFS')
        def fnc(frontier):
            return frontier.pop()
        return self.__graphSearch__(fnc)

    def __graphSearch__(self, removeFromFrontier):
        expandedNodes = 0
        explored = []
        frontier = []
        frontier.append(self)
        while True:
            if len(frontier) == 0: 
                raise Exception("ERROR: Can't search graph with empty frontier.")
            v = removeFromFrontier(frontier)
            if v.isFinished():
                debugPrint('Graph search finished with ' + str(expandedNodes) + ' expanded nodes and cost ' + str(v.cost))
                return v.path()
            if v.currentState not in explored:
                explored.append(v.currentState)
                sucessors = v.expand()
                frontier.extend(sucessors)
                expandedNodes += 1

    def __eq__(self, other): 
        if not isinstance(other, Puzzle):
            return NotImplemented
        return self.currentState == other.currentState and self.action == other.action and self.parent == other.parent and self.cost == other.cost

### Puzzle Unit Tests
def puzzle_tests():
    debugPrint("Starting tests")
    def assertIsValid():
        # Checks if sequence "12345678_" is a valid state
        obj = Puzzle("12345678_")
        assert obj.isValid() == True

        # Checks if sequence "_23541687" is a valid state
        obj = Puzzle("_23541687")
        assert obj.isValid() == True

        # Checks if a sequence without a "_" is invalid
        obj = Puzzle("123456789")
        assert obj.isValid() == False

        # Checks if a sequence with more than one "_" is invalid
        obj = Puzzle("_1234567_")
        assert obj.isValid() == False

        # Checks if a sequence with less than 9 characters invalid
        obj = Puzzle("1234567_")
        assert obj.isValid() == False

        # Checks if a sequence with more than 9 characters invalid
        obj = Puzzle("123456789_")
        assert obj.isValid() == False

        # Checks if a sequence with characters diferent from 0-9 and _ is invalid
        obj = Puzzle("1234A678_")
        assert obj.isValid() == False
    
    assertIsValid()

    def assertIsFinished():
        # Checks if a sequence is finished
        obj = Puzzle("12345678_")
        assert obj.isFinished() == True

        # Checks if a sequence is finished
        obj = Puzzle("_23541687")
        assert obj.isFinished() == False
    
    assertIsFinished()

    def assertEmptyIndex():
        obj = Puzzle("12345678_")
        assert obj.emptyIndex() == 8

        obj = Puzzle("_23541687")
        assert obj.emptyIndex() == 0

        obj = Puzzle("1234_5678")
        assert obj.emptyIndex() == 4
    
    assertEmptyIndex()

    def assertAvailableDirections():
        obj = Puzzle("_12345678")
        assert obj.availableDirections() == [Direction.BOTTOM, Direction.RIGHT]

        obj = Puzzle("1_2345678")
        assert obj.availableDirections() == [Direction.BOTTOM, Direction.LEFT, Direction.RIGHT]

        obj = Puzzle("12_345678")
        assert obj.availableDirections() == [Direction.BOTTOM, Direction.LEFT]

        obj = Puzzle("123_45678")
        assert obj.availableDirections() == [Direction.TOP, Direction.BOTTOM, Direction.RIGHT]

        obj = Puzzle("1234_5678")
        assert obj.availableDirections() == [Direction.TOP, Direction.BOTTOM, Direction.LEFT, Direction.RIGHT]

        obj = Puzzle("12345_678")
        assert obj.availableDirections() == [Direction.TOP, Direction.BOTTOM, Direction.LEFT]

        obj = Puzzle("123456_78")
        assert obj.availableDirections() == [Direction.TOP, Direction.RIGHT]

        obj = Puzzle("1234567_8")
        assert obj.availableDirections() == [Direction.TOP, Direction.LEFT, Direction.RIGHT]

        obj = Puzzle("12345678_")
        assert obj.availableDirections() == [Direction.TOP, Direction.LEFT]

    
    assertAvailableDirections()

    def assertApplyDirection():
        obj = Puzzle("123_45678")
        obj.applyDirection(Direction.TOP)
        assert obj.currentState == "_23145678"
        assert obj.action == Direction.TOP

        obj = Puzzle("1234567_8")
        obj.applyDirection(Direction.TOP)
        assert obj.currentState == "1234_6758"
        assert obj.action == Direction.TOP

        obj = Puzzle("12_345678")
        obj.applyDirection(Direction.BOTTOM)
        assert obj.currentState == "12534_678"
        assert obj.action == Direction.BOTTOM

        obj = Puzzle("1234_5678")
        obj.applyDirection(Direction.BOTTOM)
        assert obj.currentState == "1234756_8"
        assert obj.action == Direction.BOTTOM

        obj = Puzzle("12_345678")
        obj.applyDirection(Direction.LEFT)
        assert obj.currentState == "1_2345678"
        assert obj.action == Direction.LEFT

        obj = Puzzle("1234567_8")
        obj.applyDirection(Direction.LEFT)
        assert obj.currentState == "123456_78"
        assert obj.action == Direction.LEFT

        obj = Puzzle("_12345678")
        obj.applyDirection(Direction.RIGHT)
        assert obj.currentState == "1_2345678"
        assert obj.action == Direction.RIGHT

        obj = Puzzle("1234567_8")
        obj.applyDirection(Direction.RIGHT)
        assert obj.currentState == "12345678_"
        assert obj.action == Direction.RIGHT

    assertApplyDirection()

    def assertSucessors():
        obj = Puzzle("_12345678")
        assert obj.successors() == [Puzzle("312_45678", Direction.BOTTOM), Puzzle("1_2345678", Direction.RIGHT)]
        
        obj = Puzzle("1234_5678")
        assert obj.successors() == [Puzzle("1_3425678", Direction.TOP), Puzzle("1234756_8", Direction.BOTTOM), Puzzle("123_45678", Direction.LEFT), Puzzle("12345_678", Direction.RIGHT)]

    assertSucessors()

    def assertExpand():
        obj = Puzzle("_12345678")
        assert obj.expand() == [Puzzle("312_45678", Direction.BOTTOM, obj, 1), Puzzle("1_2345678", Direction.RIGHT, obj, 1)]
        
        obj = Puzzle("1234_5678")
        assert obj.expand() == [Puzzle("1_3425678", Direction.TOP, obj, 1), Puzzle("1234756_8", Direction.BOTTOM, obj, 1), Puzzle("123_45678", Direction.LEFT, obj, 1), Puzzle("12345_678", Direction.RIGHT, obj, 1)]

    assertExpand()

    def assertPath():
        obj1 = Puzzle("_12345678")
        obj2 = Puzzle("1_2345678", Direction.RIGHT, obj1, 1)
        obj3 = Puzzle("1423_5678", Direction.BOTTOM, obj2, 2)
        obj4 = Puzzle("142_35678", Direction.LEFT, obj3, 3)
        assert obj4.path() == [Direction.RIGHT, Direction.BOTTOM, Direction.LEFT]
        
    assertPath()

    def assertBreadthFirstSearch():
        obj = Puzzle("12345_786")
        assert obj.breadthFirstSearch() == [Direction.BOTTOM]
        
    assertBreadthFirstSearch()

    def assertDepthFirstSearch():
        obj = Puzzle("12345_786")
        assert obj.depthFirstSearch() == [Direction.LEFT, Direction.LEFT, Direction.BOTTOM, Direction.RIGHT, Direction.RIGHT, Direction.TOP, Direction.LEFT, Direction.LEFT, Direction.BOTTOM, Direction.RIGHT, Direction.RIGHT, Direction.TOP, Direction.LEFT, Direction.LEFT, Direction.BOTTOM, Direction.RIGHT, Direction.RIGHT, Direction.TOP, Direction.LEFT, Direction.LEFT, Direction.BOTTOM, Direction.RIGHT, Direction.RIGHT, Direction.TOP, Direction.LEFT, Direction.LEFT, Direction.BOTTOM, Direction.RIGHT, Direction.RIGHT]
        
    assertDepthFirstSearch()

    debugPrint("ALL TESTS PASSED!!")

puzzle_tests()


### Public API


## Assignment 1
def sucessor(estado):
    puzzle = Puzzle(estado)
    assert puzzle.isValid() == True
    def fnc(element):
        return f"({ element.action.description() },{ element.currentState })"

    result = " ".join(map(fnc, puzzle.successors()))
    print(result)

sucessor("2_3541687")

## Assignment 2
def expande(estado, custo):
    puzzle = Puzzle(estado, cost = custo)
    assert puzzle.isValid() == True
    def fnc(element):
        return f"({ element.action.description() },{ element.currentState },{ element.cost },{ element.parent.currentState })"

    result = " ".join(map(fnc, puzzle.expand()))
    print(result)

expande("2_3541687", 0)

# Assignment 3.1
def avalia_bfs(estado):
    puzzle = Puzzle(estado)
    assert puzzle.isValid() == True
    def fnc(element):
        return f"{ element.description() }"

    result = " ".join(map(fnc, puzzle.breadthFirstSearch()))
    print(result)

avalia_bfs("123456_78")

## Assignment 3.2
def avalia_dfs(estado):
    puzzle = Puzzle(estado)
    assert puzzle.isValid() == True
    def fnc(element):
        return f"{ element.description() }"

    result = " ".join(map(fnc, puzzle.depthFirstSearch()))
    print(result)

avalia_dfs("123456_78")

