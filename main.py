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
    def __init__(self, initialState):
        debugPrint(f"New Puzzle initialized \"{ initialState }\"")
        self.currentState = initialState

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
        self.currentState = result

    def successors(self):
        directions = self.availableDirections()
        
        def fnc(direction):
            puzzle = Puzzle(self.currentState)
            puzzle.applyDirection(direction)
            return (direction, puzzle.currentState)
        
        return list(map(fnc, directions))
            
    def expand(self, cost):
        successors = self.successors()
        
        def fnc(successor):
            return (successor[0], successor[1], cost + 1, self.currentState)
        
        return list(map(fnc, successors))


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

        obj = Puzzle("1234567_8")
        obj.applyDirection(Direction.TOP)
        assert obj.currentState == "1234_6758"

        obj = Puzzle("12_345678")
        obj.applyDirection(Direction.BOTTOM)
        assert obj.currentState == "12534_678"

        obj = Puzzle("1234_5678")
        obj.applyDirection(Direction.BOTTOM)
        assert obj.currentState == "1234756_8"

        obj = Puzzle("12_345678")
        obj.applyDirection(Direction.LEFT)
        assert obj.currentState == "1_2345678"

        obj = Puzzle("1234567_8")
        obj.applyDirection(Direction.LEFT)
        assert obj.currentState == "123456_78"

        obj = Puzzle("_12345678")
        obj.applyDirection(Direction.RIGHT)
        assert obj.currentState == "1_2345678"

        obj = Puzzle("1234567_8")
        obj.applyDirection(Direction.RIGHT)
        assert obj.currentState == "12345678_"

    assertApplyDirection()

    def assertSucessors():
        obj = Puzzle("_12345678")
        assert obj.successors() == [(Direction.BOTTOM, "312_45678"), (Direction.RIGHT, "1_2345678")]
        
        obj = Puzzle("1234_5678")
        assert obj.successors() == [(Direction.TOP, "1_3425678"), (Direction.BOTTOM, "1234756_8"), (Direction.LEFT, "123_45678"), (Direction.RIGHT, "12345_678")]

    assertSucessors()

    def assertExpand():
        obj = Puzzle("_12345678")
        assert obj.expand(0) == [(Direction.BOTTOM, "312_45678", 1, "_12345678"), (Direction.RIGHT, "1_2345678",1 , "_12345678")]
        
        obj = Puzzle("1234_5678")
        assert obj.expand(42) == [(Direction.TOP, "1_3425678", 43, "1234_5678"), (Direction.BOTTOM, "1234756_8", 43, "1234_5678"), (Direction.LEFT, "123_45678", 43, "1234_5678"), (Direction.RIGHT, "12345_678", 43, "1234_5678")]

    assertExpand()

    debugPrint("ALL TESTS PASSED!!")

puzzle_tests()


### Public API


## Assignment 1
def sucessor(estado):
    puzzle = Puzzle(estado)
    assert puzzle.isValid() == True
    def fnc(element):
        return f"({ element[0].description() },{ element[1] })"

    result = " ".join(map(fnc, puzzle.successors()))
    print(result)

sucessor("2_3541687")

## Assignment 2
def expande(estado, custo):
    puzzle = Puzzle(estado)
    assert puzzle.isValid() == True
    def fnc(element):
        return f"({ element[0].description() },{ element[1] },{ element[2] },{ element[3] })"

    result = " ".join(map(fnc, puzzle.expand(custo)))
    print(result)

expande("2_3541687", 0)

