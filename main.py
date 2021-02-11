class Puzzle:
    def __init__(self, initialState):
        self.initialState = initialState
        self.currentState = initialState

    def isValid(self):
        if len(self.currentState) != 9:
            return False
        
        for char in "12345678_":
            if char not in self.currentState:
                return False
        
        return True

    def isFinished(self):
        return self.currentState == "12345678_"


# Puzzle Unit Tests
def puzzle_tests():
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

puzzle_tests()