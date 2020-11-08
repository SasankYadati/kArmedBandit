import numpy as np
def argmax(values):
    """
    Takes in a list of values and returns the index of the item 
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in values
    """
    top_value = float("-inf")
    ties = []
    
    for i in range(len(values)):
        if values[i] > top_value:
            top_value = values[i]
            ties = []
        if values[i] == top_value:
            ties.append(i)
    return np.random.choice(ties)

if __name__ == '__main__':
    test_array = [0, 0, 0, 2, 0, 3, 0, 0, 1, 0]
    assert argmax(test_array) == 5, "Check your argmax implementation returns the index of the largest value"
    test_array = [0, 0, 0, 2, 0, 3, 0, 3, 1, 0]
    assert argmax(test_array) in [5, 7], "Check your argmax implementation returns the index of the largest value"
    test_array = [np.inf, -np.inf, 0.5, 0.5, 0.5]
    assert argmax(test_array) == 0, "Check your argmax implementation returns the index of the largest value"