import numpy as np


def accuracy(y_input, y_target):
    assert len(y_input) == len(y_target)
    y_input = (y_input >= 0.5).astype(float)
    #print(y_input)
    #print(y_target)
    correct = 0
    for y0, y1 in zip(y_input, y_target):
        if y0 == y1:
            correct += 1
    return correct / len(y_input)

