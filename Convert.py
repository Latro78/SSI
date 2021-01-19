import numpy as np
def convertInputArray(array, pixelsCount):
    arrayConverted = np.where(array <= 0, -1, 1)
    arrayConverted = arrayConverted.reshape(pixelsCount)
    return arrayConverted