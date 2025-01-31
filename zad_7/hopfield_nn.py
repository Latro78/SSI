import numpy as np
from zad_5 import pokaz_bitmap as bm
from zad_7 import Convert as c
import time


class HopfieldNN:
    # Konstruktor
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.n = self.height * self.width
        self.matrix = np.zeros((self.width, self.height))
        self.weights = np.zeros((self.n, self.n))


###########################################################
    # "malowanie" obrazu do korekcji
    def set_matrix(self, array):
        if self.matrix.shape[0] == array.shape[0] & self.matrix.shape[1] == array.shape[1]:
            self.matrix = array
        else:
            print("Nie można przepisać bitmapy")

#############################################################
    # uczenie
    def train_image(self, schemes):
        convertedSchemes = np.array(schemes)
        convertedSchemes = np.where(schemes <= 0, -1, 1)

        #zmiana na tablice jednowymiarową
        convertedSchemesV = np.array([c.changeArray(np.array(scheme), self.n) for scheme in convertedSchemes])
        # korekcja wag
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    self.weights[i, j] = 0
                else:
                    for k in range(convertedSchemesV.shape[0]):
                        self.weights[i, j] += convertedSchemesV[k, i] * convertedSchemesV[k, j] / self.n

###############################################################
    # korekcja obrazu
    def recognize_image(self, schemes):
        schemeNrRow = self.matrix.shape[0] #5
        schemeNrCol = self.matrix.shape[1] #5
        #print(self.matrix)
        image = np.where(np.array(self.matrix) <= 0, -1, 1)
        #print(image)
        image = image.reshape(self.n)
        #oiprint(image)
        imageTmp = np.zeros(self.n)

        # kroki naprawiajace obraz
        try:

            for i in range(self.n):
                for j in range(self.n):
                    imageTmp[i] += image[j] * self.weights[i, j]
                #time.sleep(1)
                #print(imageTmp)
            image = np.where(imageTmp < 0, -1, 1)

            image = np.where(image == -1, 0, 1).reshape(schemeNrRow, schemeNrCol)

            self.matrix = image

        except:
            # print jezeli obraz nie zostal zmieniony
           print("Błąd przy korygowaniu obrazu")

        finally:
            return self.matrix

###################################################################
    # pokazanie bitmapy
    def show_matrix(self):
        bm.show_bm(self.matrix)

#################################################################3
    # destruktor
    def __del__(self):
        print('Instancja usunięta')