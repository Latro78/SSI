import numpy as np
from zad_5 import pokaz_bitmap as bm
from zad_7 import Convert as c


class HopfieldNN:
    # Konstruktor
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.n = self.height * self.width
        self.matrix = np.zeros((self.width, self.height))
        self.weights = np.zeros((self.n, self.n))



    # "malowanie" obrazu do korekcji
    def set_matrix(self, array):
        if self.matrix.shape[0] == array.shape[0] & self.matrix.shape[1] == array.shape[1]:
            self.matrix = array
        else:
            print("Nie można przepisać bitmapy")

    # uczenie
    def train_image(self, schemes):
        convertedSchemes = np.array(schemes)
        for s in range(schemes.shape[0]):
            convertedSchemes[s] = np.where(np.array(schemes) <= 0, -1, 1)
            convertedSchemes[s] = convertedSchemes[s].reshape(self.n)
        # korekcja wag
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    self.weights[i, j] = 0
                else:
                    for k in range(convertedSchemes.shape[0]):
                        self.weights[i, j] += (convertedSchemes[k, i] * convertedSchemes[k, j]) / self.n

    # korekcja obrazu
    def recognize_image(self, schemes):
        schemeNrRow = self.matrix.shape[0]
        schemeNrCol = self.matrix.shape[1]
        image = np.where(np.array(self.matrix) <= 0, -1, 1)
        image = image.reshape(self.n)
        imageTmp = np.zeros(self.n)

        # kroki naprawiajace obraz
        try:
            for i in range(self.n):
                for j in range(self.n):
                    imageTmp[i] += image[i] * self.weights[i, j]
            imageTmp = np.where(np.array(self.matrix) < 0, 0, 1).reshape(schemeNrRow, schemeNrCol)
            self.matrix = imageTmp

        except:
            # print jezeli obraz nie zostal zmieniony
           print("Błąd przy korygowaniu obrazu")

        finally:
            return image
    # pokazanie bitmapy
    def show_matrix(self):
        bm.show_bm(self.matrix)
    # destruktor
    def __del__(self):
        print('Instancja usunięta')