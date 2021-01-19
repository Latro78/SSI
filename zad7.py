import numpy as np
from zad_7 import schemes as s
from zad_7 import hopfield_nn
from zad_5 import pokaz_bitmap as bm

obraz = hopfield_nn.HopfieldNN(5,5)
obraz.set_matrix(np.array(
    [[1, 0, 0, 0, 1],
     [0, 1, 1, 1, 0],
     [0, 0, 1, 0, 0],
     [0, 1, 1, 1, 0],
     [1, 0, 0, 0, 1]]
))
obraz.show_matrix()
obraz.train_image(np.array(s.schemes))
obrazPoprawiony = obraz.recognize_image(np.array(s.schemes))
bm.show_bm(obrazPoprawiony)