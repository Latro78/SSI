import numpy as np
from zad_7 import schemes as s
from zad_7 import hopfield_nn

obraz = hopfield_nn.HopfieldNN(5,5)
obraz.train_image(np.array(s.schemes))
