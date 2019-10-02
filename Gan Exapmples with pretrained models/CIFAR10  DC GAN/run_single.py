from keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
model = load_model('generator_model_080.h5')
vector = asarray([[0.75 for _ in range(100)]])
X = model.predict(vector)
X = (X + 1) / 2.0
pyplot.imshow(X[0, :, :])
pyplot.show()