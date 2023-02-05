import neuralfit as nf
import numpy as np

# Define dataset
x = np.asarray([[0,0], [0,1], [1,0], [1,1]])
y = np.asarray([[0], [1], [1], [0]])

# Reshape to timeseries format
# 4 samples of length 2 with 1 feature as input
# 4 samples of length 1 with 1 feature as output
x = np.reshape(x, (4,2,1))
y = np.reshape(y, (4,1,1)) 

# Define model (1 input, 1 output)
model = nf.Model(1,1, recurrent = True)

# Compile and evolve
model.compile(optimizer='alpha', loss='mse', monitors=['size'])
model.evolve(x, y, epochs=500)

# Make predictions, take last prediction of each series
print(model.predict(x)[:,-1,0])

# Save model
model.save('model.nf')
