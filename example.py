import neuralfit as nf
import numpy as np

# Define dataset
x = np.asarray([[0,0], [0,1], [1,0], [1,1]])
y = np.asarray([[0], [1], [1], [0]])

# Define model (2 inputs, 1 output)
model = nf.Model(2, 1)

# Compile and evolve
model.compile('alpha', loss='mse', monitors=['size'])
model.evolve(x,y)

# Make predictions
print(model.predict(x))

# Save model
model.save('model.nf')

# Export model to Keras
keras_model = model.to_keras()
