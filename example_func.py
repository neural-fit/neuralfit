import neuralfit as nf
import numpy as np

# Define dataset
x = np.asarray([[0,0], [0,1], [1,0], [1,1]])
y = np.asarray([[0], [1], [1], [0]])

# Define evaluation function (MSE)
def evaluate (genomes):
    losses = np.zeros(len(genomes))

    for i in range(len(genomes)):
        for j in range(x.shape[0]):
            result = genomes[i].predict(x[j])
            losses[i] += (result - y[j])**2
    
    return losses/x.shape[0]

# Define model (2 inputs, 1 output)
model = nf.Model(2, 1)

# Compile and evolve
model.compile(monitors=['size'])
model.func_evolve(evaluate)

# Make predictions
print(model.predict(x))

# Save model
model.save('model.nf')

# Export model to Keras
keras_model = model.to_keras()
