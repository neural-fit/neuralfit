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

# Define evaluation function (MSE)
def evaluate (genomes):
    losses = np.zeros(len(genomes))

    for i in range(len(genomes)):
        for j in range(x.shape[0]):
            genomes[i].clear()
            result = genomes[i].predict(x[j])[-1]
            losses[i] += (result - y[j])**2
    
    return losses/x.shape[0]

# Define model (1 input, 1 output)
model = nf.Model(1, 1, recurrent=True)

# Compile and evolve
model.compile(monitors=['size'])
model.func_evolve(evaluate, epochs=500)

# Make predictions
print(model.predict(x)[:,-1,0])

# Save model
model.save('model.nf')
