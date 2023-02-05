<p align="center"><img src="https://i.imgur.com/qW1uHNl.png" width=400px></p>

<p align="center"><a href="https://badge.fury.io/py/neuralfit"><img src="https://badge.fury.io/py/neuralfit.svg" alt="PyPI version" height="18"></a><a href="https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha">    <img height=18 src="https://img.shields.io/badge/stability-alpha-33bbff.svg" alt="Alpha"></a></p>

NeuralFit is a simple neuro-evolution library for Python with an API that is similar to the [Keras](https://keras.io/) library. You can train models with evolution instead of backpropagation in just a few lines of codes. You can also export evolved models to Keras so you can use them in your existing environments without changing a thing. For more information, please visit https://neuralfit.net/. <b>🐛 NeuralFit is currently in the alpha phase and being developed rapidly, expect many bugs and crashes.</b>

## Installation
NeuralFit is tested and supported on 64-bit machines running Windows or Ubuntu with Python 3.7-3.10 installed. You can install NeuralFit via `pip` with version 19.3 or higher.

```
pip install neuralfit
```

## Get started
As an example, you can evolve a model to learn the [XOR gate](https://en.wikipedia.org/wiki/XOR_gate) as follows.

```python
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
```

To verify that your model has fitted the dataset correctly, we can check its predictions. 

```python
print(model.predict(x))
```

Done with evolving your model? Simply save the model or export it to Keras!

```python
# Save model
model.save('model.nf')

# Export model to Keras
keras_model = model.to_keras()
```

There is much more to discover about NeuralFit, and new features are being added rapidly. Please visit the official [documentation](https://neuralfit.net/documentation/) for more information or follow one of our [examples](https://neuralfit.net/examples/). 

## Recurrent models
It is also possible to evolve recurrent models, although they can not (yet) be exported to Keras. These models can evolve backwards connections that allow models to have memories, making them ideal for timeseries prediction. The above XOR dataset can be transformed to a timeseries, where each input feature is inputted sequentially instead of all at once.

```python
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
```

After the model has evolved, we can give it timeseries and extract the last output to see what the model predicts.

```python
# Make predictions
print(model.predict(x)[:,-1,0])
```

It is also possible to define a target output series that has the length of each input series, which is useful for stock market prediction for example. Please keep an eye on our [examples page](https://neuralfit.net/examples/), as more recurrent examples will be added soon. 


## Issues?
Found a bug? Want to suggest a feature? Or have a general question about the workings of NeuralFit? Head over to the [issues section](https://github.com/neural-fit/neuralfit/issues) and feel free to create an issue. Note that any questions about licenses or payments should be sent to info@neuralfit.net. 

## Become a supporter
Unfortunately, it is not possible to sustain the research and development for NeuralFit withour your help. Please consider supporting us by buying a [license subscription](https://neuralfit.net/licenses/), which also unlocks various extra features. We promise that all NeuralFit features will be available for everyone if other sources of income allow for it 💚. 

