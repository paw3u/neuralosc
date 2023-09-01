import numpy as np
import matplotlib.pyplot as plt


class nosc_layer:
	def __init__(self, inputs_num, outputs_num):
		super(nosc_layer, self).__init__()
		self.inputs_num = inputs_num
		self.outputs_num = outputs_num
		self.weights = 2.0 * np.random.random((outputs_num, inputs_num)) - 1.0
		self.weights_t = np.zeros(self.weights.shape)
		self.bias = 2.0 * np.random.random((outputs_num)) - 1.0
		self.bias_t = np.zeros(self.bias.shape)
		#self.weights = -0.3
		self.amp = 1.0
		self.omega_w = 100.0 * np.random.random((outputs_num, inputs_num)) + 100.0
		self.omega_b = 100.0 * np.random.random((outputs_num)) + 100.0
		self.phase = 0.0
		print(self.weights)

	def relu(self, x):
		return np.maximum(x, 0)

	def forward(self, inputs):
		return np.dot(self.weights, inputs) + self.bias

	def forward_t(self, inputs, t):
		amp_t = self.amp * (1.0 / (t * t + 1.0))
		self.weights_t = self.weights + amp_t * np.sin(self.omega_w * t + self.phase)
		self.bias_t = self.bias + 0.01 * amp_t * np.sin(self.omega_b * t + self.phase)
		return np.dot(self.weights_t, inputs) + self.bias_t

	def adjust(self, inputs, targets, t, delta, debug=False):
		outputs = self.forward(inputs)
		outputs_t = self.forward_t(inputs, t)
		error = (np.square(targets - outputs)).mean()
		error_t = (np.square(targets - outputs_t)).mean()
		if error_t < error:
			self.weights += delta * (self.weights_t - self.weights)
			self.bias += delta * (self.bias_t - self.bias)
		if debug:
			print(error)
		amp_t = self.amp * (1.0 / (t*t + 1.0))
		return error

class nosc_net:
	def __init__(self, dims):
		super(nosc_net, self).__init__()
		self.dims = dims
		self.layers = []
		for i in range(self.dims.size - 1):
			self.layers += [nosc_layer(dims[i], dims[i+1])]

	def forward(self, inputs):
		x = inputs
		for i in range(self.dims.size - 1):
			x = self.layers[i].forward(x)
		return x

	def forward_t(self, inputs, t):
		x = inputs
		for i in range(self.dims.size - 1):
			x = self.layers[i].forward_t(x, t)
		return x

	def adjust(self, inputs, targets, t, delta, debug=False):
		outputs = self.forward(inputs)
		outputs_t = self.forward_t(inputs, t)
		error = (np.square(targets - outputs)).mean()
		error_t = (np.square(targets - outputs_t)).mean()
		if error_t < error:
			for i in range(self.dims.size - 1):
				self.layers[i].weights += delta * (self.layers[i].weights_t - self.layers[i].weights)
				self.layers[i].bias += delta * (self.layers[i].bias_t - self.layers[i].bias)
		if debug:
			print(error)
		return error


net = nosc_net(np.array([4, 8, 2]))
secs = 4
fps = 1000
t = np.linspace(0, secs, secs*fps)
x0 = np.zeros(t.size)
x1 = np.zeros(t.size)
inputs = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
targets = np.array([[1.0, 0.0], [0.0, 1.0]])
delta = 0.1
num = 0

for i in range(t.size):
	switch = (i % 200 == 0)
	if switch:
		num ^= 1
	if num == 0:
		x0[i] = net.adjust(inputs[num], targets[num], t[i], delta, i % 200 == 0)
	else:
		x1[i] = net.adjust(inputs[num], targets[num], t[i], delta, i % 200 == 1)

print(inputs[0], net.forward(inputs[0]))
print(inputs[1], net.forward(inputs[1]))

plt.plot(t,x0)
plt.plot(t,x1)
plt.show()

exit()


layer = nosc_layer(4,2)
secs = 4
fps = 1000
t = np.linspace(0, secs, secs*fps)
x0 = np.zeros(t.size)
x1 = np.zeros(t.size)
inputs = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
targets = np.array([[1.0, 0.0], [0.0, 1.0]])
delta = 0.1
num = 0

for i in range(t.size):
	switch = (i % 20 == 0)
	if switch:
		num ^= 1
	if num == 0:
		x0[i] = layer.adjust(inputs[num], targets[num], t[i], delta, i % 200 == 0)
	else:
		x1[i] = layer.adjust(inputs[num], targets[num], t[i], delta, i % 200 == 1)

print(inputs[0], layer.forward(inputs[0]))
print(inputs[1], layer.forward(inputs[1]))

plt.plot(t,x0)
plt.plot(t,x1)
plt.show()

'''
siec: polaczenie kazdy z kazdym? -> kazde polaczenie ze zdefiniowana latencja -> zdeterminowana kolejnosc obliczen
'''
