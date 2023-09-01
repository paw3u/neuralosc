import numpy as np
import matplotlib.pyplot as plt

class nosc_layer(object):

	def __init__(self, inputs_num, outputs_num):
		super(nosc_layer, self).__init__()
		self.inputs_num = inputs_num
		self.outputs_num = outputs_num
		self.weights = 2.0 * np.random.random((outputs_num, inputs_num)) - 1.0
		self.bias = 2.0 * np.random.random((outputs_num)) - 1.0
		#self.weights = -0.3
		self.amp = 1.0
		self.omega_w = 100.0 * np.random.random((outputs_num, inputs_num)) + 100.0
		self.omega_b = 100.0 * np.random.random((outputs_num)) + 100.0
		self.phase = 0.0
		self.err_acc = 0.0
		print(self.weights)

	def relu(self, x):
		return np.maximum(x, 0)

	def forward(self, inputs, t):
		return np.dot(self.weights, inputs) + self.bias

	def adjust(self, inputs, targets, t, delta, debug=False):
		amp_t = self.amp * (-1.0 + 1.0 / (t + 1.0))
		weights_t = self.weights + amp_t * np.sin(self.omega_w * t + self.phase)
		bias_t = self.bias + 0.01 * amp_t * np.sin(self.omega_b * t + self.phase)

		outputs = np.dot(self.weights, inputs) + self.bias
		outputs_t = np.dot(weights_t, inputs) + bias_t
		error = (np.square(targets - outputs)).mean()
		error_t = (np.square(targets - outputs_t)).mean()

		if error_t < error:
			self.weights += delta * (weights_t - self.weights)
			self.bias += delta * (bias_t - self.bias)

		if debug:
			print(error)
		return error
		

layer = nosc_layer(3,2)
secs = 4
fps = 1000
t = np.linspace(0, secs, secs*fps)
x0 = np.zeros(t.size)
x1 = np.zeros(t.size)
inputs = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
targets = np.array([[1.0, 0.0], [0.0, 1.0]])

delta = 0.1
num = 0
for i in range(t.size):
	switch = (i % 200 == 0)
	if switch:
		num ^= 1
	if num == 0:
		x0[i] = layer.adjust(inputs[num], targets[num], t[i], delta, i % 200 == 0)
	else:
		x1[i] = layer.adjust(inputs[num], targets[num], t[i], delta, i % 200 == 1)

print(inputs[0], layer.forward(inputs[0], 0))
print(inputs[1], layer.forward(inputs[1], 0))

plt.plot(t,x0)
plt.plot(t,x1)
plt.show()
