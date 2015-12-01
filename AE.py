import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
import Plots
srng = RandomStreams()

f = open("costs.txt", 'w')
f.write("Starting...\n")
f.close()

def write(str):
    f = open("costs.txt", 'a')
    f.write(str)
    f.write("\n")
    f.close()
    
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, Ws, p_drop_input, p_drop_hidden):
    out = dropout(X, p_drop_input)

    for w in Ws:
        out = rectify(T.dot(out, w))
    return out

# layers should be of the form
# (input size, n_hidden1, n_hidden_2, ..., input_size)
def get_params(layers):
    params = []
    for i in range(len(layers) - 1):
        n_in = layers[i]
        n_out = layers[i+1]
        w = init_weights((n_in, n_out))
        params.append(w)
    return params

def plot_all_filters(Ws, idx):
    #for w in Ws:
    print(len(Ws))
    for i in range(int(len(Ws)/2)):
        w = Ws[i].get_value()
        dim = np.sqrt(w.shape[0])
        w = np.swapaxes(w, 0, 1)
        w = w.reshape(w.shape[0], 1, dim, dim)
        print(dim)
        print(w.shape)
        Plots.plot_filters(w, 1, idx, "layer" + str(i+1))
    return

def plotter(samples, predictions, Ws, img_x, idx):
    plot_all_filters(Ws, idx)
    shp = (samples.shape[0], 1, img_x, img_x)
    samples = samples.reshape(shp)
    predictions = predictions.reshape(shp)
    Plots.plot_predictions_grid(samples, predictions, i, shp)
    return

trX, trY, teX, teY, channels, img_x = mnist(onehot=True)
trX = trX.reshape(trX.shape[0], 784)
teX = teX.reshape(teX.shape[0], 784)
X = T.fmatrix()

layers = [784, 100, 10, 100, 784]
Ws = get_params(layers)

noise_out = model(X, Ws, 0.2, 0.5)
clean_out = model(X, Ws, 0., 0.)

noise_L = T.sum((X - noise_out)**2, axis=1)
noise_cost = noise_L.mean()

clean_L = T.sum((X - clean_out)**2, axis=1)
clean_cost = clean_L.mean()


updates = RMSprop(clean_cost, Ws, lr=0.001)

train = theano.function(inputs=[X], outputs=noise_cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=clean_cost, allow_input_downcast=True)
reconstruct = theano.function(inputs=[X], outputs=clean_out, allow_input_downcast=True)
for i in range(10000):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end])
    #print(predict(teX))
    c = predict(teX)
    print(c)
    write(str(i) + ": " + str(c))
    r = reconstruct(teX[:10, :])
    plotter(teX[:10, :], r, Ws, img_x, i)
