#config:utf-8

import numpy as np
import pickle
import gzip
from PIL import Image
import pylab
import os

def sigmoid(x, beta=1.0):
    return 1.0 / (1.0 + np.exp(beta * -x))

def sigmoid_deriv(u):
    return (u * (1 - u))

def scale(x):
    eps = 1e-8
    x = x.copy()
    x -= x.min()
    x *= 1.0 / (x.max() + eps)
    return 255.0*x

def add_bias(x, axis=None):
    return np.insert(x, 0, 1, axis=axis)

def enc(X, w, b):
    data = []
    for i in range(int(len(X))):
        data.append(sigmoid(np.dot(w.T, X[i]) + b))
    return data

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding = 'latin1')
    return dict


class AutoEncoder(object):
    def __init__(self, n_visible_units, n_hidden_units, noise):
        #create merusenne twister
        self.rng = np.random.RandomState(1)
        #initial weight scope
        r = np.sqrt(6. / (n_hidden_units + n_visible_units + 1))
        #encode weight setting
        self.enc_w = np.array(self.rng.uniform(-r, r, (n_visible_units, n_hidden_units)))
        #bias setting
        self.enc_b = np.zeros(n_hidden_units)
        self.dec_b = np.zeros(n_visible_units)
        #initial value setting
        self.n_visible = n_visible_units
        self.n_hidden = n_hidden_units
        self.noise = noise
        print('hoge')

    def encode(self, x):
        #evaluate y
        return sigmoid(np.dot(self.enc_w.T, x) + self.enc_b)

    def decode(self, y):
        #evaluate z
        return sigmoid(np.dot(self.enc_w, y) + self.dec_b)

    def corrupt(self, x, noise):
        #make noise
        return self.rng.binomial(size = x.shape, n = 1, p = 1.0 - noise) * x

    def get_cost(self, x, z):
        eps = 1e-10
        return - np.sum((x * np.log(z + eps) + (1.-x) * np.log(1.-z + eps)))

    def partial_differentiation(self, x_batch):

        #cost:diff enc,dec. initial grad
        cost = 0.
        grad_enc_w = np.zeros(self.enc_w.shape)
        grad_enc_b = np.zeros(self.enc_b.shape)
        grad_dec_b = np.zeros(self.dec_b.shape)

        for x in x_batch:
            #add noise in data.
            tilde_x = self.corrupt(x, self.noise)
            #encode
            y = self.encode(tilde_x)
            #decode
            z = self.decode(y)

            #get_cost:L = -Lh = xlogz + (1-x)log(1-z)
            cost += self.get_cost(x,z)

            alpha_h2 = x - z
            alpha_h1 = np.dot(self.enc_w.T, alpha_h2) * sigmoid_deriv(y)

            grad_enc_b = alpha_h1
            grad_dec_b = alpha_h2

            alpha_h1 = np.atleast_2d(alpha_h1)
            tilde_x = np.atleast_2d(tilde_x)
            alpha_h2 = np.atleast_2d(alpha_h2)
            y = np.atleast_2d(y)

            grad_enc_w = (np.dot(alpha_h1.T, tilde_x) + (np.dot(alpha_h2.T, y)).T).T

        #Normalization
        cost /= len(x_batch)
        grad_enc_w /= len(x_batch)
        grad_enc_b /= len(x_batch)
        grad_dec_b /= len(x_batch)

        return cost, grad_enc_w, grad_enc_b, grad_dec_b

    #learning_rate, epochs:repeat learning count
    def sgd_train(self, X, learning_rate=0.2, epochs=10, batch_size = 20):
        #minibatch algorizhm
        #partition length learning_data
        batch_num = len(X) / batch_size
        batch_num = int(batch_num)

        #online
        for epoch in range(epochs):
            total_cost = 0.0 #sum error(gosa)
            #batch
            for i in range(batch_num):
                batch = X[i*batch_size : (i+1)*batch_size] #slice

                cost, gradEnc_w, gradEnc_b, gradDec_b = \
                    self.partial_differentiation(batch) #partial differentiation

                #update weight and bias
                total_cost += cost
                self.enc_w += learning_rate * gradEnc_w
                self.enc_b += learning_rate * gradEnc_b
                self.dec_b += learning_rate * gradDec_b

            print(epoch)
            print((1. / batch_num) * total_cost)

    def conpare_image(self, p, inputs, targets):
        rt = int(np.sqrt(inputs[0].size))
        for i in range(p.size):
            tilde_x = self.corrupt(inputs[p[i]], self.noise)
            y = self.encode(tilde_x)
            z = self.decode(y)
            pylab.subplot(2, p.size, i + 1)
            pylab.axis('off')
            pylab.imshow(tilde_x.reshape(rt, rt), cmap=pylab.cm.gray_r, interpolation='nearest')
            pylab.title('nimg %i' % targets[p[i]])
            pylab.subplot(2, p.size, i + 6)
            pylab.axis('off')
            pylab.imshow(z.reshape(rt, rt), cmap=pylab.cm.gray_r, interpolation='nearest')
            pylab.title('dimg %i' % targets[p[i]])
        pylab.show()
        pylab.savefig('output.png')

    def fix_parameters(self, X):
        data = []
        for i in range(int(len(X))):
            tilde_x = self.corrupt(X[i], self.noise)
            data.append(self.encode(tilde_x))
        return data

    def output(self, X):
        data = []
        for i in range(int(len(X))):
            data.append(self.decode(X[i]))
        return data

    def dump_weights(self, file_name):
        self.enc_w.dump(file_name + "_w.dmp")
        self.enc_b.dump(file_name + "_b.dmp")

    def display(self):
        tile_size = (int(np.sqrt(self.enc_w[0].size)), int(np.sqrt(self.enc_w[0].size)))
        panel_shape = (10, 10)
        margin_y = np.zeros(tile_size[1])
        margin_x = np.zeros((tile_size[0] + 1) * panel_shape[0])
        image = margin_x.copy()

        for y in range(panel_shape[1]):
            tmp = np.hstack( [ np.c_[ scale( x.reshape(tile_size) ), margin_y ]
                for x in self.enc_w[y*panel_shape[0]:(y+1)*panel_shape[0]]])
            tmp = np.vstack([tmp, margin_x])

            image = np.vstack([image, tmp])

        img = Image.fromarray(image)
        img = img.convert('RGB')
        img.show()

        #return utils.visualize_weights(self.enc_w, panel_shape, tile_size)

        #panel_shape = (int(np.sqrt(self.enc_w.shape[0])), int(np.sqrt(self.enc_w.shape[0])))
        #return utils.visualize_weights(self.enc_w, panel_shape, tile_size)

#Multi-Layer Perceptron(多層パーセプトロン)
class MLP(object):
    def __init__(self, n_input_units, n_hidden_units, n_output_units):
        self.nin = n_input_units
        self.nhid = n_hidden_units
        self.nout = n_output_units

        #入力→中間層の重みの初期値代入
        self.v = np.random.uniform(-1.0, 1.0, (self.nhid, self.nin+1))
        #中間→出力層の重みの初期値代入
        self.w = np.random.uniform(-1.0, 1.0, (self.nout, self.nhid+1))

    #leaning_rateは学習係数, 最急降下法で用いる．epochsは学習の回数
    def fit(self, inputs, targets, learning_rate=0.2, epochs=200000):
        inputs = add_bias(inputs, axis=1)

        for loop_cnt in range(epochs):
            p = np.random.randint(inputs.shape[0]) #0~50000までの値のどれかを返す
            ip = inputs[p] #inputs_pattern. パターンpの入力信号
            tp = np.zeros(10)
            tp[targets[p]] = 1 #teach_pattern. パターンpの教師信号

            #入力した値を出力するまでの処理
            oj = sigmoid(np.dot(self.v, ip)) #j列のニューロンを求める処理
            oj = add_bias(oj) #j列のニューロンに仮想ニューロンを追加
            ok = sigmoid(np.dot(self.w, oj)) #k列のニューロンを求める処理

            #出力と教師信号の差から重みの修正を行う(back propagation)
            #デルタkを求める処理
            delta_k = sigmoid_deriv(ok)*(ok - tp)
            #デルタjを求める処理
            delta_j = sigmoid_deriv(oj) * np.dot(self.w.T, delta_k)

            oj = np.atleast_2d(oj)
            delta_k = np.atleast_2d(delta_k)
            self.w = self.w - learning_rate * np.dot(delta_k.T, oj) #最急降下法

            ip = np.atleast_2d(ip)
            delta_j = np.atleast_2d(delta_j)
            self.v = self.v - learning_rate * np.dot(delta_j.T, ip)[1:, :] #最急降下法

            if(loop_cnt%10000 == 0):
                print(targets[p])
                print(tp)
                print(tp.shape)
        self.v.dump("jweight.dmp")
        self.w.dump("kweight.dmp")

    def experiment(self, inputs, test_data, targets):
        jw = np.load("jweight.dmp")
        kw = np.load("kweight.dmp")
        inputs = add_bias(inputs, axis=1)
        count = 0
        ms = np.zeros(10)
        for i,t in zip(inputs, range(targets.size)):
            oj = sigmoid(np.dot(jw, i))
            oj = add_bias(oj)
            ok = sigmoid(np.dot(kw, oj))
            if(np.argmax(ok) == targets[t]):
                count += 1
            else:
                ms[targets[t]] += 1
                pylab.title('miss %i' % np.argmax(ok))
                pylab.imshow(test_data[t].reshape(28, 28), cmap=pylab.cm.gray_r)
                pylab.show()
        print("Correct Answer Rate:%s, %d" % (count/targets.size, count))
        print("Miss Rate %s" % (ms/targets.size))

if __name__ == '__main__':
    #load_data
    x_train = None
    y_train = []
    for i in range(1, 6):
        data_dictionary = unpickle("cifar-10-batches-py/data_batch_%d" % i)
        if x_train == None:
            x_train = data_dictionary['data']
        else:
            x_train = np.vstack((x_train, data_dictionary['data']))
        y_train = y_train + data_dictionary['labels']
    y_train = np.array(y_train)
    x_train = x_train.reshape((len(x_train), 3, 1024))
    test_data_dictionary = unpickle("cifar-10-batches-py/test_batch")
    x_test = test_data_dictionary['data']
    x_test = x_test.reshape(len(x_test), 3, 1024)
    y_test = np.array(test_data_dictionary['labels'])

    if(os.path.exists("last_w.dmp")):
        print("file Exists")
    else:
        #Pre-training
        zero = AutoEncoder(n_visible_units=1024, n_hidden_units=784, noise=0.1)
        zero.sgd_train(x_train)
        data = zero.fix_parameters(x_train)
        zero.dump_weights(file_name="zero")
        #first_train
        first = AutoEncoder(n_visible_units=784, n_hidden_units=256, noise=0.1)
        first.sgd_train(x_train)
        data = first.fix_parameters(data)
        #vl_data = first.fix_parameters(valid_data[0])
        first.dump_weights(file_name="first")
        #second_train
        second = AutoEncoder(n_visible_units=256, n_hidden_units=100, noise=0.2)
        second.sgd_train(data)
        data = second.fix_parameters(data)
        #vl_data = second.fix_parameters(vl_data)
        second.dump_weights(file_name="second")
        #third_train
        third = AutoEncoder(n_visible_units=100, n_hidden_units=49, noise=0.3)
        third.sgd_train(data)
        data = third.fix_parameters(data)
        #vl_data = third.fix_parameters(vl_data)
        third.dump_weights(file_name="third")
        #last_train
        last = AutoEncoder(n_visible_units=49, n_hidden_units=25, noise=0.3)
        last.sgd_train(data)
        data = last.fix_parameters(data)
        last.dump_weights(file_name="last")
        #p = np.random.random_integers(0, len(valid_data[0]), 5)
        #first.conpare_image(p, np.array(valid_data[0]), np.array(valid_data[1]))

    #fine tuning
    w0 = np.load("zero_w.dmp")
    b0 = np.load("zero_b.dmp")
    w1 = np.load("first_w.dmp")
    b1 = np.load("first_b.dmp")
    w2 = np.load("second_w.dmp")
    b2 = np.load("second_b.dmp")
    w3 = np.load("third_w.dmp")
    b3 = np.load("third_b.dmp")
    w4 = np.load("last_w.dmp")
    b4 = np.load("last_b.dmp")
    inputs = enc(x_train, w0, b0)
    inputs = enc(inputs, w1, b1)
    inputs = enc(inputs, w2, b2)
    inputs = enc(inputs, w3, b3)
    inputs = enc(inputs, w4, b4)
    mlp = MLP(n_input_units=25, n_hidden_units=40, n_output_units=10)
    if(os.path.exists("jweight.dmp")):
        print("file Exists")
    else:
        mlp.fit(inputs, y_train)

    #test
    inputs = enc(x_test, w1, b1)
    inputs = enc(inputs, w2, b2)
    inputs = enc(inputs, w3, b3)
    inputs = enc(inputs, w4, b4)
    mlp.experiment(inputs, x_test, y_test)
