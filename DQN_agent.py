import numpy as np
import tensorflow as tf
import  random
from collections import  deque
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.95
        self.learning_rate = 0.001
        self.loss = None
        self.sess = tf.Session()

    def add_layer(inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    def build_model(self, state):
        X = tf.placeholder(tf.float32, [None, self.state_size], name="X")
        Y = tf.placeholder(tf.float32, [None, self.action_size], name="Y")
        #建立神經元
        l1 = self.add_layer(state, self.state_size, 12, tf.nn.relu)
        l2 = self.add_layer(l1, 12, 12, tf.nn.relu)
        prediction=self.add_layer(l2, 12, self.action_size)
        #預測和真實的誤差
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y - self.build_model(state)), reduction_indices=[1]))
        return prediction

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.build_model(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        #從memory中取出小量資料
        minibatch = random.sample(self.memory, batch_size)
        #從memory中提取訊息
        for state, action, reward, next_state, done in minibatch:
            #如果達成，設定獎勵
            goal = reward
            if not done:
                goal = reward+self.gamma*np.amax(self.build_model(next_state))
            goal_f = self.build_model(state)
            goal_f[0][action] = goal
            self.sess.run(tf.global_variables_initializer())











