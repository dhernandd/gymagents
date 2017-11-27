# Little implementations following Juliani's blog posts

# --- SIMPLE Q-LEARNING FOR THE FROZEN LAKE ENVIRONMENT --- #

import gym
import numpy as np
import random
import tensorflow as tf

from agent import Agent #@UnresolvedImport 

env = gym.make('FrozenLake-v0')


class QLAgent(Agent):
    """
    """
    def __init__(self, learning_rate=0.8, discount=0.95):
        """
        """
        self.SA_values = np.zeros([env.observation_space.n, env.action_space.n])
 
        # Set learning parameters
        self.lr = learning_rate
        self.discount = discount


    def act(self, S, num_ep):
        """
        This policy consists of adding Gaussian noise to the entries of the
        state-action value table corresponding to the current state, and then
        picking the entry with the largest value.
        
        The standard deviation of the Gaussian noise is reduced as more episodes
        are considered.
        """
        return np.argmax(self.SA_values[S, :] + 
                         np.random.randn(1, env.action_space.n)*(1./(num_ep+1)) ) 
    
    def update_SA_table(self, S, A, Snew, R):
        """
        """
        self.SA_values[S, A] = ( self.SA_values[S, A] + 
                                 self.lr*(R + self.discount*np.max(self.SA_values[Snew, :]) 
                                          - self.SA_values[S, A]) )
            
    def train(self, num_episodes):
        """
        """
        total_reward_list = []
        for _ in range(num_episodes):
            
            # Get initial state
            S = env.reset()
            total_reward = 0
            k = 0

            while k < 100:
                k += 1
                A = self.act(S, num_episodes)
                Snew, R, d, _ = env.step(A)
                self.update_SA_table(S, A, Snew, R)
                total_reward += R
                S = Snew
                if d == True:
                    break
            total_reward_list.append(total_reward)
        
        print 'The total rewards were:', total_reward_list[::20]


class QNetwork():
    """
    """
    def __init__(self, discount=0.9, epsilon=0.1):
        """
        """
        self.discount = discount
        self.epsilon=epsilon
        
        tf.reset_default_graph()
        # Define the inputs to the Tensorflow Graph
        self.inputs = tf.placeholder(shape=[1,16], dtype=tf.float32)
        self.nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
        
        # Define the relevant ops
        self.W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
        # The value network simply maps each state to a vector of action values.
        self.Qout = tf.matmul(self.inputs, self.W) 
        self.predict = tf.argmax(self.Qout, 1)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        
    
    def act(self, sess, S):
        """
        """
        return sess.run(self.predict, feed_dict={self.inputs : np.identity(16)[S:S+1]})[0]

    
    def train(self, num_episodes=2000):
        """
        """
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        updateModel = trainer.minimize(self.loss)
        
        with tf.Session() as sess:
#             sess.run(init)
            sess.run(tf.global_variables_initializer())
            jList = []
            rList = []
            for i in range(num_episodes):

                S = env.reset()
                totalR = 0
                d = False
                j = 0
                while j < 99:
                    j += 1
                    
                    # Compute the action A for the current state S. This follows
                    # an epsilon-greedy policy to ensure exploration. 
                    A = self.act(sess, S)
                    if np.random.rand(1) < self.epsilon:
                        A = env.action_space.sample()
                     
                    # Take action A, register the new state and reward.
                    Snew, R, d, _ = env.step(A)

                    # Use the value network to build a list of the action values
                    # Q(S, A) for all A. Also build such list for the new state
                    # Snew and find the max of Q(Snew, A) over all the As
                    Q_S = sess.run(self.Qout, 
                                        feed_dict={self.inputs : np.identity(16)[S:S+1]})
                    Q_Snew = sess.run(self.Qout, 
                                  feed_dict={self.inputs : np.identity(16)[Snew:Snew+1]})
                    maxQnew = np.max(Q_Snew)
                    
                    # The value of (S, A) should look more like the obtained
                    # reward plus the max discounted Q(Snew, A). This is the
                    # target that the parameters of our model should try to
                    # reproduce.
                    targetQ = Q_S
                    targetQ[0, A] = R + self.discount*maxQnew
                    
                    # So train the model to approach the target.
                    _, _ = sess.run([updateModel, self.W],
                                     feed_dict={self.inputs : np.identity(16)[S:S+1],
                                                self.nextQ : targetQ})
                    totalR += R
                    S = Snew
                    if d == True:
                        # Reduce epsilon so that greedy approaches Bellman
                        # optimal
                        self.epsilon = 1./((i/50) + 10)
                        break
                jList.append(j)
                rList.append(totalR )
        
        return rList

#create lists to contain total rewards and steps per episode

if __name__ == '__main__':
    
#     agent = QLAgent()
    agent = QNetwork()
    
    num_ep = 5000
    rList = agent.train(num_ep)
    print "Percent of succesful episodes: " + str(sum(rList)/num_ep) + "%"
    
    
    
