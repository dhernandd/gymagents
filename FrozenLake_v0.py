# Little implementations following Juliani's blog posts

# --- SIMPLE Q-LEARNING FOR THE FROZEN LAKE ENVIRONMENT --- #

import gym
import numpy as np

from agent import Agent

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


if __name__ == '__main__':
    
    agent = QLAgent()
    agent.train(5000)
    
    
    
