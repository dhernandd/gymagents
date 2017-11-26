# Copyright 2017 Daniel Hernandez Diaz, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
"""
Agent objects take actions according to a policy.

In their initialization, agents should be fed at least an action space,
representing all the possible actions the agent can take. Within the gym, the
action space is obtained from the environment in question. So for
instance, for the Cartpole environment, the action space would be [1, -1],
representing pushing the pole with constant force in either direction.

The other defining feature of an agent is the method act. This method takes
information i from the environment - an observation - and implements the policy.
That is, it decides (returns) an action based on the current state.

In order to decide which action to choose, agents may store a set of parameters
{phi}. In that case, the act method is a function a(i; phi). The ultimate goal
is to find optimal phis such that the total reward is maximized. Hence, this
must involve a procedure to update the parameters phi, the LEARNING ALGORITHM,
in the quest for the optimum. Only in the simplest of cases, this procedure
should be coded outside the Agent class.
"""
import os
# This is required in my machine for rendering the Cartpole video. 
os.environ['PATH'] = '/usr/local/bin:' + os.environ['PATH'] 

import logging
import sys

import numpy as np

import gym
from gym import wrappers

from agent import Agent


class RandomAgent(Agent):
    """
    The stupidest agent
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
    
    
class ConstantAgent(Agent):
    """
    A constant agent. Always applies the same action. In truth, maybe even
    stupider than the one above.
    """
    def __init__(self, action_space):
        """
        """
        self.action_space = action_space
        
        
    def act(self, observation, reward, done):
        return 1
    

class FirstAgent(Agent):
    """
    """
    def __init__(self, action_space, cur_pars):
        self.action_space = action_space
        self._cur_pars = cur_pars
        self.best_params = np.zeros([4])
    
    def act(self, observation, test=False):
        """
        Returns either 0 or 1 indicating the action to take in this state.
        """
        if test:
            return sum(self.best_params*np.array(observation)) > 0
        else:
            return sum(self.cur_pars*np.array(observation)) > 0
    
    @property
    def cur_pars(self):
        return self._cur_pars
    
    @cur_pars.setter
    def cur_pars(self, value):
        self._cur_pars = value
        
    def test_best_params(self, env):
        rewards = [0]*10
        for i in range(10):
            ob = env.reset()
            total_reward = 0
            while True:
                action = self.act(ob, test=True)
                ob, reward, done, _ = env.step(action)
                total_reward += 1
                if done:
                    break
            rewards[i] = total_reward
        print rewards
        
        return np.mean(rewards)
    
    def train(self, env):
        episode_count = 10000
        reward = 0
        done = False
        best_total_reward = 0
        best_mean_total_reward = 0
        for i in range(episode_count):
            ob = env.reset()
            total_reward = 0
            
            # Set the agent's current parameters to a set of random numbers. The
            # goal is find the best possible set by randomly sampling from a
            # uniform distribution. The agent is always exploring here in the
            # most trivial fashion
            self.cur_pars = np.random.randn(4)
            while True:
                action = self.act(ob)  
                # the env.step method (more precisely, the env._step method
                # called by env.step) here is overridden by the wrapper that
                # renders the video.
                ob, reward, done, _ = env.step(action) 
                total_reward += 1
                if done:
                    break
            if total_reward > best_total_reward:
                print '(TR, BTR):', total_reward, best_total_reward
                best_total_reward = total_reward
                agent.best_params = agent.cur_pars
                print 'Best mean, best_params:', agent.test_best_params(env), agent.best_params
                # Note there's no env.render() here. But the environment still
                # can open window and render if asked by env.monitor: it calls
                # env.render('rgb_array') to record video. Video is not recorded
                # every episode, see capped_cubic_video_schedule for details.
    
        # Close the env and write monitor result info to disk
        env.close()



class HillClimbingAgent(object):
    """
    """
    def __init__(self, noisescale, params=None):
        """
        """
        self.noisescale = noisescale
        self._cur_pars = np.random.randn(4) if params is None else params

    @property
    def cur_pars(self):
        return self._cur_pars
    
    @cur_pars.setter
    def cur_pars(self, value):
        self._cur_pars = value
    
    
    
if __name__ == '__main__':

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    
#     agent = RandomAgent(env.action_space)
#     agent = ConstantAgent(env.action_space)
    agent = FirstAgent(env.action_space, np.zeros([4]))
    agent.train(env)

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
#     gym.upload(outdir)
