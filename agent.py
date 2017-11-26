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

class Agent(object):
    """
    """
    def __init__(self, action_space):
        self.action_space = action_space


    def act(self, observation, reward, done):
        assert False, "Please define me! An agent that cannot \
        act is not agent, more like a pillow"
    
    