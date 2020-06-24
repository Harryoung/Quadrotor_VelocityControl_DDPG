# Quadrotor_VelocityControl_DDPG
Using paddlepaddle to solve the velocity control task of quadrotor by DDPG

# First Edition

I just used the settings for hovering-control task to fit this velocity-control task. One thing to be noted is that the activation function of Critic should be None. For hidden layers, tanh seems like a better activation than relu in reinforce learning.
The result seems not good. Test reward converged to about -20 after traing for 1M steps. The quadrotor tends to fall down slowly and I don't understand how the reward is measured. Maybe some extra work on the features could be done, for example, we can add the diffence between next real velocity and next expected velocity to the reward. It should be noted that the real velocity is given in local (body) cordinate while the next expected velocity is given in global (reference) cordinate.

