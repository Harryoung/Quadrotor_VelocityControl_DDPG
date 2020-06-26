# Quadrotor_VelocityControl_DDPG
Using paddlepaddle to solve the velocity control task of quadrotor by DDPG. The environment is created by RLSchool, and the DDPG is imported from PARL, which is the Reinforce Learning repository developed by PaddlePaddle group of Baidu.

# First Edition

I just used the settings for hovering-control task to fit this velocity-control task. One thing to be noted is that the activation function of Critic should be None. For hidden layers, tanh seems like a better activation than relu in reinforce learning.
The result seems not good. Test reward converged to about -20 after training for 1M steps.

![first result](./fig/1st.gif)

As described by README.md of RLSchool/quadrotor, "Yellow arrow is the expected velocity vector; orange arrow is the real velocity vector." THe result should look like this, the yellow vector and orange vector are as similar as possible. 
But my quadrotor just tends to fall down slowly and the two vectors are not similar at all! 

# Second Edition
There must be something wrong I think. I don't understand how the reward is measured. Maybe some extra work on the features could be done, for example, we can add the difference between next real velocity and next expected velocity to the reward.
It should be noted that the real velocity is given in local (body) coordinate while the next expected velocity is given in global (reference) coordinate. So before add the expected velocity into the input features, i.e. the obs, we should first conduct a coordinate transform.

We can get the rotation matrix by this.
```python
def get_rotation_matrix(yaw, pitch, roll):
    m_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]], dtype="float32")
    m_pitch = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                        [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)]], dtype="float32")
    m_roll = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]], dtype="float32")
    r_matrix = np.matmul(m_roll, np.matmul(m_pitch, m_yaw))

    return r_matrix
```
The by ```np.matmul(r_matrix, velocity_in_global)```, we can get the expected next velocity in local coordinate.

On the first hand, I added the above local expected velocity into the obs, On the other hand, I added the inner product between next velocity and next expected velocity (both in local coordinate) into the reward, as ''v_diff'.
I hope this additional reward could lead the model to fit the controlling of velocity. By the way, the scale is much different between real reward and v_diff, so I multiplied a scale-factor (0.01) to the v_diff.

![second result](./fig/2nd.gif)

But again, my quadrotor just can't learn to fit the expected velocity. ╮(╯﹏╰）╭

# Third Edition
I run to read the source code of RLSchool and try to find the computation of the reward. It turns out to be very simple.
```python
def _get_reward():
    reward = - min(self.dt * self.simulator.power, self.healthy_reward)
    ...
    elif self.task == 'velocity_control':
        task_reward = -0.001 * self._get_velocity_diff(velocity_target)
        reward += task_reward
    ...
    return reward

def _get_velocity_diff(self, velocity_target):
    vt_x, vt_y, vt_z = velocity_target
    diff = abs(vt_x - self.state['b_v_x']) + \
        abs(vt_y - self.state['b_v_y']) + \
        abs(vt_z - self.state['b_v_z'])
    return diff
```
The reward consists of three parts: the energy cost, healthy_reward (const 0 according to the code, something wrong?) and the velocity difference computed by Manhaton Distance.
The scale factor of v_diff they use is 0.001, maybe too small I think.

Another important thing is that the next expected velocity is actually in the local coordinate according to the code!
This is so frustrating, and I think I need to open an issue there. In this edition, I plan to update the v_diff by treating the next expected velocity as in the local coordinate.
Now the code is running...

