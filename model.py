import parl
from parl import layers


class ActorModel(parl.Model):
    def __init__(self, act_dim):
       self.fc1 = layers.fc(size=64, act="tanh")
       self.fc2 = layers.fc(size=64, act="tanh")
       self.fc3 = layers.fc(size=act_dim, act="tanh")

    def policy(self, obs):
        hd1 = self.fc1(obs)
        hd2 = self.fc2(hd1)
        logits = self.fc3(hd2)
        return logits


class CriticModel(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(size=64, act="tanh")
        self.fc2 = layers.fc(size=64, act="tanh")
        self.fc3 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        # 输入 state, action, 输出对应的Q(s,a)

        concat = layers.concat([obs, act], axis=1)
        hd1 = self.fc1(concat)
        hd2 = self.fc2(hd1)
        Q = self.fc3(hd2)
        Q = layers.squeeze(Q, axes=[1])
        return Q


class QuadrotorModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()
