import torch
from sklearn.metrics.pairwise import cosine_similarity


class PE_GAME:
    def __init__(self, d_model, max_len):
        super(PE_GAME, self).__init__()
        self.state = torch.zeros(max_len, d_model)
        self.state_size = int(d_model * max_len)
        self.action_size = self.state_size
        self.max_len = max_len
        self.d_model = d_model

    def reset(self):
        self.state = torch.zeros(self.max_len, self.d_model)
        return self.state

    def step(self, action):
        action = action.view(self.max_len, self.d_model)
        state = self.state
        self.state = self.state + action
        # tensor안의 값을 -1 ~ 1로 제한
        self.state = torch.clamp(self.state, min=-1.0, max=1.0)
        next_state = self.state

        reward_list = []
        # total = []
        for i in range(self.max_len):
            for j in range(self.max_len - i):
                x = j + i
                """
                y는 이상적인 정답함수
                """
                # print(i, x)
                # tmp = []
                # tmp.append([i, x])

                y = ((-1 - (+1)) / (self.max_len - 1)) * (x - i) + 1
                # print('y')
                # print(y)

                vector1 = action[i].tolist()
                vector2 = action[x].tolist()
                similarity = cosine_similarity([vector1], [vector2])

                reward = -((y - similarity) ** 2)
                reward_list.append(reward)
        reward = sum(reward_list)[0][0]
        # print(reward_list)
        done = False
        return next_state, reward, done


# d_model = 3
# max_len = 4
# action = torch.tensor([[-1.1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
# # action = torch.tensor([[1, 2, 3, 1, 2, 2, 7, 8, 9, 10, 11, 12]])

# game = PE_GAME(d_model, max_len)
# game.reset()
# step = game.step(action)
# print(step)
