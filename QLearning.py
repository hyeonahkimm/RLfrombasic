import numpy as np
import random

SX = 0
SY = 0

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


class Gridworld():  # 5 x 7 with obstacles at (0, 2), (1, 2), (2, 2), (2, 4), (3, 4), (4, 4)
    def __init__(self):
        self.x = SX
        self.y = SY

    def step(self, a):
        if a == LEFT:
            self.move_left()
        elif a == UP:
            self.move_up()
        elif a == RIGHT:
            self.move_right()
        elif a == DOWN:
            self.move_down()

        reward = -1
        done = self.is_done()
        return (self.x, self.y), reward, done

    def move_left(self):
        if self.y == 0:
            pass  # nothing to process
        elif self.y == 3 and self.x in [0, 1, 2]:
            pass
        elif self.y == 5 and self.x in [2, 3, 4]:
            pass
        else:
            self.y -= 1  # 장애물 없을때만 왼쪽으로 한칸 이동 가능

    def move_right(self):
        if self.y == 6:
            pass  # nothing to process
        elif self.y == 1 and self.x in [0, 1, 2]:
            pass
        elif self.y == 3 and self.x in [2, 3, 4]:
            pass
        else:
            self.y += 1

    def move_up(self):
        if self.x == 0:
            pass
        elif self.x == 3 and self.y == 2:
            pass
        else:
            self.x -= 1

    def move_down(self):
        if self.x == 4:
            pass
        elif self.x == 1 and self.y == 4:
            pass
        else:
            self.x += 1

    def is_done(self):
        if self.x == 4 and self.y == 6:
            return True
        else:
            return False

    def reset(self):
        self.x = SX
        self.y = SY
        return self.x, self.y


class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4))
        self.eps = 0.9
        self.alpha = 0.1

    def select_action(self, s):
        # eps-greedy (prob with eps: random, 1-eps: greedy)
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0, 3)
        else:
            action = np.argmax(self.q_table[x, y, :])
        return action

    def update_table(self, transition):
        # episode -> q update
        s, a, r, s_prime = transition
        x, y = s
        x_prime, y_prime = s_prime
        self.q_table[x, y, a] = self.q_table[x, y, a] + self.alpha * (r + np.amax(self.q_table[x_prime, y_prime, :]) - self.q_table[x, y, a])

    def anneal_eps(self):  # decay eps
        self.eps -= -0.01
        self.eps = max(0.2, self.eps)

    def show_table(self):
        q_lst = self.q_table.tolist()  # 왜 굳이?
        data = np.zeros((5, 7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]  # action만큼
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)


def main():
    env = Gridworld()
    agent = QAgent()

    for n_epi in range(1000):
        done = False

        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s, a, r, s_prime))
            s = s_prime
        agent.anneal_eps()

    agent.show_table()


if __name__ == '__main__':
    main()
