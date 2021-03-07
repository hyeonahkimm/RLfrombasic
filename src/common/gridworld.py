import io
import numpy as np
import sys
from gym.envs.toy_test import discrete
from copy import deepcopy as dc

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape = [4, 4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape) #np.prod : array 내부 element들의 곱
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape) # np.arange(x) 0부터 x까지 [0, 1, ..., x]
        it = np.nditer(grid, flags=['multi_index']) # iterator

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index # 왜 y,x 순서? => (row, column)가 (y, x) 에 대응

            P[s] = {a: [] for a in range(nA)}  # a = 0, ..,3 돌면서 [] 생성 (s = iterindex 이고 state)
            # P[s][a] = (prob, next_state, reward, is_done)

            def is_done(s): # terminal or not
                return s == 0 or s == (nS - 1)
            reward = 0.0 if is_done(s) else -1.0 # reward는 현재 state와 action 기준 (여기서는 action 종류 관계없이 동일)

            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)] # 왜 [ ]?
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if y == 0 else s - MAX_X # 맨 윗줄이면 그대로, 아니면 MAX_X 만큼 빼기 (s는 1차원 배열이니까)
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y -1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()