from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
import position
import random

tf.compat.v1.enable_v2_behavior()


class SnakeEnv(py_environment.PyEnvironment):
    def __init__(self):
        self.width = 1080
        self.height = 1770
        self.left_edge = [0, 0, 20, 1600]
        self.rigth_edge = [1060, 0, 1080, 1600]
        self.top_edge = [0, 0, 1080, 20]
        self.bot_edge = [0, 1580, 1080, 1600]
        self.still_traveling = True

        a = random.randint(0, 4)
        b = (a + 1) % 4

        self.my_variables = [position.Position(a).return_position()]
        self.his_variables = [position.Position(b).return_position()]
        self.my_counter = 1
        self.his_counter = 1
        self.score = 0
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.float32, name='observation')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        
        self.saved_policy = tf.compat.v2.saved_model.load('model')
        self.policy_state = self.saved_policy.get_initial_state(batch_size=0)
        

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec
    
    def preprocess(self,arr,var):
        last = (0, 0)
        for x in var:
            if x[3] == 0:
                clm = int(x[0] / 30)
                if clm > 35:
                    clm = 35
                last = (int(x[1] / 30), clm)
                for i in range(int(x[1] / 30), int(x[2] / 30)):
                    arr[i][clm] = 1
                    # last = (i, clm)
            if x[3] == 180:
                clm = abs(int(x[0] / 30))
                if clm > 36:
                    clm = 36
                for i in range(int(x[1] / 30), int(x[2] / 30)):
                    arr[abs(i) - 1][clm - 1] = 1
                    last = (abs(int(x[1] / 30)) - 1, clm - 1)
            if x[3] == 90:
                row = abs(int(x[0] / 30))
                if x[1] < -1080:
                    x[1] = -1080
                for i in range(int(x[1] / 30), int(x[2] / 30)):
                    arr[row][abs(i) - 1] = 1
                    last = (row, abs(int(x[1] / 30)) - 1)
            if x[3] == -90:
                row = abs(int(x[0] / 30))
                last = (row - 1, int(x[1] / 30))
                for i in range(int(x[1] / 30), int(x[2] / 30)):
                    arr[row - 1][i] = 1
                    # last = (row - 1, i)
        return last

    def preprocessing(self,var1, var2, counter):
        arr = np.zeros((54, 36), dtype=np.float32)
        self.preprocess(arr,var2)
        last = self.preprocess(arr,var1)

        arr[last[0], last[1]] = 2

        
        right = last[1] + 1
        while right < 35 and arr[last[0], right] == 0:
            right += 1

        left = last[1] - 1
        while left > 0 and arr[last[0], left] == 0 :
            left -= 1

        top = last[0] - 1
        while top > 0 and arr[top, last[1]] == 0:
            top -= 1

        bot = last[0] + 1
        while bot < len(arr) - 1 and arr[bot, last[1]] == 0 :
            bot += 1

        lastposition = var1[counter - 1][3]
        results = [0, 0, 0]

        if lastposition == 90:
            results[0] = abs(top - last[0])
            results[1] = abs(right - last[1])
            results[2] = abs(bot - last[0])
        elif lastposition == -90:
            results[0] = abs(bot - last[0])
            results[1] = abs(left - last[1])
            results[2] = abs(top - last[0])
        elif lastposition == 180:
            results[2] = abs(left - last[1])
            results[1] = abs(bot - last[0])
            results[0] = abs(right - last[1])
        else:
            results[0] = abs(left - last[1])
            results[1] = abs(top - last[0])
            results[2] = abs(right - last[1])


        return np.array(results, dtype=np.float32)

    def load(self):

        observation = tf.constant([self.time_step2], dtype=tf.float32)
        step_type = tf.constant([0], dtype=tf.int32)  
        discount = tf.constant([0.0], dtype=tf.float32)  
        reward = tf.constant([0.0], dtype=tf.float32)  
        
        time_step_wrapper= ts.TimeStep(
            step_type, reward, discount, observation)
        
        policy_step = self.saved_policy.action(time_step_wrapper, self.policy_state)
        self.policy_state = policy_step.state
        self.hisstep(policy_step.action.numpy()[0])
        
    def _reset(self):
        self.still_traveling = True

        a = random.randint(0, 4)
        b = (a + 1) % 4

        self.my_variables = [position.Position(a).return_position()]
        self.his_variables = [position.Position(b).return_position()]
        self.my_counter = 1
        self.his_counter = 1
        self.bump_self = 0
        self.score = 0
        result = self.preprocessing(self.my_variables, self.his_variables, self.my_counter)
        self.time_step2 = self.preprocessing(self.his_variables, self.my_variables, self.his_counter)
        self.load()
        return ts.restart(result)
    
    def hisstep(self, action):

        if not self.still_traveling:
            return self.reset()


        if action == 1:
            # turn right
            lastposition = self.his_variables[self.his_counter - 1][3]

            if lastposition == 90:
                lastposition = 180
            elif lastposition == -90:
                lastposition = 0
            elif lastposition == 180:
                lastposition = -90
            elif lastposition == 0:
                lastposition = 90

            first_rect_postition1 = [0, 0, 0, 0]

            first_rect_postition1[0] = self.his_variables[self.his_counter - 1][1]
            first_rect_postition1[1] = -self.his_variables[self.his_counter - 1][0] - 30
            first_rect_postition1[2] = -self.his_variables[self.his_counter - 1][0] - 30
            first_rect_postition1[3] = lastposition

            self.his_variables.append(first_rect_postition1)
            self.his_counter += 1

        elif action == 2:
            # turn left
            lastposition = self.his_variables[self.his_counter - 1][3]

            if lastposition == 90:
                lastposition = 0
            elif lastposition == -90:
                lastposition = 180
            elif lastposition == 180:
                lastposition = 90
            elif lastposition == 0:
                lastposition = -90

            first_rect_postition = [0, 0, 0, 0]

            first_rect_postition[0] = -self.his_variables[self.his_counter - 1][1] - 30
            first_rect_postition[1] = self.his_variables[self.his_counter - 1][0]
            first_rect_postition[2] = self.his_variables[self.his_counter - 1][0]
            first_rect_postition[3] = lastposition

            self.his_variables.append(first_rect_postition)
            self.his_counter += 1

        self.his_variables[self.his_counter - 1][1] -= 30

        # SETTING THE CHECKER
        lastposition = self.his_variables[self.his_counter - 1][3]
        gauche = self.his_variables[self.his_counter - 1][0]
        haut = self.his_variables[self.his_counter - 1][1]

        if lastposition == 90:
            checker = [-haut - 1, gauche, -haut, gauche + 30]
        elif lastposition == -90:
            checker = [haut, -gauche - 30, haut - 1, -gauche]
        elif lastposition == 180:
            checker = [-gauche - 30, -haut - 1, -gauche, -haut]
        else:
            checker = [gauche, haut, gauche + 30, haut + 1]

        for forloop in range(self.his_counter):
            left = self.his_variables[forloop][0]
            top = self.his_variables[forloop][1]
            bot = self.his_variables[forloop][2]
            lastposition = self.his_variables[forloop][3]

            if lastposition == 90:
                rect1 = [-bot, left, -top, left + 30]
            elif lastposition == -90:
                rect1 = [top, -left - 30, bot, -left]
            elif lastposition == 180:
                rect1 = [-left - 30, -bot, -left, -top]
            else:
                rect1 = [left, top, left + 30, bot]

            if checker[0] < rect1[2] and checker[2] > rect1[0] and checker[1] < rect1[3] and checker[3] > \
                    rect1[1] and forloop < self.his_counter - 1:
                if self.bump_self == 0:
                    self.bump_self = 1
                else:
                    self.still_traveling = False

        for forloop in range(self.my_counter):
            left = self.my_variables[forloop][0]
            top = self.my_variables[forloop][1]
            bot = self.my_variables[forloop][2]
            lastposition = self.my_variables[forloop][3]

            if lastposition == 90:
                rect1 = [-bot, left, -top, left + 30]
            elif lastposition == -90:
                rect1 = [top, -left - 30, bot, -left]
            elif lastposition == 180:
                rect1 = [-left - 30, -bot, -left, -top]
            else:
                rect1 = [left, top, left + 30, bot]

            if checker[0] < rect1[2] and checker[2] > rect1[0] and checker[1] < rect1[3] and checker[3] > \
                    rect1[1] and forloop < self.his_counter - 1:
                self.still_traveling = False

        # CHECK IF BUMP TO EDGES
        left_edge = self.left_edge
        right_edge = self.rigth_edge
        top_edge = self.top_edge
        bot_edge = self.bot_edge

        left_b = checker[0] < left_edge[2] and checker[1] < left_edge[3] and \
                 checker[3] > left_edge[1]
        right_b = checker[2] > right_edge[0] and checker[1] < right_edge[3] and \
                  checker[3] > right_edge[1]
        top_b = checker[0] < top_edge[2] and checker[2] > top_edge[0] and checker[1] < top_edge[3]

        bot_b = checker[0] < bot_edge[2] and checker[2] > bot_edge[0] and checker[3] > \
                bot_edge[1]

        if left_b or right_b or top_b or bot_b:
            self.still_traveling = False

        self.time_step2 = self.preprocessing(self.his_variables, self.my_variables, self.his_counter)

    def _step(self, action):

        if not self.still_traveling:
            return self.reset()

        # while still_traveling:
        self.score += 1

        if action == 1:
            # turn right
            lastposition = self.my_variables[self.my_counter - 1][3]

            if lastposition == 90:
                lastposition = 180
            elif lastposition == -90:
                lastposition = 0
            elif lastposition == 180:
                lastposition = -90
            elif lastposition == 0:
                lastposition = 90

            first_rect_postition1 = [0, 0, 0, 0]

            first_rect_postition1[0] = self.my_variables[self.my_counter - 1][1]
            first_rect_postition1[1] = -self.my_variables[self.my_counter - 1][0] - 30
            first_rect_postition1[2] = -self.my_variables[self.my_counter - 1][0] - 30
            first_rect_postition1[3] = lastposition

            self.my_variables.append(first_rect_postition1)
            self.my_counter += 1

        elif action == 2:
            # turn left
            lastposition = self.my_variables[self.my_counter - 1][3]

            if lastposition == 90:
                lastposition = 0
            elif lastposition == -90:
                lastposition = 180
            elif lastposition == 180:
                lastposition = 90
            elif lastposition == 0:
                lastposition = -90

            first_rect_postition = [0, 0, 0, 0]

            first_rect_postition[0] = -self.my_variables[self.my_counter - 1][1] - 30
            first_rect_postition[1] = self.my_variables[self.my_counter - 1][0]
            first_rect_postition[2] = self.my_variables[self.my_counter - 1][0]
            first_rect_postition[3] = lastposition

            self.my_variables.append(first_rect_postition)
            self.my_counter += 1

        self.my_variables[self.my_counter - 1][1] -= 30

        # SETTING THE CHECKER
        lastposition = self.my_variables[self.my_counter - 1][3]
        gauche = self.my_variables[self.my_counter - 1][0]
        haut = self.my_variables[self.my_counter - 1][1]

        if lastposition == 90:
            checker = [-haut - 1, gauche, -haut, gauche + 30]
        elif lastposition == -90:
            checker = [haut, -gauche - 30, haut - 1, -gauche]
        elif lastposition == 180:
            checker = [-gauche - 30, -haut - 1, -gauche, -haut]
        else:
            checker = [gauche, haut, gauche + 30, haut + 1]

        for forloop in range(self.my_counter):
            left = self.my_variables[forloop][0]
            top = self.my_variables[forloop][1]
            bot = self.my_variables[forloop][2]
            lastposition = self.my_variables[forloop][3]

            if lastposition == 90:
                rect1 = [-bot, left, -top, left + 30]
            elif lastposition == -90:
                rect1 = [top, -left - 30, bot, -left]
            elif lastposition == 180:
                rect1 = [-left - 30, -bot, -left, -top]
            else:
                rect1 = [left, top, left + 30, bot]

            if checker[0] < rect1[2] and checker[2] > rect1[0] and checker[1] < rect1[3] and checker[3] > \
                    rect1[1] and forloop < self.my_counter - 1:
                if self.bump_self == 0:
                    self.bump_self = 1
                else:
                    self.still_traveling = False

        for forloop in range(self.his_counter):
            left = self.his_variables[forloop][0]
            top = self.his_variables[forloop][1]
            bot = self.his_variables[forloop][2]
            lastposition = self.his_variables[forloop][3]

            if lastposition == 90:
                rect1 = [-bot, left, -top, left + 30]
            elif lastposition == -90:
                rect1 = [top, -left - 30, bot, -left]
            elif lastposition == 180:
                rect1 = [-left - 30, -bot, -left, -top]
            else:
                rect1 = [left, top, left + 30, bot]

            if checker[0] < rect1[2] and checker[2] > rect1[0] and checker[1] < rect1[3] and checker[3] > \
                    rect1[1] and forloop < self.my_counter - 1:
                self.still_traveling = False

        # CHECK IF BUMP TO EDGES
        left_edge = self.left_edge
        right_edge = self.rigth_edge
        top_edge = self.top_edge
        bot_edge = self.bot_edge

        left_b = checker[0] < left_edge[2] and checker[1] < left_edge[3] and \
                 checker[3] > left_edge[1]
        right_b = checker[2] > right_edge[0] and checker[1] < right_edge[3] and \
                  checker[3] > right_edge[1]
        top_b = checker[0] < top_edge[2] and checker[2] > top_edge[0] and checker[1] < top_edge[3]

        bot_b = checker[0] < bot_edge[2] and checker[2] > bot_edge[0] and checker[3] > \
                bot_edge[1]

        if left_b or right_b or top_b or bot_b:
            self.still_traveling = False

        result = self.preprocessing(self.my_variables, self.his_variables, self.my_counter)

        if self.still_traveling:
            reward = 1
            self.load()
            return ts.transition(result, reward)
        else:
            reward = 0
            self.load()
            return ts.termination(result, reward)


'''
file = open("test.txt",'w')
for x in arr:
    for a in x:
        file.write(str(a)+" ")
    file.write("\n")
file.close()
'''
# from tf_agents.environments import tf_py_environment
# pyenv = SnakeEnv()
# env = tf_py_environment.TFPyEnvironment(pyenv)

# model = tf.saved_model.load('model1')
# print(model.v.numpy())



