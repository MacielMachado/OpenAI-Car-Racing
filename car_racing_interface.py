from cart_racing import CarRacing
import numpy as np
import pygame
import time
import os


class CarRacingInterface():
    ''' Cria uma interface para que outros módulos acessem o jogo.
    '''
    def __init__(self, render):
        '''
        '''
        self.path = os.getcwd() + '/tutorial/'
        self.a = np.array([0.0, 0.0, 0.0])
        self.start_saving = False
        self.render = render
        self.isopen = True
        self.actions = []
        self.states = []

    def run(self):
        '''
        '''
        self.initialize_environment()
        self.run_game()

    def run_game(self):
        '''
        '''
        while self.isopen:
            self.reset_environment()
            while True:
                self.save_game()
                self.register_input()
                self.step(self.a)
                self.total_reward += self.r
                if self.steps % 200 == 0 or self.done:
                    print("\naction " + str([f"{x:+0.2f}" for x in self.a]))
                    print(f"step {self.steps} total_reward {self.total_reward:+0.2f}")
                self.steps += 1
                self.isopen = self.env.render()
                if self.done or self.restart or self.isopen is False:
                    break

        hash = str(int(time.time()))
        np.save(self.path+'states_'+hash+'.npy', 
                self.states)
        np.save(self.path+'actions_'+hash+'.npy', 
                self.actions)
        self.env.close()

    def save_game(self):
        '''
        '''
        if True:
            self.actions.append(self.a.copy())
            self.states.append(self.s.copy())


    def step(self, action):
        '''
        '''
        self.s, self.r, self.done, self.info = self.env.step(action)


    def initialize_environment(self):
        '''
        '''
        np.random.seed(0)
        self.env = CarRacing()
        # self.env.seed(0)
        if self.render: self.env.render()

    def reset_environment(self):
        ''' Reset the environment and the variables related to it.
        '''
        # self.env.seed(0)
        np.random.seed(0)
        self.s = self.env.reset()
        self.total_reward = 0.0
        self.steps = 0
        self.restart = False

    def register_input(self):
        ''' Associates inputs from the keyboard and game actions.
        '''
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    self.a[0] = +1.0
                if event.key == pygame.K_UP:
                    self.a[1] = +1.0
                    self.start_saving = True
                if event.key == pygame.K_DOWN:
                    self.a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    hash = str(int(time.time()))
                    np.save(self.path+'states_'+hash+'.npy', 
                            self.states)
                    np.save(self.path+'actions_'+hash+'.npy', 
                            self.actions)
                    global restart
                    self.restart = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self.a[0] = 0
                if event.key == pygame.K_RIGHT:
                    self.a[0] = 0
                if event.key == pygame.K_UP:
                    self.a[1] = 0
                if event.key == pygame.K_DOWN:
                    self.a[2] = 0

if __name__ == '__main__':
    construtor = CarRacingInterface(render=True)
    construtor.run()