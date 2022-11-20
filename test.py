from car_racing_interface import CarRacingInterface
from cart_racing import CarRacing
from data_precessing import DataHandler
from model import Model, Model2
import torch

class Tester():
    def __init__(self, model, env, render=True, device="mps"):
        self.device = device
        self.render = render
        self.model = model.to(device)
        self.env = env

    def run(self):
        reward = 0
        obs = self.env.reset()
        while True:
            self.model.eval()
            obs = DataHandler().to_greyscale(obs)
            obs = DataHandler().normalizing(obs)
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(self.device)
            action_tensor = self.model(obs_tensor).to("cpu")
            action = action_tensor.detach().numpy()[0]

            obs, new_reward, done, _ = self.env.step(action)
            reward += new_reward
            if self.render: self.env.render()
            if done: break

if __name__ == '__main__':
    model = Model()
    model.load_state_dict(torch.load("./model_pytorch/model.pkl"))
    env_interface = CarRacingInterface(render = True)
    env = CarRacing()
    Tester(model=model,env=env).run()
