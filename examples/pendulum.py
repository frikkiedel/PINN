import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from functions import pendulum_motion, numerical_derivative
import time

class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

torch.manual_seed(42)
model = FCN(1,1,50,5)

# Initial and boundary conditions
t_final = 3
g = 9.81
R = 0.2
beta = 0

# For initial condition loss
dt = 1e-8
theta_dot_0 = 0
theta_0 = 0.6 #0.3

theta_initial = [theta_0, theta_0]
theta_initial = torch.tensor(theta_initial).view(-1,1)

theta_dot_initial = [theta_dot_0, theta_dot_0]
theta_dot_initial = torch.tensor(theta_dot_initial).view(-1,1)

t_initial = torch.tensor([0.0, 0.0]).requires_grad_(True).view(-1,1)

# Physics informed points
n = 200 # collation points
t = torch.linspace(0,t_final,n).view(-1,1).requires_grad_(True)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 50000

loss_history = [[],[]] # Epoch, loss
pbar = tqdm(total=n_epochs, desc="Initializing")

for i in range(n_epochs):

    optimizer.zero_grad()
    
    # Initial condition loss
    theta_0_model = model(t_initial)
    loss_i = torch.mean((theta_initial - theta_0_model) ** 2)

    theta_t_0 = torch.autograd.grad(theta_0_model, t_initial, torch.ones_like(theta_0_model), create_graph=True)[0]
    loss_i += torch.mean((theta_dot_0 - theta_t_0) ** 2)

    # Physics-informed loss
    
    theta = model(t)
    theta_t = torch.autograd.grad(theta, t, torch.ones_like(theta), create_graph=True)[0]
    theta_tt = torch.autograd.grad(theta_t, t, torch.ones_like(theta_t), create_graph=True)[0]
    loss_physics = torch.mean( (theta_tt + (g/R)*torch.sin(theta) + beta*theta_t)**2 )
    
    # Backpropagate joint loss

    p1 = 1000

    p2 = np.sin(i/100)**2
    loss = loss_i*p1 + loss_physics*p2

    loss.backward(retain_graph=True)
    optimizer.step()

    loss_history[0].append(loss.detach().numpy())
    loss_history[1].append(i)
    pbar.set_description(f'Epoch {i + 1}/{n_epochs}, Loss IC: {loss_i.item():.8f}, Loss Phys: {loss_physics.item():.8f}')
    
    #pbar.set_description(f'Epoch {i + 1}/{n_epochs}, Loss IC: {loss_i.item():.8f}, Loss BC: {loss_b.item():.8f}, Loss Phys: {loss_physics.item():.8f}, Loss Stat.: {stat_loss.item():.8f}')
    pbar.update(1)
pbar.close()

plt.figure(figsize=(10, 6))
plt.plot(loss_history[1], loss_history[0])
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.title('Loss over time')
plt.show()
lw = 4

t_numerical, theta_numerical = pendulum_motion(t_final, theta_0, theta_dot_0, g=g, R=R, damping_coefficient=beta, num_points=100)
t_dot_numerical, theta_dot_numerical = numerical_derivative(t_numerical,theta_numerical)

t = torch.tensor(t_numerical, dtype=torch.float32).view(-1,1)
t.requires_grad=True

u = model(t)
u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

plt.figure(figsize=(13, 5))
plt.subplot(1,2,1)
plt.plot(t_numerical, theta_numerical, 'k', linewidth=lw)
plt.plot(t.detach().numpy(), u.detach().numpy(), 'r--', linewidth=lw)
plt.xlabel('time (s)')
plt.ylabel(r'angular position ($\theta$)')
plt.grid(True)
plt.legend(['True', 'Predicted'])

MSE = torch.mean( (torch.tensor(theta_numerical) - u)**2 )

MSE_str = '{:.5f}'.format(MSE.item())
plt.suptitle('PINN Solution to damped single pendulum \n MSE = ' + MSE_str)

plt.subplot(1,2,2)
plt.plot(t_dot_numerical, theta_dot_numerical, 'k', linewidth=lw)
plt.plot(t.detach().numpy(), u_t.detach().numpy(), 'r--', linewidth=lw)
plt.xlabel('time (s)')
plt.ylabel(r'angular velocity ($\dot{\theta}$)')
plt.grid(True)
plt.legend(['True', 'Predicted'])
plt.show()