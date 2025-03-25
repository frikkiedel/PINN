import nbformat
from nbconvert import PythonExporter
import torch
import numpy as np
from scipy.integrate import solve_ivp

def clean_script_line(line):
    if line.startswith('#!') or line.startswith('# coding:'):
        return False
    if line.strip().startswith('# In['):
        return False
    return True

def notebook_to_script(notebook_path, output_path):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Convert to Python script
    python_exporter = PythonExporter()
    python_exporter.exclude_markdown = True
    python_exporter.exclude_output = True
    
    (body, _) = python_exporter.from_notebook_node(nb)
    
    # Clean script lines
    clean_lines = filter(clean_script_line, body.splitlines())

    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(clean_lines))

#notebook_to_script('burgers_pinn.ipynb', 'clean.py')

def numerical_derivative(x, y):
    """
    Calculate the numerical derivative of y with respect to x using central difference method.

    Parameters:
    x (list or numpy array): Input values.
    y (list or numpy array): Corresponding function values.

    Returns:
    x (list or numpy array): Input values.
    y_prime (list or numpy array): Numerical derivative values.
    """
    import numpy as np

    # Check if x and y have the same length
    if len(x) != len(y):
        raise ValueError("Input arrays x and y must have the same length.")

    n = len(x)
    y_prime = np.zeros(n)

    # Calculate the derivative for interior points using central difference
    for i in range(1, n - 1):
        y_prime[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])

    # Calculate the derivative for the first and last points using forward and backward difference
    y_prime[0] = (y[1] - y[0]) / (x[1] - x[0])
    y_prime[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])

    return x, y_prime

### PENDULUM FUNCTIONS ###
def pendulum_motion(final_time, initial_position, initial_velocity, g=9.81, R=0.1, damping_coefficient=0.1, num_points=100):
    """
    Numerically solves the motion of a damped pendulum using the complete equation of motion.
    
    :param final_time: Final time of simulation
    :param initial_position: Initial angular position of the pendulum (in radians)
    :param initial_velocity: Initial angular velocity of the pendulum (in radians/s)
    :param g: Acceleration due to gravity
    :param R: Length of the pendulum
    :param damping_coefficient: Coefficient for the damping term, proportional to angular velocity
    :param num_points: Number of integration points (default is 100)
    :return: Tuple (t, theta) where t is the time array and theta is the angular position array
    """
    # Differential equation for the pendulum
    def pendulum_equation(t, y):
        theta, omega = y
        dtheta_dt = omega
        domega_dt = -(g/R)*np.sin(theta) - damping_coefficient*omega
        return [dtheta_dt, domega_dt]

    # Initial conditions
    y0 = [initial_position, initial_velocity]

    # Time span
    t_span = [0, final_time]
    t_eval = np.linspace(0, final_time, num_points)

    # Solving the differential equation
    sol = solve_ivp(pendulum_equation, t_span, y0, t_eval=t_eval)

    return sol.t, sol.y[0]

### Pendulum animations ###

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

def animate_pendulum_to_gif(times, angles, radius=1.0, filename='pendulum_animation.gif'):
    """
    Animates a pendulum's motion and saves it as a GIF.
    
    This function now handles multi-dimensional angle arrays by iterating over the first dimension.

    :param times: List or Tensor of time points
    :param angles: List or Tensor of angular positions in radians (can be multi-dimensional)
    :param radius: Length of the pendulum
    :param filename: Filename to save the GIF
    """

    # Convert tensors to numpy arrays, detaching if necessary
    if isinstance(times, torch.Tensor):
        times = times.detach().cpu().numpy()
    if isinstance(angles, torch.Tensor):
        angles = angles.detach().cpu().numpy()

    # Handle multi-dimensional angles array
    if len(angles.shape) > 1:
        # Assuming the first dimension corresponds to different time steps
        angles = angles[:, 0]

    plt.ioff()

    images = []
    buffers = []
    for angle in angles:
        fig, ax = plt.subplots()
        ax.set_xlim(-radius * 1.2, radius * 1.2)
        ax.set_ylim(-radius * 1.2, radius * 1.2)
        ax.set_aspect('equal', adjustable='box')
        x = radius * np.sin(angle)
        y = -radius * np.cos(angle)
        ax.plot([0, x], [0, y], lw=2, marker='o')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(Image.open(buf))
        buffers.append(buf)

        plt.close(fig)

    images[0].save(filename, save_all=True, append_images=images[1:], duration=np.mean(np.diff(times))*1000, loop=0)

    for buf in buffers:
        buf.close()

    plt.ion()
    return images

def combine_gifs(frames1, frames2, total_time=2000, filename='combined_animation.gif'):
    """
    Combines two sets of frames into a single GIF, side by side, 
    with the total animation time being 2 seconds.
    
    :param frames1: List of PIL Image frames from the first animation
    :param frames2: List of PIL Image frames from the second animation
    :param total_time: Total time for the GIF in milliseconds (default 2000 for 2 seconds)
    :param filename: Filename to save the combined GIF
    """
    combined_images = []

    for img1, img2 in zip(frames1, frames2):
        total_width = img1.width + img2.width
        total_height = max(img1.height, img2.height)
        combined_image = Image.new('RGB', (total_width, total_height))

        combined_image.paste(img1, (0, 0))
        combined_image.paste(img2, (img1.width, 0))

        combined_images.append(combined_image)

    # Calculate frame duration
    frame_duration = total_time / len(combined_images)

    combined_images[0].save(filename, save_all=True, append_images=combined_images[1:], loop=0, duration=frame_duration)