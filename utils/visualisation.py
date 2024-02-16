import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

def dynamic_plot(landmarks: pd.DataFrame, label: str):
    """
    Plot a dynamic 3D scatter plot of the landmarks

    Args:
        landmarks (pd.DataFrame): The landmarks to plot
        label (str): The title of the plot
    """
    x = landmarks.filter(regex='.*pos_x.*').to_numpy()
    y = landmarks.filter(regex='.*pos_y.*').to_numpy()
    z = landmarks.filter(regex='.*pos_z.*').to_numpy()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(label)

    ax.set_xlim(0, 0.8)
    ax.set_ylim(0.5, 1)
    ax.set_zlim(0.1, 0.7)
    scat = ax.scatter3D(x[0], y[0], z[0], c='r', marker='o')

    def animate(j):
        if j < landmarks.shape[0]:
            scat._offsets3d = (x[j], y[j], z[j])
        return scat

    return animation.FuncAnimation(fig, animate, frames=landmarks.shape[0], interval=30, blit=False)