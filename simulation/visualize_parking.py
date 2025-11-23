import matplotlib.pyplot as plt

def plot_env(env):
    obs = env.obstacles.obstacles
    fig, ax = plt.subplots()
    for o in obs:
        rect = plt.Rectangle((o['x']-o['w']/2,o['y']-o['h']/2),o['w'],o['h'],fill=True,alpha=0.5)
        ax.add_patch(rect)
    x,y,yaw,v = env.state
    ax.plot([x],[y],'ro')
    gx,gy,gyaw = env.goal
    ax.plot([gx],[gy],'gx')
    ax.set_aspect('equal', 'box')
    plt.show()
