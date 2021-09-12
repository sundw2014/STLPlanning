import sys
import numpy as np

from PWLPlan import plan, Node
from vis import vis

def test():
    x0 = [-1., -1]

    wall_half_width = 0.05
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    walls = []

    walls.append(np.array([-1.5, -1.5, -1.5, 1.5], dtype = np.float64))
    walls.append(np.array([1.5, 1.5, -1.5, 1.5], dtype = np.float64))
    walls.append(np.array([-1.5, 1.5, -1.5, -1.5], dtype = np.float64))
    walls.append(np.array([-1.5, 1.5, 1.5, 1.5], dtype = np.float64))

    obs = []
    for wall in walls:
        if wall[0]==wall[1]:
            wall[0] -= wall_half_width
            wall[1] += wall_half_width
        elif wall[2]==wall[3]:
            wall[2] -= wall_half_width
            wall[3] += wall_half_width
        else:
            raise ValueError('wrong shape for axis-aligned wall')
        wall *= np.array([-1,1,-1,1])
        obs.append((A, wall))

    b1 = np.array([1, -0.7, 0.25, 0.5], dtype = np.float64)
    B1 = (A, b1)
    b2 = np.array([0, 0.9, 1, -0.5], dtype = np.float64)
    B2 = (A, b2)
    b3 = np.array([-0.2, 0.7, -0.8, 1.2], dtype = np.float64)
    B3 = (A, b3)
    C = np.array([0.4, 0.4, 0.4, 0.4], dtype = np.float64)
    C = (A, C)

    goal = [1,1]

    plots = [[[B1,], 'y'], [[B2,], 'r'], [[B3,], 'g'], [[C,], 'b'], [obs, 'k']]
    tmax = 10.
    vmax = 1.

    notC = Node('negmu', info={'A':C[0], 'b':C[1]})
    B1 = Node('mu', info={'A':B1[0], 'b':B1[1]})
    B2 = Node('mu', info={'A':B2[0], 'b':B2[1]})
    notB3 = Node('negmu', info={'A':B3[0], 'b':B3[1]})

    phi_1 = Node('F', deps=[Node('A', deps=[B1,], info={'int':[0,5]}),], info={'int':[0,tmax]})
    phi_2 = Node('A', deps=[notB3,], info={'int':[0,tmax]})
    phi_3 = Node('A', deps=[notC,], info={'int':[0,tmax]})
    spec = Node('and', deps=[phi_1, phi_2, phi_3])

    x0s = [x0,]
    specs = [spec,]
    goals = [goal, ]
    PWL = plan(x0s, specs, bloat=0.05, size=0.11/2, num_segs=7, tmax=tmax, vmax=vmax, hard_goals=goals)

    return x0s, plots, PWL

if __name__ == '__main__':
    results = vis(test)
