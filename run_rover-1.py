import sys
import numpy as np

from PWLPlan import plan, Node
from vis import vis

def test():
    x0 = [5.5, 6.5]

    wall_half_width = 0.05
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    walls = []

    walls.append(np.array([0, 0, 0, 11], dtype = np.float64))
    walls.append(np.array([12, 12, 0, 11], dtype = np.float64))
    walls.append(np.array([0, 12, 0, 0], dtype = np.float64))
    walls.append(np.array([0, 12, 11, 11], dtype = np.float64))

    walls.append(np.array([3, 3, 3, 4], dtype = np.float64))
    walls.append(np.array([3, 3, 7, 8], dtype = np.float64))
    walls.append(np.array([9, 9, 3, 4], dtype = np.float64))
    walls.append(np.array([9, 9, 7, 8], dtype = np.float64))

    walls.append(np.array([6, 6, 2, 9], dtype = np.float64))
    walls.append(np.array([5, 7, 2, 2], dtype = np.float64))
    walls.append(np.array([5, 7, 9, 9], dtype = np.float64))

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

    b1 = np.array([-0.8, 2.2, -0.8, 2.2], dtype = np.float64)
    b2 = np.array([-0.8, 2.2, -8.8, 10.2], dtype = np.float64)
    b3 = np.array([-9.8, 11.2, -0.8, 2.2], dtype = np.float64)
    b4 = np.array([-9.8, 11.2, -8.8, 10.2], dtype = np.float64)
    observation_spots = [(A, b1), (A, b2), (A, b3), (A, b4)]

    b5 = np.array([-0.8, 2.2, -4.8, 6.2], dtype = np.float64)
    b6 = np.array([-9.8, 11.2, -4.8, 6.2], dtype = np.float64)
    transmitters = [(A, b5), (A, b6)]

    b7 = np.array([-4, 6, -4, 7], dtype = np.float64)
    b8 = np.array([-6, 8, -4, 7], dtype = np.float64)
    charging_stations = [(A, b7), (A,b8)]

    tc = 10.
    td = 10.
    tmax = 30.
    vmax = 3.

    charging = Node('or', deps=[Node('mu', info={'A':A, 'b':b}) for A,b in charging_stations])
    charge_in_tc = Node('F', deps=[charging, ], info={'int':[0, tc]})
    phi_1 = Node('A', deps=[Node('or', deps=[charging, charge_in_tc]),], info={'int':[0,tmax]})

    Ois = [Node('mu', info={'A':A, 'b':b}) for A, b in observation_spots]
    phi_2 = Node('and', deps=[Node('F', deps=[Oi,], info={'int':[0,tmax]}) for Oi in Ois])

    transmitting = Node('or', deps=[Node('mu', info={'A':A, 'b':b}) for A, b in transmitters])
    transmitting_in_td = Node('F', deps=[transmitting, ], info={'int':[0, td]})
    notOis = [Node('negmu', info={'A':A, 'b':b}) for A, b in observation_spots]
    notO = Node('and', deps=notOis)
    phi_3 = Node('A', deps=[Node('or', deps=[notO, transmitting_in_td]),], info={'int':[0,tmax]})

    avoid_obs = Node('and', deps=[Node('negmu', info={'A':A, 'b':b}) for A, b in obs])
    phi_4 = Node('A', deps=[avoid_obs,], info={'int':[0,tmax]})

    spec = Node('and', deps=[phi_1, phi_2, phi_3, phi_4])

    x0s = [x0,]
    specs = [spec,]
    PWL = plan(x0s, specs, bloat=0.2, MIPGap = 0.85, num_segs=20, tmax=tmax, vmax=vmax)

    plots = [[transmitters, 'y'], [charging_stations, 'b'], [observation_spots, 'g'], [obs, 'k']]
    return x0s, plots, PWL

if __name__ == '__main__':
    results = vis(test)
