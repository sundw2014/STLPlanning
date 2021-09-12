import sys
import numpy as np

from PWLPlan import plan, Node
from vis import vis

def test():
    x0s = [[2, 0.5], [4, 0.5], [6, 0.5], [8, 0.5]]

    wall_half_width = 0.05
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    walls = []

    walls.append(np.array([0, 0, 0, 4], dtype = np.float64))
    walls.append(np.array([10, 10, 0, 4], dtype = np.float64))
    walls.append(np.array([0, 10, 0, 0], dtype = np.float64))
    walls.append(np.array([0, 10, 4, 4], dtype = np.float64))

    walls.append(np.array([0, 4., 2, 2], dtype = np.float64))
    walls.append(np.array([5., 10, 2, 2], dtype = np.float64))

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

    b1 = np.array([-1.5, 2.5, -3, 4], dtype = np.float64)
    b2 = np.array([-3.5, 4.5, -3, 4], dtype = np.float64)
    b3 = np.array([-5.5, 6.5, -3, 4], dtype = np.float64)
    b4 = np.array([-7.5, 8.5, -3, 4], dtype = np.float64)
    goals = [(A, b1), (A, b2), (A, b3), (A, b4)]

    tmax = 5.
    vmax = 3.

    specs = []
    for i in range(4):
        # avoids = [Node('negmu', info={'A':A, 'b':b}) for A, b in obs]
        # avoids += [Node('negmu', info={'A':goals1[j][0], 'b':goals1[j][1]}) for j in range(4) if j is not i]
        # avoids += [Node('negmu', info={'A':goals2[j][0], 'b':goals2[j][1]}) for j in range(4) if j is not i]
        # avoid_obs = Node('and', deps=avoids)

        # # notGoal1 = Node('negmu', info={'A':goals1[i][0], 'b':goals1[i][1]})
        # goal1 = Node('mu', info={'A':goals1[i][0], 'b':goals1[i][1]})
        # finally_reach_goal1 = Node('F', deps=[goal1,], info={'int':[0, tmax]})
        # # goal2 = Node('mu', info={'A':goals2[i][0], 'b':goals2[i][1]})
        # # finally_reach_goal2 = Node('F', deps=[goal2,], info={'int':[0, tmax]})
        # # trans = [Node('or', deps=[notGoal1, finally_reach_goal2]),]
        # # always = Node('A', deps=[Node('and', deps=[avoid_obs,]+trans), ], info={'int':[0, tmax]})
        # always = Node('A', deps=[Node('and', deps=[avoid_obs,]), ], info={'int':[0, tmax]})

        # specs.append(Node('and', deps=[always, finally_reach_goal1]))

        avoids = [Node('negmu', info={'A':A, 'b':b}) for A, b in obs]
        avoids += [Node('negmu', info={'A':goals[j][0], 'b':goals[j][1]}) for j in range(4) if j is not i]
        avoid_obs = Node('and', deps=avoids)
        always_avoid_obs = Node('A', deps=[avoid_obs,], info={'int':[0,tmax]})
        reach_goal = Node('mu', info={'A':goals[i][0], 'b':goals[i][1]})
        finally_reach_goal = Node('F', deps=[reach_goal,], info={'int':[0, tmax]})
        specs.append(Node('and', deps=[always_avoid_obs, finally_reach_goal]))

    PWL = plan(x0s, specs, bloat=0.2, MIPGap=0.05, num_segs=6, tmax=tmax, vmax=vmax)

    plots = [[goals, 'g'], [obs, 'k']]
    return x0s, plots, PWL

if __name__ == '__main__':
    results = vis(test)
