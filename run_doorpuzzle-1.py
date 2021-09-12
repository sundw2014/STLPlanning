import sys
import numpy as np

from PWLPlan import plan, Node
from vis import vis

def test():
    x0 = [4.5, 2.]

    wall_half_width = 0.1
    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    _walls = []

    _walls.append(np.array([0, 0, 0, 8], dtype = np.float64))
    _walls.append(np.array([8, 8, 0, 8], dtype = np.float64))
    _walls.append(np.array([0, 8, 0, 0], dtype = np.float64))
    _walls.append(np.array([0, 8, 8, 8], dtype = np.float64))
    _walls.append(np.array([0, 7, 6, 6], dtype = np.float64))
    _walls.append(np.array([2, 2, 1, 4], dtype = np.float64))
    _walls.append(np.array([2, 4, 4, 4], dtype = np.float64))
    _walls.append(np.array([4, 4, 4, 6], dtype = np.float64))
    _walls.append(np.array([6, 6, 0, 5], dtype = np.float64))

    walls = []
    for wall in _walls:
        if wall[0] == wall[1]:
            wall[0] -= wall_half_width
            wall[1] += wall_half_width
        elif wall[2] == wall[3]:
            wall[2] -= wall_half_width
            wall[3] += wall_half_width
        else:
            raise ValueError('wrong shape for axis-aligned wall')
        wall *= np.array([-1,1,-1,1])
        walls.append((A, wall))

    _doors = []
    _doors.append(np.array([2, 2, 0, 1], dtype = np.float64))
    _doors.append(np.array([6, 6, 5, 6], dtype = np.float64))
    _doors.append(np.array([6, 8, 2, 2], dtype = np.float64))
    _doors.append(np.array([0, 2, 4, 4], dtype = np.float64))
    _doors.append(np.array([7, 8, 6, 6], dtype = np.float64))

    doors = []
    for door in _doors:
        if door[0]==door[1]:
            door[0] -= wall_half_width
            door[1] += wall_half_width
        elif door[2]==door[3]:
            door[2] -= wall_half_width
            door[3] += wall_half_width
        else:
            raise ValueError('wrong shape for axis-aligned door')
        door *= np.array([-1,1,-1,1])
        doors.append((A, door))

    _keys = []
    _keys.append(np.array([5, 1], dtype = np.float64))
    _keys.append(np.array([3, 3], dtype = np.float64))
    _keys.append(np.array([1, 1], dtype = np.float64))
    _keys.append(np.array([7, 1], dtype = np.float64))
    _keys.append(np.array([3, 5], dtype = np.float64))

    keys = []
    key_half_width = 0.55
    for key in _keys:
        key = np.array([-(key[0] - key_half_width), (key[0] + key_half_width), -(key[1] - key_half_width), (key[1] + key_half_width)])
        keys.append((A, key))

    b = np.array([-0.5, 1.5, -6.5, 7.5], dtype = np.float64)
    goal = (A, b)

    tmax = 30.
    vmax = 3.

    avoid_walls = Node('and', deps=[Node('negmu', info={'A':A, 'b':b}) for A, b in walls])
    always_avoid_walls = Node('A', deps=[avoid_walls, ], info={'int':[0,tmax]})

    avoid_doors = [Node('negmu', info={'A':A, 'b':b}) for A, b in doors]
    pick_keys = [Node('mu', info={'A':A, 'b':b}) for A, b in keys]
    untils = [Node('U', deps=[avoid_door, pick_key], info={'int':[0,tmax]}) for avoid_door, pick_key in zip(avoid_doors, pick_keys)]

    reach_goal = Node('mu', info={'A':goal[0], 'b':goal[1]})
    finally_reach_goal = Node('F', deps=[reach_goal,], info={'int':[0,tmax]})

    spec = Node('and', deps = untils + [always_avoid_walls, finally_reach_goal])

    x0s = [x0,]
    specs = [spec,]
    PWL = plan(x0s, specs, bloat=0.18, MIPGap = 0.99, num_segs=26, tmax=tmax, vmax=vmax)

    plots = [[[goal,], 'b'], [keys, 'g'], [doors, 'r'], [walls, 'k']]
    return x0s, plots, PWL

if __name__ == '__main__':
    results = vis(test)
