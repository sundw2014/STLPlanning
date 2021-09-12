import sys
import numpy as np

from PWLPlan import plan, Node
from vis import vis

def test():

    wall_half_width = 0.1

    ps = np.array([
        [55,144],
        [104,44],
        [211,19],
        [300,89],
        [451,88],
        [453,202],
        [304,203],
        [211,272],
        [101,247],
        [119,146],
        [143,97],
        [197,85],
        [242,119],
        [239,175],
        [195,208],
        [144,195]], dtype=np.float64)

    ps[:, 1] = 281 - ps[:, 1]
    ps = (ps / 532. * 20.).tolist()

    x0 = np.array(ps[9:16]).mean(axis=0).tolist()
    _walls = [
        [ps[0], ps[1]],
        [ps[1], ps[2]],
        [ps[2], ps[3]],
        [ps[3], ps[4]],
        [ps[4], ps[5]],
        [ps[5], ps[6]],
        [ps[6], ps[7]],
        [ps[7], ps[8]],
        [ps[8], ps[0]],
        [ps[0], ps[9]],
        [ps[1], ps[10]],
        [ps[2], ps[11]],
        [ps[3], ps[12]],
        [ps[6], ps[13]],
        [ps[7], ps[14]],
        [ps[8], ps[15]]]

    def lineFromPoints(P, Q):
        a = Q[1] - P[1]
        b = P[0] - Q[0]
        c = a*(P[0]) + b*(P[1])
        return np.array([a, b]), c

    walls = []
    for wall in _walls:
        A0, b0 = lineFromPoints(*wall)
        A1 = A0
        b1 = b0 + np.linalg.norm(A0) * wall_half_width
        A2 = -A0
        b2 = -(b0 - np.linalg.norm(A0) * wall_half_width)

        A0 = np.array([-A0[1], A0[0]])
        b0 = (np.array(wall).mean(axis = 0) * A0).sum()
        half_length = np.sqrt((((np.array(wall[0]) - np.array(wall[1])))**2).sum()) / 2
        A3 = A0
        b3 = b0 + np.linalg.norm(A0) * half_length
        A4 = -A0
        b4 = -(b0 - np.linalg.norm(A0) * half_length)

        A = np.array([A1, A2, A3, A4], dtype = np.float64)
        b = np.array([b1, b2, b3, b4], dtype = np.float64)
        walls.append((A, b))

    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    _doors = []
    ymin = ps[6][1]; ymax = ps[4][1]
    xmin = ps[6][0]; xmax = ps[5][0]
    _doors.append(np.array([xmin + 1 * (xmax - xmin) / 7., xmin + 1 * (xmax - xmin) / 7., ymin, ymax], dtype = np.float64))
    _doors.append(np.array([xmin + 2 * (xmax - xmin) / 7., xmin + 2 * (xmax - xmin) / 7., ymin, ymax], dtype = np.float64))
    _doors.append(np.array([xmin + 3 * (xmax - xmin) / 7., xmin + 3 * (xmax - xmin) / 7., ymin, ymax], dtype = np.float64))
    _doors.append(np.array([xmin + 4 * (xmax - xmin) / 7., xmin + 4 * (xmax - xmin) / 7., ymin, ymax], dtype = np.float64))
    _doors.append(np.array([xmin + 5 * (xmax - xmin) / 7., xmin + 5 * (xmax - xmin) / 7., ymin, ymax], dtype = np.float64))
    _doors.append(np.array([xmin + 6 * (xmax - xmin) / 7., xmin + 6 * (xmax - xmin) / 7., ymin, ymax], dtype = np.float64))

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
    _keys.append([2, 3, 11, 12])
    _keys.append([1, 2, 10, 11])
    _keys.append([0, 1, 9, 10])
    _keys.append([8, 0, 15, 9])
    _keys.append([7, 8, 14, 15])
    _keys.append([6, 7, 13, 14])

    keys = []
    key_half_width = 0.3
    for key in _keys:
        key = np.array([ps[key[0]], ps[key[1]], ps[key[2]], ps[key[3]]], dtype = np.float64).mean(axis=0)
        key = np.array([-(key[0] - key_half_width), (key[0] + key_half_width), -(key[1] - key_half_width), (key[1] + key_half_width)])
        keys.append((A, key))

    b = np.array([-(xmin + 6.5 * (xmax - xmin) / 7. - 0.3), xmin + 6.5 * (xmax - xmin) / 7. + 0.3, -((ymin + ymax) / 2 - 0.3), (ymin + ymax) / 2 + 0.3], dtype = np.float64)
    goal = (A, b)

    tmax = 1000.
    vmax = 3.

    # goal = keys[0]
    keys = keys[0:6]
    doors = doors[0:6]

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
    PWL = plan(x0s, specs, bloat=0.2, MIPGap = 0.99, num_segs=28, tmax=tmax)

    plots = [[[goal,], 'b'], [keys, 'g'], [doors, 'r'], [walls, 'k']]
    return x0s, plots, PWL

if __name__ == '__main__':
    results = vis(test)
