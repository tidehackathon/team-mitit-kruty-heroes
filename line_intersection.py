import numpy as np
import cv2


def calculate_distances_between_points(points):
    """Calculate distances between all pairs of points inside array."""
    points_diff = points - np.expand_dims(points, axis=1)
    return np.sqrt(np.sum(points_diff ** 2, axis=-1))


def dist_between_lines(line_a, line_b):
    """Estimate the closest distance between two line segments.

    Args:
        line_a (np.ndarray): line a defined by pair of points a0, a1
        line_b (np.ndarray): line b defined by pair of points b0, b1

    Returns:
        float: distance between lines
    """

    a0 = np.append(line_a[:2], [0], axis=0)
    a1 = np.append(line_a[2:], [0], axis=0)
    b0 = np.append(line_b[:2], [0], axis=0)
    b1 = np.append(line_b[2:], [0], axis=0)

    # Calculate denomitator
    _a = a1 - a0
    _b = b1 - b0
    mag_a = np.linalg.norm(_a)
    mag_b = np.linalg.norm(_b)

    _a = _a / mag_a
    _b = _b / mag_b

    cross = np.cross(_a, _b)
    denom = np.linalg.norm(cross)**2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        dot0 = np.dot(_a, (b0-a0))
        dot1 = np.dot(_a, (b1-a0))

        # Is segment B before A?
        if dot0 <= 0 >= dot1:
            return np.linalg.norm(a0-b0) if np.abs(dot0) < np.abs(dot1) else np.linalg.norm(a0-b1)
        # Is segment B after A?
        if dot0 >= mag_a <= dot1:
            return np.linalg.norm(a1-b0) if np.abs(dot0) < np.abs(dot1) else np.linalg.norm(a1-b1)

        # Segments overlap, return distance between parallel segments
        return np.linalg.norm(((dot0 * _a) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    tdiff = (b0 - a0)
    td0 = max(0, np.linalg.det([tdiff, _b, cross]) / denom)
    td1 = max(0, np.linalg.det([tdiff, _a, cross]) / denom)

    # Projected closest point on segment A
    point_a = a0 + (_a * td0) if td0 <= mag_a else a1
    # Projected closest point on segment B
    point_b = b0 + (_b * td1) if td1 <= mag_b else b1

    # Clamp projection A
    if (td0 <= 0) or (td0 > mag_a):
        point_b = b0 + (_b * np.clip(np.dot(_b, (point_a-b0)), 0, mag_b))

    # Clamp projection B
    if (td1 <= 0) or (td1 > mag_b):
        point_a = a0 + (_a * np.clip(np.dot(_a, (point_b-a0)), 0, mag_a))

    return np.around(np.linalg.norm(point_a-point_b), decimals=5)


def get_line_intersection(line0, line1):
    """Estimate intersection between two lines.

    Args:
        line0 (list): coordinates of first line
        line1 (list): coordinates of second line

    Returns:
        angle (float): angle between lines (in radians)
        (tuple): coordinates of intersection in order x, y
            (both equal to None if lines are parallel)
    """
    x11, y11, x12, y12 = line0[0]
    x21, y21, x22, y22 = line1[0]
    kl1, kl2 = 0, 0
    if (x12 - x11) == 0 and (x22 - x21) == 0:
        return np.pi, (None, None)
    if (x12 - x11) == 0:
        kl2 = (y22 - y21) / (x22 - x21)
        b2 = -x21 * kl2 + y21
        x = x11
        y = kl2 * x - x21 * kl2 + y21
    elif (x22 - x21) == 0:
        kl1 = (y12 - y11) / (x12 - x11)
        b1 = -x11 * kl1 + y11
        x = x21
        y = kl1 * x - x11 * kl1 + y11
    else:
        kl1 = (y12 - y11) / (x12 - x11)
        kl2 = (y22 - y21) / (x22 - x21)
        b1 = -x11 * kl1 + y11
        b2 = -x21 * kl2 + y21
        if kl1 != kl2:
            x = (b2 - b1) / (kl1 - kl2)
            y = kl1 * x + b1
    if kl1 == kl2:
        angle = np.pi
        x, y = None, None
    elif kl2 * kl1 == -1:
        angle = np.pi/2
    else:
        angle = np.arctan((kl2 - kl1)/(1 + kl2 * kl1))
    return angle, (x, y)


def line_intersection_extraction(mask_in):
    _, mask = cv2.threshold(mask_in, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(mask, 100, 200)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 70, None, 125, 75)

    inter_points = []
    for i, line0 in enumerate(linesP):
        for line1 in linesP[i+1:]:

            dist = dist_between_lines(np.array(line0[0]), np.array(line1[0])) 
            if dist > 300:
                continue
            angle, ip = get_line_intersection(line0, line1)
            if ip[0] is None or ip[0] < 0 or ip[1] < 0 or ip[0] > mask.shape[1] or ip[1] > mask.shape[0]:
                continue
            if np.abs(angle) < 0.3:
                continue
            #print(i, ip, angle)
            inter_points.append(ip)

    # filter duplicates/too close intersection points
    if len(inter_points) != 0:
        dist = calculate_distances_between_points(np.array(inter_points))
        repeated_idxs = sorted(
            np.unique(
                [j for i, j in zip(*np.nonzero(dist <= 1))
                 if j > i]
            )
        )
        for idx in repeated_idxs[::-1]:
            inter_points.pop(idx)

    return linesP, inter_points
