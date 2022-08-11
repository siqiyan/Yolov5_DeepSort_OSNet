def det2gps(output):
    bbox_left = output[0]
    bbox_top = output[1]
    bbox_w = output[2] - output[0]
    bbox_h = output[3] - output[1]
    x = float(bbox_left + 0.5 * bbox_w)
    y = float(bbox_top + bbox_h)
    x1 = (1.40427155e+00*x+  3.16291211e+00*y+ -2.37218408e+02)/(-9.46816392e-19*x+  5.39028731e-03*y+ 1.00000000e+00)
    y1 = (0.00000000e+00*x+  5.20442410e+00*y+ -2.85011442e+02)/(-9.46816392e-19*x+  5.39028731e-03*y+ 1.00000000e+00)
    lx = (113.53501525000001-113.53491673666667)/(777-514)*(x1-514)+113.53491673666667 + 0.00002  # lat
    ly = (53.49529403205129-53.49512804333333)/(603-429)*(y1-429)+53.49512804333333 # lon
    return lx, ly
