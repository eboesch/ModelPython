import numpy as np

def sigmoid(x):
        z = 1/(1 + np.exp(-x))
        return z


def gauss_topo(x, width, height):
    topo = np.zeros_like(x)
    toponf = height * np.exp(-(x / float(width)) ** 2)

    topo = toponf[1:-1] + 0.25 * (
        toponf[0:-2] - 2.0 * toponf[1:-1] + toponf[2:]
    )

    return topo

def rect_topo(x, width, height):
    topo = np.zeros_like(x)
    topo = np.where(np.abs(x)< width, height, 0)

    return topo[1:-1]


def trapez_topo(x, width_in, width_out,  height):
    """
    for width_in = 0, get a triangle

    for width_out = width_in + 1e-13, get rect_topo
    """
    topo = np.zeros_like(x)

    # make central plateau
    topo = np.where(np.abs(x)< width_in, height, 0)

    # make right downward slope
    topo = np.where((width_in < x) & (x < width_out), (x-width_out)*height/(width_in-width_out), topo)

    # make left upward slope
    topo = np.where((-width_in > x) & (x > -width_out), (x+width_out)*height/(width_out-width_in), topo)

    return topo[1:-1]

def double_trapez_topo(x, width_1, width_2, width_3, width_4, height_1, height_2):
    """
    for width_in = 0, get a triangle

    for width_out = width_in + 1e-13, get rect_topo
    """
    topo = np.zeros_like(x)

    # make central plateau
    topo = np.where(np.abs(x)< width_1, height_1, topo)

    # make side plateaus
    topo = np.where((np.abs(x)>width_2) & (np.abs(x)< width_3), height_2, topo)

    # make right / slope
    topo = np.where((width_1 < x) & (x < width_2), ((height_2-height_1)*x + height_1*width_2 - width_1*height_2)/(width_2-width_1), topo)
    # make right \ slope
    topo = np.where((width_3 < x) & (x < width_4), ((-height_2)*x + height_2*width_4)/(width_4-width_3), topo)

    # make left \ slope
    topo = np.where((-width_1 > x) & (x > -width_2), ((height_1-height_2)*x - height_2*width_1 + width_2*height_1)/(width_2-width_1), topo) 

    # make left / slope
    topo = np.where((-width_3 > x) & (x > -width_4), ((height_2)*x + width_4*height_2)/(width_4-width_3), topo)

    

    return topo[1:-1]

def double_gauss_topo(x, width, pos, height):
    topo = np.zeros_like(x)
    toponf = height * np.exp(-( (x-pos) / float(width)) ** 2) + height * np.exp(-( (x+pos) / float(width)) ** 2)

    topo = toponf[1:-1] + 0.25 * (
        toponf[0:-2] - 2.0 * toponf[1:-1] + toponf[2:]
    )

    return topo 

def rounded_topo(x, width_in, width_out,  height):
    """
    is not doing it's job -> only works for height = 2000
    """
    topo = np.zeros_like(x)
    wid = (width_out - width_in)/2

    
    # make right downward slope
    topo = np.where(x>=0, height*sigmoid(-(x-width_in-wid)*height/wid), topo)

    # make left upward slope
    topo = np.where(x<0, height*sigmoid((x+width_in+wid)*height/wid), topo)

    return topo[1:-1]


def rounded_topo_f(x, width_in, width_out,  height):
        """
        is not doing it's job -> only works for height = 2000
        """
        topo = np.zeros_like(x)
        wid = (width_out - width_in)/2

        # make central plateau
        topo = np.where(np.abs(x)< width_in, height, 0)

        h = 0.1*height
        height = 1.2*height
        # make right downward slope
        topo = np.where((width_in <= x) & (x <= width_out), height*sigmoid(-(x-width_in-wid)*height/wid)-h, topo)

        # make left upward slope
        topo = np.where((-width_in >= x) & (x >= -width_out), height*sigmoid((x+width_in+wid)*height/wid)-h, topo)

        return topo[1:-1]


    
def fetch_topo(x, topowd, topomx, topotype):
    if topotype=="gauss":
        return gauss_topo(x, topowd, topomx)
    elif topotype=="double_gauss":
        return double_gauss_topo(x, topowd, topowd, topomx)
    elif topotype=="rect":
        return trapez_topo(x, topowd, topowd +1e-13, topomx)
    elif topotype=="double_rect":
        return double_trapez_topo(x, topowd/2, topowd/2 +1e-13, topowd*3/2, topowd*3/2 +1e-13, topomx/2, topomx)
    elif topotype=="triang":
        return trapez_topo(x, 0, topowd, topomx )
    elif topotype=="double_triang":
        return double_trapez_topo(x, 0, topowd, topowd, topowd*3/2, topomx/2, topomx)
    elif topotype=="trapez":
        return trapez_topo(x, topowd/2, topowd*3/2, topomx)
    elif topotype=="double_trapez":
        return double_trapez_topo(x, topowd/2, topowd, topowd*2, topowd*5/2, topomx/2, topomx)
    else: 
        print("Unknown Topography Type")
        quit()