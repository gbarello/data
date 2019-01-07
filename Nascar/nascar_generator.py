import numpy as np

def rotmat(th):
    return np.array([[np.cos(np.pi*th),np.sin(np.pi*th)],[-np.sin(np.pi*th),np.cos(np.pi*th)]])

def make_random_video(vlen,dt,size = 50,getpic = True,switch_states = False):

    out = []
    loc = (np.array([2,4])*np.random.rand(2) - np.array([1,2])) * .9

    state = 1    
    if np.random.rand() < .5 and switch_states:
        state = -state        

    for k in range(max(int(1 + 4./dt),vlen)):
        out.append(loc)
        if abs(loc[1]) < 2:
            if loc[0] < 0:
                loc = loc + state*dt*np.array([0,-1])
            else:
                loc = loc - state*dt*np.array([0,-1])

        elif loc[1] > 1:
            loc = np.array([0,2]) + np.dot(rotmat(-state*dt),(loc - np.array([0,2])))
        else:
            loc = np.array([0,-2]) + np.dot(rotmat(-state*dt),(loc - np.array([0,-2])))


    out = np.array(out)
    
    out += np.random.normal(0,.01,out.shape)

    if getpic == False:
        return out
            
    imout = np.array([paint_car(o,size) for o in out])

    ind = np.random.randint(0,max(int(1 + 4./dt),vlen) - vlen + 1)
    
    return imout[ind:ind+vlen], np.array(out)[ind:ind+vlen]
#    return imout, np.array(out)

def paint_car(loc,size):
    out = np.zeros([size,size])

    L = (loc + [1,2])/np.array([2,6])

    L = np.int32(L*size)

    L = np.clip(L,0,size-1)

    out[L[0],L[1]] = 1

    return out
