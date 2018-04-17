import numpy as np


def triangle_in(size,pos):
    if pos[0] < 0 or pos[1] < 0:
        return False
    if pos[0] <= pos[1]:
        return True
    else:
        return False
        
def square_in(size,pos):
    if pos[0] < 0 or pos[1] < 0:
        return False
    
    if pos[0] < size and pos[1] < size:
        return True
    else:
        return False
    
def shape_in(shape,size,pos):
    if shape == "triangle":
        return triangle_in(size,pos)
    elif shape == "square":
        return square_in(size,pos)
    else:
        print("shape not recognized")
        exit()

def paint_shape(frame,shape,size,pos):
    x = pos[0]
    y = pos[1]
    
    for i in np.arange(np.floor(size)):
        for j in np.arange(np.floor(size)):
            if shape_in(shape,size,[i,j]):
                frame[int(x+i),int(y+j)] = 1

    return frame

def update_variables(positions,speeds,sizes,fsize):
    tpos = np.copy(positions)
    tpos = positions + speeds

    tspeed = np.copy(speeds)

    LL = tpos
    LU = np.copy(tpos + np.array([[k - 1,k - 1] for k in sizes]))

    ii = 0

    while (np.any(LL < 0) or np.any(LU >= fsize)) and ii < 1000:
        ii += 1
        for k in range(len(LL)):
            if LL[k,0] < 0:
                tspeed[k,0] *= -1
                tpos[k,0] = -tpos[k,0]
                                
            if LL[k,1] < 0:
                tspeed[k,1] *= -1
                tpos[k,1] = -tpos[k,1]
                
            if LU[k,0] >= fsize:
                tspeed[k,0] *= -1
                tpos[k,0] = tpos[k,0] - 2*(LU[k,0] - fsize)
                
            if LU[k,1] >= fsize:
                tspeed[k,1] *= -1
                tpos[k,1] = tpos[k,1] - 2*(LU[k,1] - fsize)
                
            LL = np.copy(tpos)
            LU = np.copy(tpos + np.array([[k - 1,k - 1] for k in sizes]))
    if ii == 1000:
        print("failed to update frames")
        exit()

    positions = tpos
    speeds = tspeed

    return positions,speeds

def paint_shapes(frame,shapes,ssize,pos):
    for k in range(len(shapes)):
        frame = paint_shape(frame,shapes[k],ssize[k],pos[k])
    return frame

def make_video(shapes,pinit,sinit,ssize,fsize,time):
    
    video = []
    varibs = []
    pos = np.copy(pinit)
    spd = np.copy(sinit)
    
    for k in range(time):
        temp = np.zeros([fsize,fsize])
        varibs.append([np.copy(pos),np.copy(spd)])
        
        temp = paint_shapes(temp,shapes,ssize,pos)
        video.append(np.copy(temp))
        pos,spd = update_variables(pos,spd,ssize,fsize)

    video = np.array(video)

    varibs = np.array(varibs)

    return video,varibs,shapes,ssize

def make_random_video(nshapes,fsize,vlen):
    shapes = np.random.choice(["triangle","square"],[nshapes],replace = True)
    sizes = np.random.uniform(2,10,[nshapes])
    ipos = [np.random.uniform(0,fsize - k,[2]) for k in sizes]
    ivel = np.random.uniform(-3,3,[nshapes,2])

    out =  make_video(shapes,ipos,ivel,sizes,fsize,vlen)

    ddic = {"triangle":0,"square":1}

    OS = np.array([ddic[k] for k in out[2]])

    return out[0],out[1],out[3],OS

if __name__ == "__main__":
    A = make_random_video(2,30,100)

    np.savetxt("./test_video.csv",np.reshape(A[0],[100,-1]))
