import numpy as np
import matplotlib.pyplot as plt
import math
shape = (32, 32)
Lx = shape[0]
Ly = shape[1]


# Spin block
block = np.random.random(size=shape)
# J/(kT)
Jcp = 0.01

liftx=int(Lx/2)
lifty=int(Ly/2)
# life variables initialization

#log distribution

def metropolis(b):
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp = np.random.random()
            bij = b[i][j]
            de = (
                        math.cos(2 * math.pi * (b[(i + Lx + 1) % Lx][j] - bij)) +
                        math.cos(2 * math.pi * (b[(i + Lx - 1) % Lx][j] - bij)) +
                        math.cos(2 * math.pi * (b[i][(j + Ly + 1) % Ly] - bij)) +
                        math.cos(2 * math.pi * (b[i][(j + Ly - 1) % Ly] - bij)) -
                        math.cos(2 * math.pi * (b[(i + Lx + 1) % Lx][j] - temp)) -
                        math.cos(2 * math.pi * (b[(i + Lx - 1) % Lx][j] - temp)) -
                        math.cos(2 * math.pi * (b[i][(j + Ly + 1) % Ly] - temp)) -
                        math.cos(2 * math.pi * (b[i][(j + Ly - 1) % Ly] - temp))
                 )
            if np.random.random() < math.exp(-Jcp*de):
                b[i][j] = temp

def Randln():
    temp = np.random.random()
    temp = -math.log(temp)/Jcp
    return temp
def EC(b,liftx,lifty):
    E = np.array([1.0,1.0,1.0,1.0])
    for i in range(shape[0]):
        for j in range(shape[1]):
            event=b[(liftx + Lx + 1) % Lx][lifty]
            temp = (1 + event - b[liftx][lifty])%1
            r=Randln()
            if (temp< 0.5) :
                E[0] = temp + int(r/2) + math.acos(1-r%2)/(2*math.pi)
            else:
                temp = 1 - temp
                r = r + 1 - math.cos(2*math.pi*temp)
                E[0] = int(r/2) + math.acos(1-r%2)/(2*math.pi) - temp

            event=b[(liftx + Lx - 1) % Lx][lifty]
            temp = (1 + event - b[liftx][lifty])%1
            r=Randln()
            if (temp< 0.5) :
                E[1] = temp + int(r/2) + math.acos(1-r%2)/(2*math.pi)
            else:
                temp = 1 - temp
                r = r + 1 - math.cos(2*math.pi*temp)
                E[1] = int(r/2) + math.acos(1-r%2)/(2*math.pi) - temp

            event=b[liftx][(Ly + lifty + 1) % Ly]
            temp = (1 + event - b[liftx][lifty])%1
            r=Randln()
            if (temp< 0.5):
                E[2] = temp + int(r/2) + math.acos(1-r%2)/(2*math.pi)
            else:
                temp = 1 - temp
                r = r + 1 - math.cos(2*math.pi*temp)
                E[2] = int(r/2) + math.acos(1-r%2)/(2*math.pi) - temp

            event=b[liftx][(Ly + lifty - 1) % Ly]
            temp = (1 + event - b[liftx][lifty])%1
            r=Randln()
            if (temp< 0.5) :
                E[3] = temp + int(r/2) + math.acos(1-r%2)/(2*math.pi)
            else:
                temp = 1 - temp
                r = r + 1 - math.cos(2*math.pi*temp)
                E[3] = int(r/2) + math.acos(1-r%2)/(2*math.pi) - temp

            b[liftx][lifty] = (b[liftx][lifty] + E.min())%1
            minx=E.argmin()
            if(minx==0):
                liftx=(liftx+1)%shape[0]
            if(minx==1):
                liftx=(liftx-1+Lx)%shape[0]
            if(minx==2):
                lifty=(lifty+1)%shape[1]
            if(minx==3):
                lifty=(lifty-1+Ly)%shape[1]
plt.ion()
plt.show()

im = plt.imshow(block, cmap='hsv', vmin=0, vmax=1, interpolation='none')
t = 0
for i in range(10):
    EC(block,liftx,lifty)
    im.set_data(block)
    plt.draw()
    plt.pause(.0000001)
    t = t + 1
    print(t)
Jcp = 50
while True:
    EC(block,liftx,lifty)
    im.set_data(block)
    plt.draw()
    plt.pause(.0000001)
    t = t+1
    print(t)
