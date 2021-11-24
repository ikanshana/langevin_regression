import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

def f(p):
    time.sleep(10)
    p.put('yolo')
    time.sleep(10)
    p.put(3)

p = mp.Queue()
q = mp.Queue()
r = mp.Queue()

I = [0,0,0]
P = [p,q,r]
if __name__ == '__main__':
    for i in range(3):
        I[i] = mp.Process(target=f,args=(P[i],))
        I[i].start()

        print('Fait !')

    print(P[0].get())
    print("Oh la la c'est long !")
    try:
        print(P[0].get(block=False))
    except:
        print("Hey ! Mais c'est trop long !")

    for i in range(3):
        print('test')
        I[i].join()
