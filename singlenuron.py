import random
import math

TEACH_NUM = 4    
INP_NUM   = 2    
alpha     = 0.01 

w = []; dw = []
theta = 0 
d_theta = 0

teach_x = [[0,0],[0,1],[1,0],[1,1]]
teach_y = [0,1,1,1]

def init_w ():                  
    global w,theta
    for i in range (INP_NUM):
        w.append(random.random() * 2 - 1 )
    theta = random.random() * 2 - 1 

def sigmoid (x):
    return 1.0 / ( 1.0 + math.exp(-x)) 

def forward (x):
    u = 0
    for i in range(INP_NUM):
        u += w[i] * x[i]
    u += theta         
    return sigmoid(u)

def func_error ():    
    e = 0
    for t in range(TEACH_NUM):
        y = forward(teach_x[t])
    e += 0.5 * (y - teach_y[t]) * (y - teach_y[t])
    return e

def clear_dw ():       
    global dw,d_theta
    for i in range(INP_NUM):
        dw.append(0)
    d_theta = 0

def calc_dw (x_t, y_hat):
    global dw,d_theta
    y = forward(x_t)
    dy = y * (1-y)
    for i in range(INP_NUM): 
        dw[i] += (y - y_hat) * dy * x_t[i]
    d_theta += (y - y_hat) * dy

def update_w ():
    global w,theta
    for i in range(INP_NUM): 
        w[i] -= alpha * dw[i]
    theta -= alpha * d_theta

if __name__ == '__main__':                

    init_w()  

    for loop in range(100000):
        if loop % 1000 == 0:
            print(f'{loop:6d}'+' '+str(func_error()))

        clear_dw()
        for t in range(TEACH_NUM): calc_dw(teach_x[t], teach_y[t])
        update_w()
    print(f'{loop:6d}'+' '+str(func_error()))     

    for t in range(TEACH_NUM):
        y=forward(teach_x[t])
        print(str(t)+' y='+str(y)+' <--> y_hat = '+str(teach_y[t]))