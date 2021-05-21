import pygame
import sys
import random
import time
import math
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
pygame.init()

enable_plot = False
social_distancing_ratio = .7
#consts
WIDTH,HEIGHT = (800,800)
#end consts

HEALTHY_COLOR = (0,251,60)
INFECTED_COLOR = (225,26,13)
RECOVERED_COLOR = (13,32,244)
DECEASED_COLOR = (43,13,23)

total_healthy = 0
total_infected = 0

#setup
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Social Distancing Simulation -- Justin Stitt")
background_color = (42,42,44)
clock = pygame.time.Clock()
font = pygame.font.Font('freesansbold.ttf', 32)
sfont = pygame.font.Font('freesansbold.ttf', 16)

text = font.render('time: (0)',True,(255,255,255))
sd_text = sfont.render('Social Distancing Ratio: {}'.format(social_distancing_ratio),True,(255,255,255))
fps = 60
frame = 1
#end setup
balls = []
ball_size = 5
ball_speed = 2
ball_max_speed = 5

selected_ball = None

class Ball:
    def __init__(self):
        self.pos = [0,0]
        self.v = [0,0]
        self.r = ball_size
        self.mass = 1
        self.i = -1
        self.state = 0 #0 = healthy, 1 = infected, 2 = recovered, 3 = deceased
        self.moving = True
        #display
        self.color = HEALTHY_COLOR#default color
    def update(self):
        self.check_border()
        if(self.moving == True):
            self.calc_movement()
        self.update_color()
        self.render()
    def render(self):
        if(math.isnan(self.pos[0]) or math.isnan(self.pos[1])):
            return
        pygame.draw.circle(screen,self.color,(int(self.pos[0]),int(self.pos[1])),self.r)
    def check_border(self):
        if(self.pos[0] - self.r <= 0):
            self.pos[0] = 0 + self.r + 2
            self.v[0] *= -1
        elif(self.pos[0] + self.r >= WIDTH):
            self.pos[0] = WIDTH - self.r - 2
            self.v[0] *= -1
        if(self.pos[1] - self.r <= 0):
            self.pos[1] = 0 + self.r + 2
            self.v[1] *= -1
        elif(self.pos[1] + self.r >= HEIGHT):
            self.pos[1] = HEIGHT - self.r - 2
            self.v[1] *= -1
    def calc_movement(self):
        #update ball physics
        if(self.v[0] > ball_max_speed):
            self.v[0] = ball_max_speed
        elif(self.v[0] < -ball_max_speed):
            self.v[0] = -ball_max_speed
        if(self.v[1] > ball_max_speed):
            self.v[1] = ball_max_speed
        elif(self.v[1] < -ball_max_speed):
            self.v[1] = -ball_max_speed
        self.pos[0] += self.v[0]
        self.pos[1] += self.v[1]
    def update_color(self):
        if self.state == 0:
            self.color = HEALTHY_COLOR
        elif self.state == 1:
            self.color = INFECTED_COLOR
        elif self.state == 2:
            self.color = RECOVERED_COLOR
        elif self.state == 3:
            self.color = DECEASED_COLOR



collidingPairs = []

def check_static():
    for ball in balls:
        for target in balls:
            if(ball.i != target.i):
                if(overlap(*ball.pos,ball.r,*target.pos,target.r)):
                    #collision has occured
                    collidingPairs.append([ball,target])

                    #INFECTION
                    if(ball.state == 1 and target.state != 1):#infected
                        target.state = 1#now target is infected
                    if(target.state == 1 and ball.state != 1):
                        ball.state = 1
                    #END INFECTION

                    #distance between ball centers
                    distance = math.sqrt((ball.pos[0] - target.pos[0])**2 + (ball.pos[1] - target.pos[1])**2)
                    #calculate displacement required
                    _overlap = .5 * (distance - ball.r - target.r)
                    #displace current ball away from collision
                    if(distance == 0):
                        distance = 0.01
                    ball.pos[0] -= _overlap * (ball.pos[0] - target.pos[0]) / distance#normalize
                    ball.pos[1] -= _overlap * (ball.pos[1] - target.pos[1]) / distance#normalize
                    #displace target ball away from collision
                    target.pos[0] += _overlap * (ball.pos[0] - target.pos[0]) / distance#normalize
                    target.pos[1] += _overlap * (ball.pos[1] - target.pos[1]) / distance#normalize

def check_dynamic():
    for pair in collidingPairs:
        b1 = pair[0]
        b2 = pair[1]
        #distance between balls
        distance = math.sqrt((b1.pos[0] - b2.pos[0])**2 + (b1.pos[1] - b2.pos[1])**2);
        #normal
        if(distance == 0):
            distance = 0.01
        nx = (b2.pos[0] - b1.pos[0]) / distance
        ny = (b2.pos[1] - b1.pos[1]) / distance
        #tangent
        tx = -ny
        ty = nx
        #dot product tangent
        dpTan1 = b1.v[0] * tx * b1.v[1] * ty
        dpTan2 = b2.v[0] * tx + b2.v[1] * ty
        #dot product normal
        dpNorm1 = b1.v[0] * nx + b1.v[1] * ny
        dpNorm2 = b2.v[0] * nx + b2.v[1] * ny
        #conservation of momentum in 1D
        m1 = (dpNorm1 * (b1.mass - b2.mass) + 2.0 * b2.mass * dpNorm2) / b1.mass #+ b2.mass)
        m2 = (dpNorm2 * (b2.mass - b1.mass) + 2.0 * b1.mass * dpNorm1) / b1.mass #+ b2.mass)

        #update ball velocities
        b1.v[0] = tx * dpTan1 + nx * m1
        b1.v[1] = ty * dpTan1 + ny * m1
        b2.v[0] = tx * dpTan2 + nx * m2
        b2.v[1] = ty * dpTan2 + ny * m2
    collidingPairs.clear()



def add_ball(x,y,r = 50,v = [1,1]):
    _ball = Ball()
    _ball.pos = [x,y]
    _ball.radius = r
    _ball.mass = r * 5
    _ball.v = v
    _ball.i = len(balls)
    balls.append(_ball)
#aux funcs
def overlap(x1,y1,r1,x2,y2,r2):
    return (((x1-x2)**2 + (y1-y2)**2) <= (r1+r2)**2)

def isPointInCircle(x1,y1,r1,px,py):
    return abs((x1 - px)**2 + (y1 - py)**2) < (r1 * r1);

def rand_balls(num):
    for x in range(num):
        x_v = ball_speed if random.random() < 0.5 else -ball_speed
        y_v = ball_speed if random.random() < 0.5 else -ball_speed
        add_ball(random.randint(ball_size*2,WIDTH - ball_size*2),random.randint(ball_size*2,HEIGHT-ball_size*2),ball_size,[x_v,y_v])

def set_infected(percentage):
    for ball in balls:
        if(random.random() < percentage):
            ball.state = 1#set infected

def social_distance(percentage):
    #what percent of the population choses to social distance?
    for ball in balls:
        if(random.random() < percentage):
            ball.moving = False
            ball.v = [0,0]
def get_healthy():
    _count = 0
    for ball in balls:
        if ball.state == 0:
            _count += 1
    return _count
def get_infected():
    _count = 0
    for ball in balls:
        if ball.state == 1:
            _count += 1
    return _count
def get_recovered():
    _count = 0
    for ball in balls:
        if ball.state == 2:
            _count += 1
    return _count
def get_deceased():
    _count = 0
    for ball in balls:
        if ball.state == 3:
            _count += 1
    return _count
#end aux

#add_ball(200,250,10,[5,-2])
#add_ball(100,125,10,[0,0])
rand_balls(150)
set_infected(.1)
social_distance(social_distancing_ratio)

#plot setup
if(enable_plot):
    old_infected = get_infected()
    #x = np.array([frame])
    #y = np.array([old_infected])
    x = np.array([])
    y1 = np.array([])#infected
    y2 = np.array([])#healthy
    plt.rcParams['animation.html'] = 'jshtml'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = ["Healthy","Infected"]
    pallette = ["#39cc40","#e32749"]

    fig.show()
#end plot setup

def plot():
    global x,y1,y2, old_infected
    current_infected = get_infected()
    current_healthy = get_healthy()
    new_infected = current_infected - old_infected
    if(current_infected >= len(balls)):
        return
    x = np.append(x,frame)
    y1 = np.append(y1,current_infected)
    y2 = np.append(y2,current_healthy)
    old_infected = current_infected

    ax.set_xlabel('days')
    ax.set_ylabel('# of people')
    ax.stackplot(x,y2,y1, labels=labels, colors = pallette, alpha = 0.4)
    ax.legend(loc='upper left')
    fig.canvas.draw()
    ax.cla()

def update():
    global selected_ball
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    #update timer for text
    _time = frame
    text = font.render('days: {}'.format(_time),True,(255,255,255))
    screen.blit(text,(15,HEIGHT-45))
    #obj updates
    for ball in balls:
        ball.update()
    check_static()
    check_dynamic()
    total_healthy = get_healthy()
    total_infected = get_infected()
    h_text = font.render('Healthy: {}'.format(total_healthy),True,HEALTHY_COLOR)
    i_text = font.render('Infected: {}'.format(total_infected),True,INFECTED_COLOR)
    screen.blit(h_text,(555,HEIGHT-90))
    screen.blit(i_text,(555,HEIGHT-40))
def render():
    pass

while True:
    screen.fill(background_color)
    update()
    render()
    screen.blit(sd_text,(245,HEIGHT-20))
    pygame.display.flip()
    frame+=1
    if(frame % 3 == 0 and enable_plot):
        plot()
    clock.tick(fps)