import pygame
from pygame.locals import *
from random import randint
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import pylab

#from simple_game_play_NN_agent import Agent



SIZE = 1200, 600

AGENT_COL = (255, 150, 0)


# screen
screet_red = randint(25,225)
screen_green = randint(25,225)
screen_blue = randint(25,255)
Screen_col = (screet_red, screen_green, screen_blue)

# font
pygame.font.init()
font1 = pygame.font.SysFont('arial.ttf', 24)


# movements
def move_up():
    global a_y
    a_y -= 2*game_speed
    return a_y
    
def move_down():
    global a_y
    a_y += 2*game_speed
    return a_y

def move_left():
    global a_x
    a_x -= 2*game_speed
    return a_x

def move_right():
    global a_x
    a_x += 2*game_speed
    return a_x


# scripted AI
def ai_move():

    if rect_target.center != rect.center:

        check = randint(1,2)

        if check == 1:
            if rect_target.centerx > rect.centerx:
                move_right()
            elif rect_target.centerx < rect.centerx:
                move_left()
            
        elif check == 2:
            if rect_target.centery > rect.centery:
                move_down()
            elif rect_target.centery < rect.centery:
                move_up()


# simple perceptron

rg = np.random.default_rng()

batch_size = 4
data_state = pd.DataFrame()

features = rg.random((4, 4))
weights = rg.random((1, 4))[0]
targets = np.random.choice([0,1], 4)
data_state = pd.DataFrame(features, columns=["x0", "x1", "x2", "x3"])
data_state["targets"] = targets

preds_ = [1,0,0,0]


def current_state():

    global data_state
        # current_state
    if rect_target.centerx > rect.centerx:
        X_targ_right = 1
        X_val = (rect.centerx - rect_target.centerx + 400) / 800
    else:
        X_targ_right = 0

    if rect_target.centerx < rect.centerx:
        X_targ_left = 1
        X_val = (rect_target.centerx - rect.centerx + 400) / 800
    else:
        X_targ_left = 0

    if rect_target.centerx == rect.centerx:
        X_val = 0


    if rect_target.centery > rect.centery:
        Y_targ_down = 1
        Y_val = (rect.centery - rect_target.centery + 400) / 800
    else:
        Y_targ_down = 0

    if rect_target.centery < rect.centery:
        Y_targ_up = 1
        Y_val = (rect_target.centery - rect.centery + 400) / 800
    else:
        Y_targ_up = 0

    if rect_target.centery == rect.centery:
        Y_val = 0

    print(X_val, Y_val)

    if a_x >= 1100 or a_x < 450:
        X_dang = 1
    else:
        X_dang = 0

    if a_y >= 500 or a_y < 50:
        Y_dang = 1
    else:
        Y_dang = 0

    current_state = [X_val, Y_val, X_dang, Y_dang]

    data_state.append(current_state)


    data_state["targets"] = [X_targ_left, X_targ_right, Y_targ_down, Y_targ_up]
    return data_state

bias = 0.5
l_rate = 0.1
epoch_loss = []


def get_weighted_sum(feature, weights, bias):
    return np.dot(feature, weights) + bias

def sigmoid(w_sum):
    return 1/(1+np.exp(-w_sum))

def cross_entropy(target, prediction):
    return -(target*np.log10(prediction) + (1-target)*np.log10(1-prediction))

def update_weights(weights, l_rate, target, prediction, feature):
    new_weights = []
    for x, w in zip(feature, weights):
        new_w = w + l_rate*(target-prediction)*x
        new_weights.append(new_w)
    return new_weights

def update_bias(bias, l_rate, target, prediction):
    return bias + l_rate*(target-prediction)

def train_model(data_state, weights, bias, l_rate):
    global preds_
    global average_loss
    current_state()
    # for e in range(epochs):
    individual_loss = []
    for i in range(len(data_state)):
        feature = data_state.loc[i][:-1]
        target = data_state.loc[i][-1]
        print('feature: \n', feature)
        print('target: \n', target)
        w_sum = get_weighted_sum(feature, weights, bias)
        prediction = sigmoid(w_sum)

        preds_.append(prediction)
        print('pred: ', preds_)
        if len(preds_)>=8:
            preds_ = preds_[-4:-1]

        loss = cross_entropy(target, prediction)
        print('LOSS: ', loss)

        individual_loss.append(loss)
        # gradient descent
        weights = update_weights(weights, l_rate, target, prediction, feature)
        bias = update_bias(bias, l_rate, target, prediction)


    average_loss = sum(individual_loss)/len(individual_loss)
    epoch_loss.append(average_loss)
    print("**************************")
    print(average_loss)

# NN based moving
def NN_move():

    if preds_[-1] == max(preds_[-4:-1]):
        move_right()

    elif preds_[-2] == max(preds_[-4:-1]):
        move_left()

    elif preds_[-3] == max(preds_[-4:-1]):
        move_down()

    elif preds_[-4] == max(preds_[-4:-1]):
        move_up()  
    


# Neuralnet drawing definition
def NN_draw(NN):
    c_x = 0
    c_y = 50
    y_space = 500
    x_space = 375
    distance_y = y_space / len(NN)

    radius = 15

    #coordinates
    nn_coords = []
    radiuses = []
    for layer in NN:
        n_num = 0
        c_y += distance_y
        distance_x = x_space / len(layer)
        c_x = -distance_x + distance_x/2 + 10
        nn_layer_coords = []
        nn_coords.append(nn_layer_coords)
        for neuron in layer:
            radius1 = 10
            if distance_x < 30:
                radius1 = (distance_x/radius1)
            n_num +=1
            c_x += distance_x
            nn_layer_coords.append([c_x, c_y])
            radiuses.append(radius1)

    # line activity
    line = 'active'
    rand_line_col = 0
    if line == 'active':
        if int(pygame.time.get_ticks() / 100) % 3 == 0:
            rand_line_col = randint(0,1)
        if rand_line_col == 0:
            line_color = 'darkblue'
        elif rand_line_col ==1:
            line_color = 'yellow'
    elif line == 'inactive':
        line_color = 'darkblue'

    #lines drawing
    i=0
    while i < len(nn_coords)-1:
        for coord in nn_coords[i]:
            for next_coord in nn_coords[i+1]:
                pygame.draw.line(screen, line_color, coord, next_coord, width = 1)
        i+=1

    #   neuron activity
    rand_neuron_act = 0
    if int(pygame.time.get_ticks() / 100) % 3 == 0:
        rand_neuron_act = randint(0,1)
    if rand_neuron_act == 0:
        neuron_color = 'red'
    elif rand_neuron_act == 1:
        neuron_color = 'yellow'

    # neurons drawing
    i=0
    for layer in nn_coords:
        for coords in layer:
            pygame.draw.circle(screen, 'green', coords, [radius for radius in radiuses][i]+2)
            pygame.draw.circle(screen, neuron_color, coords, [radius for radius in radiuses][i])
            i+=1



# neuralnet example to be drawn
layer_in = [n for n in range(0,4)]
#layer_h1 = [n for n in range(0,25)]
layer_out = [n for n in range(0,4)]
NN = layer_in,layer_out       
neuron_stage = 'inactive'


# draw chart
chart = True
def draw_chart(size):
    global surf
    if chart == True:
        #for score in score_:

        df = pd.DataFrame(epoch_loss)
        df_plot = df.plot(kind="line", grid=True, ax=ax, title='AVG LOSS', xlabel='Score', color = [col/255 for col in AGENT_COL])
        ax.get_legend().remove()

        canvas.draw()
        raw_data = renderer.tostring_rgb()

        surf = pygame.image.fromstring(raw_data, size, "RGB")

        screen.blit(surf, (10,10))


font = {'size'   : 8}

matplotlib.rc('font', **font)

fig = pylab.figure(figsize=[3.8, 1.8], # Inches
                   dpi=100,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
                   )
ax = fig.gca()
life_length_ = [0]
df = pd.DataFrame(life_length_)
df_plot = df.plot(kind="line", grid=True, ax=ax, title='AVG LOSS', xlabel='Score', color = [col/255 for col in AGENT_COL])
ax.get_legend().remove()
ax.set_facecolor('black')
ax.title.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
fig.patch.set_facecolor('black')

canvas = agg.FigureCanvasAgg(fig)
canvas.draw()
renderer = canvas.get_renderer()
raw_data = renderer.tostring_rgb()
size = canvas.get_width_height()
surf = pygame.image.fromstring(raw_data, size, "RGB")


###

pygame.init()

screen = pygame.display.set_mode(SIZE)
action_screen = Rect(400, 0, 800, 600)
NN_screen = Rect(0, 0, 400, 600)


clock = pygame.time.Clock()


# Rects
a_x = randint(425,1100)
a_y = randint(25,500)
target_x = randint(425,1100)
target_y = randint(25,500)
rect = Rect(a_x, a_y, 50, 50)
rect_target = Rect(target_x, target_y, 55, 55)

#
# PARAMETERS

# scores
score = 0
score_ = []
rewards = 0
game_speed = 1

# life lenght
life_length = 1000
life_length_ = [0]
ll = 1

# epoch
epoch = 0

#
# MODE OPTION
#
# control = 'manual'
# control = 'scripted_ai'
control = 'NN_based'

# presetup
average_loss = 0
loss = 1


def start():
    global a_x
    global a_y
    global target_x
    global target_y
    global rect
    global rect_target
    global life_length
    global running
    a_x = randint(420,1100)
    a_y = randint(50,500)
    target_x = randint(450,1100)
    target_y = randint(50,500)
    rect = Rect(a_x, a_y, 100, 100)
    rect_target = Rect(target_x, target_y, 55, 55)
    life_length = 1000
    running = 1

def fail():
    global rewards
    global running
    global epoch
    rewards -= 10
    running = 2
    epoch += 1
    draw_chart(size)

def succeed():
    global score
    global rewards
    global running
    global epoch
    global target_x
    global target_y
    global life_length
    global rect_target

    score += 1
    score_.append(score)
    life_length_.append(life_length)
    rewards += 10
    # running = 2
    epoch += 1

    target_x = randint(450,1100)
    target_y = randint(50,500)
    rect_target = Rect(target_x, target_y, 55, 55)
    life_length = 1000
    draw_chart(size)



# while True
running = 2

def SimpleGame():
    global rect
    global a_x
    global a_y
    global life_length
    global Screen_col



    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()

            if running == 2:
                keys = pygame.key.get_pressed()    
                if keys[pygame.K_SPACE]:
                    start()


        if running == 2:

                
            if int(pygame.time.get_ticks() / 100) % 2 == 0:
                    screet_red = randint(25,225)
                    screen_green = randint(25,225)
                    screen_blue = randint(25,255)
                    Screen_col = (screet_red, screen_green, screen_blue)

            if control == 'scripted_ai' or control == 'NN_based':
                if int(pygame.time.get_ticks() / 1000) % 3 == 0:
                    start()


            screen.fill(Screen_col)

            
        elif running == 1:
            
            
            # controller
            if control == 'manual':
                keys = pygame.key.get_pressed()
                if keys[pygame.K_RIGHT]:
                    a_x += 2*game_speed

                if keys[pygame.K_LEFT]:
                    a_x -= 2*game_speed

                if keys[pygame.K_UP]:
                    a_y -= 2*game_speed

                if keys[pygame.K_DOWN]:
                    a_y += 2*game_speed

            elif control == 'scripted_ai':
                ai_move()

            elif control == 'NN_based':
                NN_move()
            
                train_model(data_state, weights, bias, l_rate)


            # goal
            if rect_target.contains(rect):
                print('Target Achieved!')
                succeed()
                

            #right side screen
            screen.fill(Screen_col)
            rect = Rect(a_x, a_y, 50, 50)
            pygame.draw.rect(screen, 'black', action_screen, 10)
            pygame.draw.rect(screen, 'green', rect_target, 5)
            pygame.draw.rect(screen, AGENT_COL, rect)
            pygame.draw.rect(screen, 'black', NN_screen)
            
            # left side screen
            neuron_stage = 'active'
            NN_draw(NN)
            
            #draw_chart(size)
            screen.blit(surf, (10,10))

            #
            # limits
            if a_x > 1150 or a_x < 400 or a_y > 550 or a_y < 0:
                fail()

            # TEXTS

            # epoch
            epoch_text = font1.render('Epoch: ' + str(epoch) + ' ', True, 'white')
            screen.blit(epoch_text, (410, 15))

            # life lenght
            life_length -= ll
            life_length_count_text = font1.render('Life length remains: ' + str(life_length) + ' ', True, 'white')
            screen.blit(life_length_count_text, (410, 30))
            if life_length == 0:
                fail()

            
            # score
            score_text = font1.render('Score: ' + str(score) + ' ', True, 'white')
            screen.blit(score_text, (410, 45))

            # loss
            LOSS_text = font1.render('AVG LOSS: ' + str(average_loss) + ' ', True, 'white')
            screen.blit(LOSS_text, (410, 60))


        pygame.display.update()
        clock.tick(60)

SimpleGame()