import pygame
from pygame.locals import *
from random import randint
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import pylab

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from tensorflow import keras

from sklearn.model_selection import train_test_split


SIZE = 1200, 600

AGENT_COL = (255, 150, 0)


# screen
screet_red = randint(25,200)
screen_green = randint(25,200)
screen_blue = randint(25,200)
Screen_col = (screet_red, screen_green, screen_blue)

# font
pygame.font.init()
font1 = pygame.font.SysFont('arial.ttf', 24)


# movements
def move_up():
    global a_y
    a_y -= 2*game_speed*rect_speed
    return a_y
    
def move_down():
    global a_y
    a_y += 2*game_speed*rect_speed
    return a_y

def move_left():
    global a_x
    a_x -= 2*game_speed*rect_speed
    return a_x

def move_right():
    global a_x
    a_x += 2*game_speed*rect_speed
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

batch_size = 12
data_state = pd.DataFrame()

features = rg.random((1, 9))
weights = rg.random((1, 4))[0]
targets = np.random.choice([0,1], 5)
data_state = pd.DataFrame(features, columns=[
    "X_val_right", "X_val_left", "Y_val_down", "Y_val_up", "X_dang_right", "X_dang_left", "Y_dang_down", "Y_dang_up", "rect_speed"
    ])
# data_state["targets"] = targets

target_state = pd.DataFrame(targets).T

preds_ = [[1,0,0,0]]
y_pred =  [[1,0,0,0]]


def current_state():

    global data_state
    global target_state
    global rect_speed

        # current_state
    if rect_target.centerx > rect.centerx:
        X_targ_right = 1
        X_val_right = (rect.centerx - rect_target.centerx + 400) / 800
    else:
        X_targ_right = 0
        X_val_right = 0

    if rect_target.centerx < rect.centerx:
        X_targ_left = 1
        X_val_left = (rect_target.centerx - rect.centerx + 400) / 800
    else:
        X_targ_left = 0
        X_val_left = 0

    if rect_target.centerx == rect.centerx:
        X_val_right = 0
        X_val_left = 0


    if rect_target.centery > rect.centery:
        Y_targ_down = 1
        Y_val_down = (rect.centery - rect_target.centery + 300) / 600
    else:
        Y_targ_down = 0
        Y_val_down = 0

    if rect_target.centery < rect.centery:
        Y_targ_up = 1
        Y_val_up = (rect_target.centery - rect.centery + 300) / 600
    else:
        Y_targ_up = 0
        Y_val_up = 0

    if rect_target.centery == rect.centery:
        Y_val_down = 0
        Y_val_up = 0

    if rect_target.center == rect.center:
        X_val_right = 1.0
        X_val_left = 1.0
        Y_val_down = 1.0
        Y_val_up = 1.0

    # print('state within: ', data_state)


    if a_x >= 1100:
        X_dang_right = 1.0
        X_val_left = 1.0
    else:
        X_dang_right = 0
    
    if a_x <= 450:
        X_dang_left = 1.0
        X_val_right = 1.0
    else:
        X_dang_left = 0

    if a_y >= 500:
        Y_dang_down = 1.0
        Y_val_up = 1.0
    else:
        Y_dang_down = 0
    
    if a_y <= 50:
        Y_dang_up = 1.0
        Y_val_down = 1.0
    else:
        Y_dang_up = 0


    current_state = [X_val_right, X_val_left, Y_val_down, Y_val_up, X_dang_right, X_dang_left, Y_dang_down, Y_dang_up, rect_speed/10]

    data_state = data_state.append(pd.Series(current_state, index=data_state.columns[:len(current_state)]), ignore_index=True)

    if data_state.shape[0] >= batch_size:
        data_state = data_state.tail(4)
#        data_state['targets'].iloc[-4:] = [X_targ_right, X_targ_left, Y_targ_down, Y_targ_up]

    

    if X_val_right > X_val_left:
        X_spd = X_val_right
    elif X_val_right < X_val_left:
        X_spd = X_val_left
    else:
        X_spd = 1.0
    
    if Y_val_down > Y_val_up:
        Y_spd = Y_val_down
    elif Y_val_down < Y_val_up:
        Y_spd = Y_val_up
    else:
        Y_spd = 1.0

    targ_speed = 1.0 - ((X_spd + Y_spd)/2)
    

    curr_target = [X_targ_right, X_targ_left, Y_targ_down, Y_targ_up, targ_speed]
    print('targ_state: ', target_state)
    target_state = target_state.append(pd.Series(curr_target, index=target_state.columns[:len(curr_target)]), ignore_index=True)

    if target_state.shape[0] >= batch_size:
        target_state = target_state.tail(4)

    print('state: \n', data_state)
    return data_state, target_state


model = Sequential()
model.add(Dense(9, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(25, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(5, activation='softmax'))

loss_function_used = MeanSquaredError()

model.compile(loss=loss_function_used, optimizer=Adam(lr=0.01), metrics=['accuracy'])

evaluation = []

def train_keras():
    global data_state
    global target_state
    global evaluation
    global model

    ds_train, ds_test, ts_train, ts_test = train_test_split(data_state, target_state, test_size=0.25, shuffle=True)

    model.fit(ds_train, ts_train, epochs=1, batch_size=4, verbose=1, validation_split=0.2)

    test_results = model.evaluate(ds_test, ts_test, verbose=1)
    evaluation.append(test_results)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

def keras_pred():
    global data_state
    global y_pred

    current_state()

    if len(data_state)%4 == 0:

        train_keras()

        y_pred = model.predict(data_state)
        # k_pred = np.argmax(y_pred, axis=1)
        print('keras prediction: ', y_pred)
        return y_pred


# NN based moving
def NN_move():
    global rect_speed
    axis_check = randint(1,2)
    if axis_check == 1:
        if y_pred[-1][-2] == max(y_pred[-1][-3:-1]):
            move_up() 

        elif y_pred[-1][-3] == max(y_pred[-1][-3:-1]):
            move_down()
    
    elif axis_check == 2:

        if y_pred[-1][-4] == max(y_pred[-1][-5:-3]):
            move_left()

        elif y_pred[-1][-5] == max(y_pred[-1][-5:-3]):
            move_right()  

    if y_pred[-1][-1] <= 0.1:
        rect_speed = 0.5
    elif y_pred[-1][-1] > 0.1 and y_pred[-1][-1] <= 0.2:
        rect_speed = 1.0
    elif y_pred[-1][-1] > 0.2 and y_pred[-1][-1] <= 0.3:
        rect_speed = 2.0
    elif y_pred[-1][-1] > 0.4 and y_pred[-1][-1] <= 0.5:
        rect_speed = 3.0
    elif y_pred[-1][-1] > 0.5:
        rect_speed = 4.0



# Neuralnet drawing definition
def NN_draw(NN):
    c_x = 0
    c_y = 150
    y_space = 425
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
            line_color = 'darkgreen'
        elif rand_line_col ==1:
            line_color = 'yellow'
    elif line == 'inactive':
        line_color = 'darkgreen'

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
layer_in = [n for n in range(0,9)]
layer_h1 = [n for n in range(0,25)]
layer_h3 = [n for n in range(0,25)]
layer_out = [n for n in range(0,5)]
NN = layer_in,layer_h1,layer_h3,layer_out       
neuron_stage = 'inactive'


# draw chart
chart = True
def draw_chart(size):
    global surf
    if chart == True:
        #for score in score_:

        # df = pd.DataFrame(epoch_loss)
        df = pd.DataFrame(evaluation)
        df_plot = df.iloc[:,0].plot(kind="line", grid=True, ax=ax, title='Loss', xlabel='Score', color = "red")
        # ax.get_legend().remove()

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
df_plot = df.plot(kind="line", grid=True, ax=ax, title='Accuracy / Loss', xlabel='Score', color = [col/255 for col in AGENT_COL])
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
rect_speed = 1.0

# life lenght
_life_length = 500
life_length_ = [0]
ll = 1

# epoch
epoch = 1
# record list
record = [0]

#
# MODE OPTION
#
# control = 'manual'
# control = 'scripted_ai'
control = 'NN_based'

# presetup
average_loss = 0
loss = 1
preds_block = []


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
    life_length = _life_length
    running = 1


def fail():
    global rewards
    global running
    global epoch
    global score
    global model

    # score_.append(score)
    rewards -= 10
    running = 2
    epoch += 1
    draw_chart(size)

    # print(model.summary())


    if score >= max(record):
        model.save("record_model")

    elif score < max(record) :
        model = keras.models.load_model("record_model")

    else:
        pass


    record.append(score)
    score = 0



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
    life_length_.append(life_length)
    rewards += 10
    # running = 2
    # epoch += 1

    target_x = randint(450,1100)
    target_y = randint(50,500)
    rect_target = Rect(target_x, target_y, 55, 55)
    life_length = _life_length
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
                    screet_red = randint(25,200)
                    screen_green = randint(25,200)
                    screen_blue = randint(25,200)
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
            
                # train_model(data_state, bias, l_rate)
                keras_pred()


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
            epoch_text = font1.render('Try: ' + str(epoch) + ' ', True, 'white')
            screen.blit(epoch_text, (410, 15))

            # life lenght
            life_length -= ll
            life_length_count_text = font1.render('Life length remains: ' + str(life_length) + ' ', True, 'white')
            screen.blit(life_length_count_text, (410, 60))
            if life_length == 0:
                fail()

            # score
            score_text = font1.render('Score: ' + str(score) + ' ', True, 'white')
            screen.blit(score_text, (410, 45))

            # record
            record_text = font1.render('Record: ' + str(max(record)) + ' ', True, 'white')
            screen.blit(record_text, (410, 30))

            # loss
            if len(evaluation) > 1:
                LOSS_text = font1.render('Loss: ' + str(round(evaluation[-1][0], 4)) + ' ', True, 'white')
            else:
                LOSS_text = font1.render('Loss: ', True, 'white')
            screen.blit(LOSS_text, (410, 75))

            if len(evaluation) > 1:
                Acc_text = font1.render('Accuracy: ' + str(round(evaluation[-1][1], 4)) + ' ', True, 'white')
            else:
                Acc_text = font1.render('Accuracy: ', True, 'white')
            screen.blit(Acc_text, (410, 90))

            speed_text = font1.render('Speed: ' + str(rect_speed) + ' ', True, 'white')
            screen.blit(speed_text, (410, 105))


        pygame.display.update()
        clock.tick(60)

SimpleGame()