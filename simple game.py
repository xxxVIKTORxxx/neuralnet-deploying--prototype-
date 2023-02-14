import pygame
from pygame.locals import *
from random import randint

SIZE = 1200, 600

RED = (255, 0, 0)


# screen
screet_red = randint(1,225)
screen_green = randint(1,225)
screen_blue = randint(1,255)
Screen_col = (screet_red, screen_green, screen_blue)




# Neuralnet drawing definition
def NN_draw(NN):
    c_x = 0
    c_y = -50
    y_space = 600
    x_space = 375
    distance_y = y_space / len(NN)

    radius = 15

    #coordinates
    nn_coords = []
    for layer in NN:
        n_num = 0
        c_y += distance_y
        distance_x = x_space / len(layer)
        c_x = -distance_x + 50
        nn_layer_coords = []
        nn_coords.append(nn_layer_coords)
        for neuron in layer:
            radius = 15
            if distance_x < 30:
                radius = (distance_x - radius)*1.33
            n_num +=1
            c_x += distance_x
            nn_layer_coords.append([c_x, c_y])

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
                pygame.draw.line(screen, line_color, coord, next_coord, width = 3)
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
    for layer in nn_coords:
        for coord in layer:
            pygame.draw.circle(screen, 'green', coord, radius)
            pygame.draw.circle(screen, neuron_color, coord, radius-5)




""" i=0
    for layer_coords in nn_coords:

        for coord in layer_coords:
            pygame.draw.lines(screen, 'white', coord, [coord for coord in [
                layer_coords for layer_coords in nn_coords[i]
                ]])
        i-=1    
        """



layer_in = [n for n in range(0,4)]
layer_h1 = [n for n in range(0,5)]
layer_h2 = [n for n in range(0,3)]
layer_out = [n for n in range(0,4)]
NN = layer_in,layer_h1,layer_h2,layer_out       
neuron_stage = 'inactive'




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

# to delete
print(f'x={rect.x}, y={rect.y}, w={rect.w}, h={rect.h}')
print(f'left={rect.left}, top={rect.top}, right={rect.right}, bottom={rect.bottom}')
print(f'center={rect.center}')

# scores
score = 0
rewards = 0


# while True
running = 2
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit()

        if running == 2:
            keys = pygame.key.get_pressed()    
            if keys[pygame.K_SPACE]:
                running = 1
                a_x = randint(420,1100)
                a_y = randint(50,500)
                target_x = randint(450,1100)
                target_y = randint(50,500)
                rect = Rect(a_x, a_y, 100, 100)
                rect_target = Rect(target_x, target_y, 55, 55)


    if running == 2:
        if int(pygame.time.get_ticks() / 100) % 2 == 0:
                screet_red = randint(1,225)
                screen_green = randint(1,225)
                screen_blue = randint(1,255)
                Screen_col = (screet_red, screen_green, screen_blue)
        screen.fill(Screen_col)

        
    elif running == 1:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            a_x += 2

        if keys[pygame.K_LEFT]:
            a_x -= 2

        if keys[pygame.K_UP]:
            a_y -= 2

        if keys[pygame.K_DOWN]:
            a_y += 2

        if a_x > 1150 or a_x < 400 or a_y > 550 or a_y < 0:
            score -= 1
            rewards -= 10
            running = 2



        if rect_target.contains(rect):
            print('Target Achieved!')
            score += 1
            rewards += 10
            running = 2

        #right side screen
        screen.fill(Screen_col)
        rect = Rect(a_x, a_y, 50, 50)
        pygame.draw.rect(screen, 'black', action_screen, 10)
        pygame.draw.rect(screen, 'green', rect_target, 5)
        pygame.draw.rect(screen, RED, rect)
        pygame.draw.rect(screen, 'black', NN_screen)
        
        # left side screen
        neuron_stage = 'inactive'
        NN_draw(NN)


    pygame.display.update()
    clock.tick(60)


pygame.quit()