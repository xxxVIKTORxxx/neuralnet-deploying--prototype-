import pygame
from pygame.locals import *
from random import randint

SIZE = 1200, 600

RED = (255, 0, 0)


# screen
screet_red = randint(25,225)
screen_green = randint(25,225)
screen_blue = randint(25,255)
Screen_col = (screet_red, screen_green, screen_blue)

# font
pygame.font.init()
font1 = pygame.font.SysFont('chalkduster.ttf', 24)

# life lenght
life_length = 1000
ll = 1

# epoch
epoch = 0


# scripted AI
def ai_move():
    #print(a_x, a_y)
    global a_x
    global a_y

    if rect_target.center != rect.center:

        check = randint(1,2)

        if check == 1:
            if rect_target.centerx > rect.centerx:
                a_x += 2
                print(a_x)
                return a_x
            elif rect_target.centerx < rect.centerx:
                a_x -= 2
                print(a_x)
                return a_x
            
        elif check == 2:
            if rect_target.centery > rect.centery:
                a_y += 2
                return a_y
            elif rect_target.centery < rect.centery:
                a_y -= 2
                return a_y

    
    


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
                radius1 = (distance_x - radius1)
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
                pygame.draw.line(screen, line_color, coord, next_coord, width = 2)
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
layer_in = [n for n in range(0,10)]
layer_h1 = [n for n in range(0,30)]
layer_h2 = [n for n in range(0,35)]
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
                life_length = 1000


    if running == 2:
        if int(pygame.time.get_ticks() / 100) % 2 == 0:
                screet_red = randint(25,225)
                screen_green = randint(25,225)
                screen_blue = randint(25,255)
                Screen_col = (screet_red, screen_green, screen_blue)
        screen.fill(Screen_col)

        
    elif running == 1:

        # controller
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            a_x += 2

        if keys[pygame.K_LEFT]:
            a_x -= 2

        if keys[pygame.K_UP]:
            a_y -= 2

        if keys[pygame.K_DOWN]:
            a_y += 2


        # goal
        if rect_target.contains(rect):
            print('Target Achieved!')
            score += 1
            rewards += 10
            running = 2
            epoch += 1

        #right side screen
        screen.fill(Screen_col)
        rect = Rect(a_x, a_y, 50, 50)
        pygame.draw.rect(screen, 'black', action_screen, 10)
        pygame.draw.rect(screen, 'green', rect_target, 5)
        pygame.draw.rect(screen, RED, rect)
        pygame.draw.rect(screen, 'black', NN_screen)
        
        # left side screen
        neuron_stage = 'active'
        NN_draw(NN)

        ai_move()


        # epoch
        epoch_text = font1.render('Epoch: ' + str(epoch) + ' ', True, 'white')
        screen.blit(epoch_text, (410, 15))


        #
        # limits
        if a_x > 1150 or a_x < 400 or a_y > 550 or a_y < 0:
            score -= 1
            rewards -= 10
            running = 2
            epoch += 1

        # life lenght
        life_length -= ll
        life_length_count_text = font1.render('Life length remains: ' + str(life_length) + ' ', True, 'white')
        screen.blit(life_length_count_text, (410, 30))
        if life_length == 0:
            score -= 1
            rewards -= 10
            running = 2
            epoch += 1
        
        # score
        score_text = font1.render('Score: ' + str(score) + ' ', True, 'white')
        screen.blit(score_text, (410, 45))

        # rewards
        rewards_text = font1.render('Rewards: ' + str(rewards) + ' ', True, 'white')
        screen.blit(rewards_text, (410, 60))


    pygame.display.update()
    clock.tick(60)


pygame.quit()