# Import a library of functions called 'pygame'
import pygame
from math import pi

import numpy as np

global currentX
global currentY
global currentZ
global currentAngle

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y


class Point3D:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z


class Line3D():

    def __init__(self, start, end):
        self.start = start
        self.end = end


def loadOBJ(filename):

    vertices = []
    indices = []
    lines = []

    f = open(filename, "r")
    for line in f:
        t = str.split(line)
        if not t:
            continue
        if t[0] == "v":
            vertices.append(Point3D(float(t[1]),float(t[2]),float(t[3])))

        if t[0] == "f":
            for i in range(1,len(t) - 1):
                index1 = int(str.split(t[i],"/")[0])
                index2 = int(str.split(t[i+1],"/")[0])
                indices.append((index1,index2))

    f.close()

    #Add faces as lines
    for index_pair in indices:
        index1 = index_pair[0]
        index2 = index_pair[1]
        lines.append(Line3D(vertices[index1 - 1],vertices[index2 - 1]))

    #Find duplicates
    duplicates = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]

            # Case 1 -> Starts match
            if line1.start.x == line2.start.x and line1.start.y == line2.start.y and line1.start.z == line2.start.z:
                if line1.end.x == line2.end.x and line1.end.y == line2.end.y and line1.end.z == line2.end.z:
                    duplicates.append(j)
            # Case 2 -> Start matches end
            if line1.start.x == line2.end.x and line1.start.y == line2.end.y and line1.start.z == line2.end.z:
                if line1.end.x == line2.start.x and line1.end.y == line2.start.y and line1.end.z == line2.start.z:
                    duplicates.append(j)

    duplicates = list(set(duplicates))
    duplicates.sort()
    duplicates = duplicates[::-1]

    #Remove duplicates
    for j in range(len(duplicates)):
        del lines[duplicates[j]]

    return lines


def loadHouse():
    house = []
    #Floor
    house.append(Line3D(Point3D(-5, 0, -5), Point3D(5, 0, -5)))
    house.append(Line3D(Point3D(5, 0, -5), Point3D(5, 0, 5)))
    house.append(Line3D(Point3D(5, 0, 5), Point3D(-5, 0, 5)))
    house.append(Line3D(Point3D(-5, 0, 5), Point3D(-5, 0, -5)))
    #Ceiling
    house.append(Line3D(Point3D(-5, 5, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(5, 5, -5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(5, 5, 5), Point3D(-5, 5, 5)))
    house.append(Line3D(Point3D(-5, 5, 5), Point3D(-5, 5, -5)))
    #Walls
    house.append(Line3D(Point3D(-5, 0, -5), Point3D(-5, 5, -5)))
    house.append(Line3D(Point3D(5, 0, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(5, 0, 5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(-5, 0, 5), Point3D(-5, 5, 5)))
    #Door
    house.append(Line3D(Point3D(-1, 0, 5), Point3D(-1, 3, 5)))
    house.append(Line3D(Point3D(-1, 3, 5), Point3D(1, 3, 5)))
    house.append(Line3D(Point3D(1, 3, 5), Point3D(1, 0, 5)))
    #Roof
    house.append(Line3D(Point3D(-5, 5, -5), Point3D(0, 8, -5)))
    house.append(Line3D(Point3D(0, 8, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(-5, 5, 5), Point3D(0, 8, 5)))
    house.append(Line3D(Point3D(0, 8, 5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(0, 8, 5), Point3D(0, 8, -5)))

    return house


def loadCar():
    car = []
    #Front Side
    car.append(Line3D(Point3D(-3, 2, 2), Point3D(-2, 3, 2)))
    car.append(Line3D(Point3D(-2, 3, 2), Point3D(2, 3, 2)))
    car.append(Line3D(Point3D(2, 3, 2), Point3D(3, 2, 2)))
    car.append(Line3D(Point3D(3, 2, 2), Point3D(3, 1, 2)))
    car.append(Line3D(Point3D(3, 1, 2), Point3D(-3, 1, 2)))
    car.append(Line3D(Point3D(-3, 1, 2), Point3D(-3, 2, 2)))

    #Back Side
    car.append(Line3D(Point3D(-3, 2, -2), Point3D(-2, 3, -2)))
    car.append(Line3D(Point3D(-2, 3, -2), Point3D(2, 3, -2)))
    car.append(Line3D(Point3D(2, 3, -2), Point3D(3, 2, -2)))
    car.append(Line3D(Point3D(3, 2, -2), Point3D(3, 1, -2)))
    car.append(Line3D(Point3D(3, 1, -2), Point3D(-3, 1, -2)))
    car.append(Line3D(Point3D(-3, 1, -2), Point3D(-3, 2, -2)))

    #Connectors
    car.append(Line3D(Point3D(-3, 2, 2), Point3D(-3, 2, -2)))
    car.append(Line3D(Point3D(-2, 3, 2), Point3D(-2, 3, -2)))
    car.append(Line3D(Point3D(2, 3, 2), Point3D(2, 3, -2)))
    car.append(Line3D(Point3D(3, 2, 2), Point3D(3, 2, -2)))
    car.append(Line3D(Point3D(3, 1, 2), Point3D(3, 1, -2)))
    car.append(Line3D(Point3D(-3, 1, 2), Point3D(-3, 1, -2)))

    return car


def loadTire():
    tire = []
    #Front Side
    tire.append(Line3D(Point3D(-1, .5, .5), Point3D(-.5, 1, .5)))
    tire.append(Line3D(Point3D(-.5, 1, .5), Point3D(.5, 1, .5)))
    tire.append(Line3D(Point3D(.5, 1, .5), Point3D(1, .5, .5)))
    tire.append(Line3D(Point3D(1, .5, .5), Point3D(1, -.5, .5)))
    tire.append(Line3D(Point3D(1, -.5, .5), Point3D(.5, -1, .5)))
    tire.append(Line3D(Point3D(.5, -1, .5), Point3D(-.5, -1, .5)))
    tire.append(Line3D(Point3D(-.5, -1, .5), Point3D(-1, -.5, .5)))
    tire.append(Line3D(Point3D(-1, -.5, .5), Point3D(-1, .5, .5)))

    #Back Side
    tire.append(Line3D(Point3D(-1, .5, -.5), Point3D(-.5, 1, -.5)))
    tire.append(Line3D(Point3D(-.5, 1, -.5), Point3D(.5, 1, -.5)))
    tire.append(Line3D(Point3D(.5, 1, -.5), Point3D(1, .5, -.5)))
    tire.append(Line3D(Point3D(1, .5, -.5), Point3D(1, -.5, -.5)))
    tire.append(Line3D(Point3D(1, -.5, -.5), Point3D(.5, -1, -.5)))
    tire.append(Line3D(Point3D(.5, -1, -.5), Point3D(-.5, -1, -.5)))
    tire.append(Line3D(Point3D(-.5, -1, -.5), Point3D(-1, -.5, -.5)))
    tire.append(Line3D(Point3D(-1, -.5, -.5), Point3D(-1, .5, -.5)))

    #Connectors
    tire.append(Line3D(Point3D(-1, .5, .5), Point3D(-1, .5, -.5)))
    tire.append(Line3D(Point3D(-.5, 1, .5), Point3D(-.5, 1, -.5)))
    tire.append(Line3D(Point3D(.5, 1, .5), Point3D(.5, 1, -.5)))
    tire.append(Line3D(Point3D(1, .5, .5), Point3D(1, .5, -.5)))
    tire.append(Line3D(Point3D(1, -.5, .5), Point3D(1, -.5, -.5)))
    tire.append(Line3D(Point3D(.5, -1, .5), Point3D(.5, -1, -.5)))
    tire.append(Line3D(Point3D(-.5, -1, .5), Point3D(-.5, -1, -.5)))
    tire.append(Line3D(Point3D(-1, -.5, .5), Point3D(-1, -.5, -.5)))

    return tire

def obToWorldRotate(ang):
    ang_rad = ang * np.pi / 180
    return np.array([[np.cos(ang_rad), 0, np.sin(ang_rad), 0],
              [0, 1, 0, 0],
              [-np.sin(ang_rad), 0, np.cos(ang_rad), 0],
              [0, 0, 0, 1]])

def obToWorldTranslate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

def drawObjAtLocation(matrix, lines):
    new_lines = []
    for line in lines:
        homog_start = np.array([line.start.x, line.start.y, line.start.z, 1])
        homog_end = np.array([line.end.x, line.end.y, line.end.z, 1])
        world_s = matrix @ homog_start
        world_e = matrix @ homog_end

        new_lines.append(Line3D(Point3D(world_s[0], world_s[1], world_s[2]), Point3D(world_e[0], world_e[1], world_e[2])))
    return new_lines

def setupNeighborhood():
    house_lines = []
    # the back house
    stack = []
    id_matrix = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    current_matrix = id_matrix
    # push matrix
    stack.append(current_matrix) # A
    current_matrix = obToWorldTranslate(0, 0, 60) @ obToWorldRotate(180) @ np.copy(stack[-1])
    house_lines.extend(drawObjAtLocation(current_matrix, loadHouse()))
    # pop matrix
    stack.pop() # A

    # work on the two rows
    # both rows should start 15 back
    current_matrix = obToWorldTranslate(0, 0, -15) @ id_matrix
    stack.append(current_matrix) # A

    # draw houses on the left row
    stack.append(current_matrix)  # B
    # first house
    current_matrix = obToWorldRotate(90) @ obToWorldTranslate(-15, 0, 0) @ np.copy(stack[-1])
    house_lines.extend(drawObjAtLocation(current_matrix, loadHouse()))

    stack.append(current_matrix)  # C
    # second house
    current_matrix = obToWorldTranslate(0, 0, 15) @ np.copy(stack[-1])
    house_lines.extend(drawObjAtLocation(current_matrix, loadHouse()))

    stack.append(current_matrix)  # D
    # third house
    current_matrix = obToWorldTranslate(0, 0, 15) @ np.copy(stack[-1])
    house_lines.extend(drawObjAtLocation(current_matrix, loadHouse()))

    stack.pop()  # D
    stack.pop()  # C
    stack.pop()  # B

    # draw houses on the right row
    stack.append(stack[-1])  # B
    # first house
    current_matrix = obToWorldRotate(-90) @ obToWorldTranslate(15, 0, 0) @ np.copy(stack[-1])
    house_lines.extend(drawObjAtLocation(current_matrix, loadHouse()))

    stack.append(current_matrix)  # C
    # second house
    current_matrix = obToWorldTranslate(0, 0, 15) @ np.copy(stack[-1])
    house_lines.extend(drawObjAtLocation(current_matrix, loadHouse()))

    stack.append(current_matrix)  # D
    # third house
    current_matrix = obToWorldTranslate(0, 0, 15) @ np.copy(stack[-1])
    house_lines.extend(drawObjAtLocation(current_matrix, loadHouse()))
    stack.pop()  # D

    stack.pop()  # C
    stack.pop()  # B
    stack.pop()  # A
    return house_lines

def setupCar():
    # draw the car
    car_lines = []
    # the back house
    stack = []
    id_matrix = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    current_matrix = id_matrix
    # push matrix
    stack.append(current_matrix)  # A
    current_matrix =  obToWorldTranslate(0, 0, 20) @ obToWorldRotate(90) @ np.copy(stack[-1])
    car_lines.extend(drawObjAtLocation(current_matrix, loadCar()))

    # tires
    car_lines.extend(setupTires(current_matrix))

    stack.pop()
    return car_lines


def setupTires(current_matrix):
    stack = []
    tire_lines = []
    stack.append(current_matrix)

    # push
    stack.append(np.copy(stack[-1]))
    # tire 1
    current_matrix = obToWorldTranslate(-2, 0, 2) @ np.copy(stack[-1])
    tire_lines.extend(drawObjAtLocation(current_matrix, loadTire()))
    stack.pop()

    # push
    stack.append(np.copy(stack[-1]))
    # tire 2
    current_matrix = obToWorldTranslate(2, 0, 2) @ np.copy(stack[-1])
    tire_lines.extend(drawObjAtLocation(current_matrix, loadTire()))
    stack.pop()

    # push
    stack.append(np.copy(stack[-1]))
    # tire 3
    current_matrix = obToWorldTranslate(-2, 0, -2) @ np.copy(stack[-1])
    tire_lines.extend(drawObjAtLocation(current_matrix, loadTire()))
    stack.pop()

    # push
    stack.append(np.copy(stack[-1]))
    # tire 4
    current_matrix = obToWorldTranslate(2, 0, -2) @ np.copy(stack[-1])
    tire_lines.extend(drawObjAtLocation(current_matrix, loadTire()))
    stack.pop()
    stack.pop()

    return tire_lines

def translate(x, y, z):
    global currentX, currentY, currentZ, currentAngle
    angle_rad = currentAngle * (np.pi / 180)
    result_x = x * np.cos(angle_rad) + z * np.sin(angle_rad)
    result_y = y
    result_z = x * - np.sin(angle_rad) + z * np.cos(angle_rad)

    currentX += result_x
    currentY += result_y
    currentZ += result_z

def homeCameraPosition():
    global currentX, currentY, currentZ, currentAngle
    currentX = 0
    currentY = 0
    currentZ = -15
    currentAngle = 0


homeCameraPosition()

# Initialize the game engine
pygame.init()

# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

STEP = 1/5

# Set the height and width of the screen
size = [512, 512]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Shape Drawing")

#Set needed variables
done = False
clock = pygame.time.Clock()
start = Point(0.0,0.0)
end = Point(0.0,0.0)
# linelist = loadHouse()
line_list = setupNeighborhood()
end_houses = len(line_list) - 1
line_list.extend(setupCar())

#Loop until the user clicks the close button.
while not done:

    # This limits the while loop to a max of 100 times per second.
    # Leave this out and we will use all CPU we can.
    clock.tick(100)

    # Clear the screen and set the screen background
    screen.fill(BLACK)

    #Controller Code#
    #####################################################################

    for event in pygame.event.get():
        if event.type == pygame.QUIT: # If user clicked close
            done=True

    pressed = pygame.key.get_pressed()

    if pressed[pygame.K_a]:
        # move left
        translate(-STEP, 0, 0)
        print("a: move left")
    elif pressed[pygame.K_d]:
        # move right
        translate(STEP, 0, 0)
        print("d: move right")
    elif pressed[pygame.K_w]:
        # move forward
        translate(0, 0, STEP)
        print("w: move forward")
    elif pressed[pygame.K_s]:
        # move backward
        translate(0, 0, -STEP)
        print("s: move backward")
    elif pressed[pygame.K_q]:
        # turn left
        currentAngle -= STEP
        print("q: turn left")
    elif pressed[pygame.K_e]:
        # turn right
        currentAngle += STEP
        print("e: turn right")
    elif pressed[pygame.K_r]:
        # move up
        translate(0, STEP, 0)
        print("r: move up")
    elif pressed[pygame.K_f]:
        # move down
        translate(0, -STEP, 0)
        print("f: move down")
    elif pressed[pygame.K_h]:
        # return home
        homeCameraPosition()
        print("h: return home")

    #Viewer Code#
    #####################################################################

    # build world to camera matrix - consists of translate and rotate
    wtc_translate = np.array([[1, 0, 0, -currentX],
                               [0, 1, 0, -currentY],
                               [0, 0, 1, -currentZ],
                               [0, 0, 0, 1]])
    # rotate around y axis
    angle_rad = currentAngle * (np.pi / 180)
    wtc_rotate = np.array([[np.cos(angle_rad), 0, - np.sin(angle_rad), 0],
                            [0,                 1,  0,                  0],
                            [np.sin(angle_rad), 0, np.cos(angle_rad),   0],
                            [0,                 0, 0,                   1]])
    world_to_camera = np.matmul(wtc_rotate, wtc_translate)

    # clip matrix
    f = 100
    n = 1
    fov = np.pi / 2
    zoom_x = 1 / np.tan(fov / 2)
    zoom_y = 1 / np.tan(fov / 2)

    clip = np.array([[zoom_x, 0, 0, 0],
                      [0, zoom_y, 0, 0],
                      [0, 0, (f+n)/(f-n), -2*n*f/(f-n)],
                      [0, 0, 1, 0]])

    to_screen_space = np.array([[size[0]/2, 0, size[0]/2],
                                 [0, -size[1]/2, size[1]/2],
                                 [0, 0, 1]])

    for i, s in enumerate(line_list):
        homog_start = np.array([s.start.x, s.start.y, s.start.z, 1])
        homog_end = np.array([s.end.x, s.end.y, s.end.z, 1])

        clipped_s = clip @ world_to_camera @ homog_start # return a 1x4 matrix so take the first row
        clipped_e = clip @ world_to_camera @ homog_end
        w_s = clipped_s[3]
        w_e = clipped_e[3]
        # Reject a line if either endpoint fails the near plane test
        # near plane test is z < -w
        if clipped_s[2] < -w_s or clipped_e[2] < -w_e:
            continue
        # Reject a line if both points fail the same view frustum test
        # test start point failure
        if clipped_s[0] > w_s or clipped_s[0] < -w_s or clipped_s[1] > w_s or clipped_s[1] < -w_s or clipped_s[2] > w_s or clipped_s[2] < -w_s:
            if clipped_e[0] > w_e or clipped_e[0] < -w_e or clipped_e[1] > w_e or clipped_e[1] < -w_e or clipped_e[2] > w_e or clipped_e[2] < -w_e:
                continue
        # if you get to this point, draw the line
        # first get it to perspective
        persp_s = (clipped_s / w_s)[:2]  # [:2] to only grab x and y coordinates
        persp_e = (clipped_e / w_e)[:2]

        persp_s = np.append(persp_s, [1])   # make these homogenous
        persp_e = np.append(persp_e, [1])   # make these homogenous

        # to screen space coordinates
        screen_s = to_screen_space @ persp_s
        screen_e = to_screen_space @ persp_e

        if i <= end_houses:  # draw the house lines blue
            pygame.draw.line(screen, BLUE, (screen_s[0], screen_s[1]), (screen_e[0], screen_e[1]))
        else:  # draw the car lines red
            pygame.draw.line(screen, RED, (screen_s[0], screen_s[1]), (screen_e[0], screen_e[1]))

    # Go ahead and update the screen with what we've drawn.
    # This MUST happen after all the other drawing commands.
    pygame.display.flip()

# Be IDLE friendly
pygame.quit()
