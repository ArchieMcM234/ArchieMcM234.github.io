import pygame, sys
from pygame.locals import *
import math
from opensimplex import OpenSimplex
import numpy as np
import random
import copy
import pickle 

PIXELSPERMETER = 5
RANDRATE =1 # set to random number %
ADDRATE = 1 # add random number
TIMESRATE = 1 # times by random number

# progress around course
# make points more consistent 
# pixelspermeter more consistent e.g radar
# efficiency
# how using oop
# draw function
# put in seperate files
# sort out constants used in functions
# sort out wheather self. or not
# check gameScreen needed
# parrallel lines
#define constants

# put wall collision in track
# finish progress


class neuralNetwork:
    def __init__(self, layerSizes):
        self.layerSizes = layerSizes
        self.values = [[] for a in range(len(self.layerSizes))]
        self.weights = [[] for a in range(len(self.layerSizes)-1)]
        self.biases = [[] for a in range(len(self.layerSizes)-1)]    
        self.fitness = 0

    def generateValues(self): # fills in previously made matrixes with random numbers normal distributed
        self.values = [[] for a in range(len(self.layerSizes))]
        self.weights = [[] for a in range(len(self.layerSizes)-1)]
        self.biases = [[] for a in range(len(self.layerSizes)-1)]        
        for layerNo in range(0, len(self.layerSizes)-1):
            for neuronNo in range(self.layerSizes[layerNo+1]):
                self.weights[layerNo].append(np.random.normal(0, 1, self.layerSizes[layerNo]))
            self.biases[layerNo] = np.random.normal(0, 1, self.layerSizes[layerNo+1])

    def recieveValues(self, weights, biases):  # need to deep copy as other wise uses pointer to original
        self.weights = copy.deepcopy(weights)
        self.biases = copy.deepcopy(biases)

    def think(self, inputArray): # goes through the layers multiplying each and then adding the biases put through the activationFunction then repeat
        if len(inputArray) != self.layerSizes[0]:
            return None
        self.values[0] = inputArray
        for layerNo in range(len(self.layerSizes)-1):
            self.values[layerNo+1]=self.activationFunction( np.add( np.dot(self.weights[layerNo], self.values[layerNo] ), self.biases[layerNo]) )
        returnArray = []
        for value in self.values[len(self.layerSizes)-1]:
            returnArray.append(round(value))
        return returnArray

    def activationFunction(self, value):
        for elementNo in range(len(value)):
            if value[elementNo] > 500:
                value[elementNo]= 500
            elif value[elementNo] < -500:
                value[elementNo] = -500
        return 1 / (1 + np.exp(-value))


    def findFitness(self, progressAround, timeTaken): 
        if progressAround == 1:
            self.fitness = 10-(timeTaken/40) # if completed the track take into account lap time
        else:
            self.fitness = progressAround # else order by their progress around
        
    def crossOver(self, partner):
        newWeights = copy.deepcopy(self.weights) # makes copy of weights biases of one of the networks
        newBiases = copy.deepcopy(self.biases)

        for layerNo in range(len(self.weights)): # iterates through layers and weights and biases and randomly picks
            for valueNo in range(len(self.weights[layerNo])):
                for weightNo in range(len(self.weights[layerNo][valueNo])):
                    decider = random.randint(1, 2)
                    if decider == 2:
                        newWeights[layerNo][valueNo][weightNo] = partner.weights[layerNo][valueNo][weightNo]

            for biasNo in range(len(self.biases[layerNo])):
                decider = random.randint(1,2)
                if decider == 2:
                    newBiases[layerNo][biasNo] = partner.biases[layerNo][biasNo]

        newNetwork = neuralNetwork(self.layerSizes) # creates new network and passes through the new values
        newNetwork.recieveValues(newWeights, newBiases)
        return newNetwork      

    def copy(self): # creates and returns copy of its self
        newNetwork = neuralNetwork(self.layerSizes)
        newNetwork.recieveValues(self.weights, self.biases)
        return newNetwork

    def mutate(self): 
        for layerNo in range(len(self.weights)):# iterates through each layer
            for valueNo in range(len(self.weights[layerNo])): # iterates through each value in a layer
                for weightNo in range(len(self.weights[layerNo][valueNo])): # iterates through each weights
                    decider = random.randint(0, 100)
                    if decider <= RANDRATE:
                        self.weights[layerNo][valueNo][weightNo] = np.random.normal(0, 1, 1)[0] # changes weights biases based on rates
                    elif decider <= ADDRATE+RANDRATE:
                        self.weights[layerNo][valueNo][weightNo] += np.random.normal(0, 1, 1)[0]
                    elif decider <= ADDRATE+RANDRATE+TIMESRATE:
                        self.weights[layerNo][valueNo][weightNo] *= np.random.normal(0, 1, 1)[0]
                        
            for biasNo in range(len(self.biases[layerNo])): # iterates through biases
                decider = random.randint(0, 100)
                if decider <= RANDRATE:
                    self.biases[layerNo][biasNo] = np.random.normal(0, 1, 1)[0]
                elif decider <= ADDRATE+RANDRATE:
                    self.biases[layerNo][biasNo] += np.random.normal(0, 1, 1)[0]
                elif decider <= ADDRATE+RANDRATE+TIMESRATE:
                    self.biases[layerNo][biasNo] *= np.random.normal(0, 1, 1)[0]     
                    
    def saveNeuralNetwork(self, filename):
        filehandler = open(filename, 'wb')
        pickle.dump(self.layerSizes, filehandler)
        pickle.dump(self.weights, filehandler)
        pickle.dump(self.values, filehandler)
        pickle.dump(self.biases, filehandler)
        pickle.dump(self.fitness, filehandler)
    def loadNeuralNetwork(self, filename):
        filehandler = open(filename, 'rb') 
        self.layerSizes = pickle.load(filehandler) 
        self.weights = pickle.load(filehandler) 
        self.values = pickle.load(filehandler) 
        self.biases = pickle.load(filehandler) 
        self.fitness = pickle.load(filehandler) 
        
class car:
    def __init__(self, startX, startY, startAngle): # get it so input constants
        self.carLength = 5
        self.carWidth = 3
        self.diagLength = 14.5
        
        self.Xcoords = startX
        self.Ycoords = startY
        self.angle = startAngle
        
        self.speed = 0
        self.acceleration = 0
        self.steering = 0
        self.angularVelocity = 0
        
        self.dragConstant = 6
        self.mass = 1500
        self.carForce =  10000
        self.breakingConstant = 2000
        self.rollingResConstant = 150

        self.cornerLines = [0 for a in range(4)]
        self.latestCheckPoint = 0
        self.lapTime = 0
        self.image = pygame.transform.scale(pygame.image.load('carimage.png'), (self.carWidth*PIXELSPERMETER, self.carLength*PIXELSPERMETER))
        
    def recieveControls(self, controls, deltaTime): # recieves controls and finds acceleration
        resistiveForce = self.dragConstant*(self.speed**2) # air resistance
        resistiveForce += self.rollingResConstant*abs(self.speed) # rolling resistance
        
        if controls[0]: # space bar
            resistiveForce += self.breakingConstant*abs(self.speed)   
            
        if self.speed > 0: # orients resistive force
            resistiveForce *=-1
            
        if controls[1] and not controls[2]: # W key     
            drivingForce = self.carForce 
        elif controls[2] and not controls[1]: # S key
            drivingForce = -self.carForce 
        else:    
            drivingForce = 0

        resultant = drivingForce + resistiveForce # calculates acceleration
        self.acceleration = resultant/self.mass
        
        if controls[3] and not controls[4]: # A key
            self.steering = 30
        elif controls[4] and not controls[3]:# D key
            self.steering = -30
        else:
            self.steering = 0    

    def updatePosition(self, deltaTime): # integrates to update position 

        self.speed += self.acceleration*deltaTime # integrates up by multiplying deltaTime
        distance = self.speed*deltaTime

        if self.steering != 0: # need to check that angle isn't zero due to sin function
            turningRadius = self.carLength/math.sin(math.radians(self.steering)) # find turning radius
            self.angularVelocity = self.speed/turningRadius        
        else:
            self.angularVelocity = 0

        self.angle += math.degrees(self.angularVelocity*deltaTime) # integrates rotation

        self.Xcoords += PIXELSPERMETER*distance*math.sin(math.radians(self.angle)) # updates coords
        self.Ycoords += PIXELSPERMETER*distance*math.cos(math.radians(self.angle))
    
    def getCornerLines(self): # gets line objects of outerlines of cars
        corners = [(0, 0) for a in range(4)]
        angles = [31, -31, 211, 149] 
        for cornerNo in range(4):
            angle = self.angle+angles[cornerNo]
            cornerX = self.Xcoords+ math.sin(math.radians(angle))*self.diagLength
            cornerY = self.Ycoords + math.cos(math.radians(angle))*self.diagLength
            corners[cornerNo] = (cornerX, cornerY) 
        for lineNo in range(4):
            self.cornerLines[lineNo] = line(corners[lineNo%4], corners[(lineNo+1)%4])
        return self.cornerLines
    
    def getProgressAroundTrack(self, track): 
        for lineNo in range(len(track.checkPointLines)):# iterates through checkpoint lines
            for carLine in self.cornerLines:# iterates through cars bounding lines
                if carLine.detectLineCollision(track.checkPointLines[lineNo]):
                    adjustedLineNo = (lineNo-90)%119 # due to arrangment of track lines must be adjusted
                    if  adjustedLineNo == self.latestCheckPoint+1 or adjustedLineNo == self.latestCheckPoint+2:
                        self.latestCheckPoint = adjustedLineNo # if has incremented then update
        return self.latestCheckPoint/118
    
    def getLapTime(self, deltaTime):
        if self.latestCheckPoint != 118: # checks if has reached end of track
            self.lapTime += deltaTime # adds change in time untill finishes
        return self.lapTime
    
    def getRadarDistances(self, track): # gets distance to wall from front of car, 
        radarLength = 300 # how far lines go out
        distances = [500, 500, 500] # initialises with default values
        for count in range(3):
            lineAngle = -30+ 30*count # iterates through three lines incrementing angle
            
            endLineX = self.Xcoords+ math.sin(math.radians(self.angle+lineAngle))*radarLength
            endLineY = self.Ycoords + math.cos(math.radians(self.angle+lineAngle))*radarLength
            radar = line((self.Xcoords, self.Ycoords), (endLineX, endLineY))# creates line set distance long at lineAngle to car
            
            trackLines = track.outerLines+track.innerLines
            for trackLine in trackLines: # goes through all lines bounding track
                point = radar.detectLineCollision(trackLine) # find point of intersection
                if point:
                    distance = math.sqrt((point[0]-self.Xcoords)**2+(point[1]-self.Ycoords)**2) # uses pythag to find distance
                    if distance<distances[count]:
                        distances[count] = distance# finds smallest distance
        return distances

class track:
    def __init__(self):
        self.outerPointList = []
        self.innerPointList = []
        self.centreX = random.randint(0, 1000)
        self.centreY = random.randint(0, 1000)
        self.innerLines = []
        self.outerLines = []
        self.checkPointLines = []
        
        
    def generatePoints(self, width, height): # uses perlin noise to create smooth random distances that then map every 3 degrees around centre point 
        noiseSpace = OpenSimplex()
        for angle in range (0, 360, 3):
            noiseX = self.centreX + math.cos(math.radians(angle))
            noiseY = self.centreY + math.sin(math.radians(angle))
            distanceFromCentre = (abs(noiseSpace.noise2d(x=noiseX, y=noiseY)+1.8))*100 # adding constant like smoothing value
            pointX = distanceFromCentre*math.cos(math.radians(angle)) + width/2
            pointY = distanceFromCentre*math.sin(math.radians(angle)) + height/2
            self.innerPointList.append((pointX, pointY))
            
            distanceFromCentre += 50 # gives the track its width
            pointX = distanceFromCentre*math.cos(math.radians(angle)) + width/2
            pointY = distanceFromCentre*math.sin(math.radians(angle)) + height/2
            self.outerPointList.append((pointX, pointY))       
            
        for lineNo in range(len(self.innerPointList)): # creates arrays of line objects
            self.innerLines.append(line(self.innerPointList[lineNo%len(self.innerPointList)], self.innerPointList[(lineNo+1)%len(self.innerPointList)]))
            self.outerLines.append(line(self.outerPointList[lineNo%len(self.outerPointList)], self.outerPointList[(lineNo+1)%len(self.outerPointList)]))
            self.checkPointLines.append(line(self.innerPointList[lineNo], self.outerPointList[lineNo]))
            
        startX = (self.outerPointList[89][0]+self.innerPointList[89][0])/2 # if change no lines change this
        startY = (self.outerPointList[89][1]+self.innerPointList[89][1])/2
        startAngle = 90-math.degrees(math.atan2((self.innerPointList[90][1]-self.innerPointList[88][1]), (self.innerPointList[90][0]-self.innerPointList[88][0])))
        return startX, startY, startAngle
    
    def detectCarCollision(self, car, gameScreen): # maybe some improvement
        for trackLine in self.innerLines:
            for carLine in car.cornerLines:
                if carLine.detectLineCollision(trackLine):
                    return True
        for trackLine in self.outerLines:
            for carLine in car.cornerLines:
                if carLine.detectLineCollision(trackLine):
                    return True
        return False
                                
    def drawTrack(self, gameScreen):
        pygame.draw.polygon(gameScreen, (0, 0, 0), self.outerPointList, 0)
        pygame.draw.polygon(gameScreen, (0, 120, 0), self.innerPointList, 0)
        pygame.draw.lines(gameScreen, (255, 255, 255), True, self.outerPointList, 1)
        pygame.draw.lines(gameScreen, (255, 255, 255), True, self.innerPointList, 1)
        
class line():
    def __init__(self, P1, P2):
        self.P1 = P1
        self.P2 = P2
        self.gradient, self.yIntercept = self.getEquation()
        
    def drawLine(self, gameScreen):
        pygame.draw.line(gameScreen, (255, 255, 255), self.P1, self.P2, 1)
        
    def getEquation(self): # effiency boost
        firstPoint, secondPoint = self.getFirstSecond()
        
        if secondPoint[0]-firstPoint[0] == 0:
            gradient = None
            yIntercept = firstPoint[0]
        else:
            gradient = (secondPoint[1]-firstPoint[1])/(secondPoint[0]-firstPoint[0])
            yIntercept = -gradient*firstPoint[0]+firstPoint[1]
        return gradient, yIntercept
    
    def getFirstSecond(self):
        firstPoint = self.P1 if self.P1[0] <= self.P2[0] else self.P2
        secondPoint = self.P1 if self.P1[0] > self.P2[0] else self.P2            
        return firstPoint, secondPoint    
    
    def detectLineCollision(self, line2):
        myFirst, mySecond = self.getFirstSecond()
        theirFirst, theirSecond = line2.getFirstSecond()        
        if self.gradient == line2.gradient: # parrallel
            if self.yIntercept == line2.yIntercept:
                if myFirst[0] >= theirFirst[0] and myFirst[0] <= theirSecond[0] or mySecond[0] >= theirFirst[0] and mySecond[0] <= theirSecond[0]:# this still need to be fixed
                    return myFirst[0], myFirst[1]
                else:
                    return False
            else:
                return False 
        elif not self.gradient: # vertical
            crossPointX = self.P1[0]
            crossPointY = line2.gradient*crossPointX + line2.yIntercept
            if crossPointY <= mySecond[1] and crossPointY >= myFirst[1] and crossPointY <= theirSecond[1] and crossPointY >= theirFirst[1]:
                return crossPointX, crossPointY
            else:
                return False
        elif not line2.gradient: # vertical
            crossPointX = line2.P1[0]
            crossPointY = self.gradient*crossPointX + self.yIntercept
            if crossPointY <= mySecond[1] and crossPointY >= myFirst[1] and crossPointY <= theirSecond[1] and crossPointY >= theirFirst[1]:
                return crossPointX, crossPointY
            else:
                return False            
        else:
            crossPointX = (line2.yIntercept-self.yIntercept)/(self.gradient-line2.gradient)
            crossPointY = self.gradient*crossPointX +self.yIntercept
        if crossPointX <= mySecond[0] and crossPointX >= myFirst[0] and crossPointX <= theirSecond[0] and crossPointX >= theirFirst[0]:
            return crossPointX, crossPointY
        else:
            return False      
            
class button(object):
    def __init__(self, position, text, fontSize):
        self.colour = (0,0,0) 
        self.position = position
        self.rect = pygame.Rect(position)
        self.font = pygame.font.Font(None, fontSize) 
        self.text = self.font.render(text, True, (0,0,0))        
    def changeText(self, text):
        self.text = self.font.render(text, True, (0,0,0))
    def draw(self, screen):
        pygame.draw.rect(screen, self.colour, self.rect, 0)
        screen.blit(self.text, self.text.get_rect(center=self.rect.center)) 
    def clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                return self.rect.collidepoint(event.pos)    
    def hover(self, mousPos):
        return self.rect.collidepoint(mousPos)

class window:
    def __init__(self): # set up the pygame window
        pygame.init()
        self.width = 1000
        self.height = 600
        self.gameScreen = pygame.display.set_mode((self.width, self.height)) 
        self.clock = pygame.time.Clock()
        self.raceTrack = track()
        
        self.menuImage = pygame.transform.scale(pygame.image.load('menuImage.png'), (350, 265))
        
        self.controlSquares = []
        self.controlSquares.append(button((10,560, 150, 30), "__", 25))
        self.controlSquares.append(button((60,450, 50, 50), "W", 25))
        self.controlSquares.append(button((60,500, 50, 50), "S", 25))
        self.controlSquares.append(button((10,500, 50, 50), "A", 25))
        self.controlSquares.append(button((110,500, 50, 50), "D", 25))
        
        self.population = [neuralNetwork([3, 7, 7, 5]) for a in range(10)]
            
    def mainMenu(self):
        newButton = button((250, 300, 500, 75), "New", 40)
        loadButton = button((250, 385, 500, 75), "Load", 40)
        closeButton = button((250, 470, 500, 75), "Close", 40)
        clock = pygame.time.Clock()
        running = True
        while running:
            self.clearWindow()
            for event in pygame.event.get():
                mousPos = pygame.mouse.get_pos()
                newButton.colour = (200, 200, 200) if newButton.hover(mousPos) else (225, 225, 225)
                loadButton.colour = (200, 200, 200) if loadButton.hover(mousPos) else (225, 225, 225)
                closeButton.colour = (200, 200, 200) if closeButton.hover(mousPos) else (225, 225, 225)
                
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                if newButton.clicked(event):
                    for brain in self.population:
                        brain.generateValues()    
                    self.gameLoop()
                if loadButton.clicked(event):
                    for brain in self.population:      
                        brain.loadNeuralNetwork('savedNeuralNetwork.nnet')
                        brain.mutate()
                    self.population[0].loadNeuralNetwork('savedNeuralNetwork.nnet')
                    self.gameLoop()
                if closeButton.clicked(event):
                    # exit
                    pygame.quit()
                    exit()
            newButton.draw(self.gameScreen)
            loadButton.draw(self.gameScreen)
            closeButton.draw(self.gameScreen)
            self.gameScreen.blit(self.menuImage, (325, 10))
            pygame.display.flip()
            clock.tick()        
            
    def gameLoop(self):
        menuButton = button((0, 0, 100, 40), "Menu", 30) #creates buttons
        saveButton = button((0, 42, 100, 40), "Save", 30)
        genNoSquare = button((700, 0, 300, 40), "Generation:  Network: /10", 25)
        genNoSquare.colour = (0, 120, 0)
        genNo = 0
        while True:
            genNo += 1
            self.raceTrack = track()
            startX, startY , startAngle= self.raceTrack.generatePoints(self.width, self.height) #generates track
            
            for brainNo in range(10):
                genNoSquare.changeText("Generation: "+str(genNo)+"  Network: "+str(brainNo+1)+"/10")
                
                brain = self.population[brainNo]
                self.raceCar = car(startX, startY, startAngle)  #places car at start of track
                
                playing = True
                while playing:
                    
                    deltaTime = self.clock.tick(30) # set the frame rate and find the time between frames
                    deltaTime /= 1000
                    
                    for event in pygame.event.get(): #iterates through events
                        mousPos = pygame.mouse.get_pos()
                        menuButton.colour = (200, 200, 200) if menuButton.hover(mousPos) else (255, 255, 255)
                        saveButton.colour = (200, 200, 200) if saveButton.hover(mousPos) else (255, 255, 255)                        
                        if event.type == QUIT:# check if user has quit
                            pygame.quit()
                            sys.exit()
                        if menuButton.clicked(event):
                            return False    # exits to menu loop   
                        if saveButton.clicked(event):
                            self.population[0].saveNeuralNetwork('savedNeuralNetwork.nnet') #saves best network
                        
                    radarDistances = self.raceCar.getRadarDistances(self.raceTrack) # find radar distances 
                    controls = brain.think(radarDistances) # pass network radar distances
                    self.raceCar.recieveControls(controls, deltaTime)# act on controls
                    self.raceCar.updatePosition(deltaTime) # updates car position
                    
                    self.raceCar.getCornerLines() # find cars bounding lines for current frame
                    progressAround = self.raceCar.getProgressAroundTrack(self.raceTrack)
                    timeTaken = self.raceCar.getLapTime(deltaTime)
 
                    self.clearWindow() 
                    self.raceTrack.drawTrack(self.gameScreen)
                    
                    rotatedImage = pygame.transform.rotate(self.raceCar.image, self.raceCar.angle) 
                    rectangle = rotatedImage.get_rect()
                    self.gameScreen.blit(rotatedImage, (self.raceCar.Xcoords-rectangle.width/2, self.raceCar.Ycoords-rectangle.height/2))# draws car    
                    
                    menuButton.draw(self.gameScreen)
                    saveButton.draw(self.gameScreen)
                    genNoSquare.draw(self.gameScreen)
                    self.drawControls(controls)     
                    
                    pygame.display.flip()
                    
                    # this if statement checks if the car has crashed has taken longer than 40 seconds or hasn't moved in the first 2 seconds
                    if self.raceTrack.detectCarCollision(self.raceCar, self.gameScreen) or progressAround ==1 or (timeTaken > 2 and progressAround<0.005) or timeTaken > 40:
                        playing = False
                                       
                brain.findFitness(progressAround, timeTaken)
            
            # evolution    

            self.population.sort(key=lambda x: x.fitness, reverse=True) # sorts networks based on perfomance
            matingPool = self.population[:4] # selects top 4
            self.population = self.population[:1]
            startNo = len(matingPool)-1
            for iterationNo in range(len(matingPool)-1): # crossover the first network with the top 4 multiple times based on fitness
                for brainNo in range(startNo, 0, -1):
                    newBrain = matingPool[0].crossOver(matingPool[brainNo]) # creates offspring
                    newBrain.mutate() # mutates network as seen in nature
                    self.population.append(newBrain) # adds new brain to population

    def drawControls(self, controls):
        for a in range(0,5):
            self.controlSquares[a].colour = (255, 255, 255) if controls[a] else (150, 150, 150)
            self.controlSquares[a].draw(self.gameScreen)
            
    def clearWindow(self):
        self.gameScreen.fill((0, 120, 0))

if __name__ == '__main__':
    gameWindow = window()
    gameWindow.mainMenu()
