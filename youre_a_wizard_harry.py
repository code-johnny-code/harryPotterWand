import cv2
import os

class Incendio:
    
    def __init__(self):

        def cast():
            print('INCENDIO')
            os.system("chromium-browser www.youtube.com/watch?v=QeYoiBmuzOM")

        self.steps = [
            {
                "name": 1,
                "start_point": (5,5),
                "end_point": (100,100),
                "color": (0, 0, 255)
            },
            {
                "name": 2,
                "start_point": (200, 200),
                "end_point": (295,295),
                "color": (0, 0, 255)
            },
            {
                "name": 3,
                "start_point": (5, 200),
                "end_point": (100,295),
                "color": (0, 0, 255)
            }
        ]
        self.sequence = [1, 2, 3]
        self.progress = []
        self.cast = cast

spell = Incendio()

def markSpellProgress(spell, cX, cY):
    
    def spellStepHit(step):
        if step["end_point"][0] > cX > step["start_point"][0] and step["end_point"][1] > cY > step["start_point"][1]:
            return True
        else:
            return False
    
    for step in spell.steps:
        if spellStepHit(step):
            correctIndex = spell.sequence.index(step["name"])
            potentialIndex = len(spell.progress)
            spellHasProgress = len(spell.progress)
            stepAlreadyHit = step["name"] in spell.progress
            stepIsNext = correctIndex == potentialIndex
            
            if not stepAlreadyHit and stepIsNext:
                spell.progress.append(step["name"])
                step["color"] = (0, 255, 0)
                if spell.progress == spell.sequence:
                    spell.cast()
                    resetSpell(spell)
                    break
            elif not stepAlreadyHit and spellHasProgress and not stepIsNext:
                resetSpell(spell)
            elif stepAlreadyHit and step["name"] != spell.progress[-1]:
                resetSpell(spell)
            

def resetSpell(spell):
    spell.progress = []
    for step in spell.steps:
        step["color"] = (0, 0, 255)

# Initialize Camera Feed
cap = cv2.VideoCapture(0)
# incendioImage = cv2.imread('./incendio.jpeg')
# rows,cols,channels = incendioImage.shape

while(True):
# Get Frame from Camera Feed
    ret, frame = cap.read()
    flippedFrame = cv2.flip(frame, 0)
    for step in spell.steps:
        userView = cv2.rectangle(flippedFrame, step["start_point"], step["end_point"], step["color"], 2)
        # test = cv2.putText(test, str(step["name"]), (step["start_point"][0], step["end_point"][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)

# Convert Camera Feed to Grayscale
    grayFrame = cv2.cvtColor(flippedFrame, cv2.COLOR_BGR2GRAY)

# Set threshold value and method
    ret, thresh = cv2.threshold(grayFrame, 240, 255, cv2.THRESH_TOZERO)
    # thresh = cv2.rectangle(thresh, start_point, end_point, color, thickness)

# Find contours
    edged = cv2.Canny(thresh, 30, 200)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Register Centroids of Blobs
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            flippedFrame = cv2.circle(flippedFrame, (cX, cY), 20, (0, 0, 255), 3)
            markSpellProgress(spell, cX, cY)
            # flippedFrame = cv2.putText(flippedFrame, (str(cX) + ',' + str(cY)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            # if cX < end_point[0] and cY < end_point[0]:
                # print('BOOP')


# Track Centroids


# Add Targets


# Track Centroids intersecting Targets


# Show Frame
    cv2.imshow('Camera', userView)
    # cv2.imshow('Threshold', thresh)
    # cv2.imshow('Canny', edged)
    # cv2.imshow('Contours', im2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
