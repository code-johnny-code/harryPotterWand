import cv2
import copy
import numpy as np

target_size_factor = 100
show_targets = False
user_progress = []

targets = [
    {
        "name": "a",
        "start_point": (50, 50),
        "color": (0, 0, 255)
    }, {
        "name": "b",
        "start_point": (150, 150),
        "color": (0, 0, 255)
    }, {
        "name": "c",
        "start_point": (150, 150),
        "color": (0, 0, 255)
    }
]

spells = [
    {
        "name": "Incendio",
        "sequence": ['a', 'b', 'c']
    },
    {
        "name": "Lumos",
        "sequence": ['c', 'b', 'a']
    },
]


def target_dimensions_box(step):
    top_left = step["start_point"]
    top_right = ((top_left[0] + target_size_factor), top_left[1])
    bottom_right = (top_right[0], (top_left[1] + target_size_factor))
    bottom_left = (top_left[0], bottom_right[1])
    return [top_left, top_right, bottom_right, bottom_left]


def target_dimensions_hexagon(step):
    point_1 = step["start_point"]
    point_2 = [(point_1[0] + target_size_factor), point_1[1]]
    point_3 = [(point_2[0] + target_size_factor), (point_2[1] + target_size_factor)]
    point_4 = [point_3[0], (point_3[1] + target_size_factor)]
    point_5 = [(point_4[0] - target_size_factor), (point_4[1] + target_size_factor)]
    point_6 = [(point_5 - target_size_factor), point_5[1]]
    point_7 = [(point_6[0] - target_size_factor), (point_6[1] - target_size_factor)]
    point_8 = [point_7[0], (point_7[1] - target_size_factor)]
    return [point_1, point_2, point_3, point_4, point_5, point_6, point_7, point_8]


def spell_step_hit(step, x, y):
    target_box = target_dimensions_box(step)
    top_edge = target_box[0][1]
    bottom_edge = target_box[2][0]
    left_edge = target_box[0][0]
    right_edge = target_box[1][0]
    in_box = top_edge >= x >= bottom_edge and left_edge <= y <= right_edge
    return in_box


def possible_spells():
    spells_possible = copy.copy(spells)
    for step_name in user_progress:
        progress_index = user_progress.index(step_name)
        for spell in spells:
            sequence_index = spell['sequence'].index(step_name)
            if progress_index != sequence_index:
                spells_possible.remove(spell)
    return spells_possible


def mark_spell_progress(x, y):
    valid_hit = False
    for spell in spells:
        for step in spell.steps:
            if spell_step_hit(step):
                correctIndex = spell.sequence.index(step["name"])
                potentialIndex = len(user_progress)
                stepAlreadyHit = step["name"] in spell.progress
                stepIsNext = correctIndex == potentialIndex

                if not stepAlreadyHit and stepIsNext:
                    spell.progress.append(step["name"])
                    step["color"] = (0, 255, 0)
                    if spell.progress == spell.sequence:
                        spell.cast()
                        reset_progress()
                        break
                elif not stepAlreadyHit and spellHasProgress and not stepIsNext:
                    reset_progress()
                elif stepAlreadyHit and step["name"] != spell.progress[-1]:
                    reset_progress()


def reset_progress():
    user_progress.clear()
    if show_targets:
        for spell in spells:
            for step in spell.steps:
                step["color"] = (0, 0, 255)


# Initialize Camera Feed
cap = cv2.VideoCapture(0)

while True:
    # Get Frame from Camera Feed
    ret, frame = cap.read()
    # My camera is upside down, so I need to flip the frame over
    flipped_frame = cv2.flip(frame, 0)
    # Make a copy of the flipped frame to show the user
    user_view = copy.copy(flipped_frame)
    # If the flag is enabled, draw all of the spell targets on the user's view
    if show_targets:
        for spell in spells:
            for step in spell.steps:
                pts = np.array(target_dimensions_hexagon(step), np.int32)
                pts = pts.reshape((-1, 1, 2))
                color = step["color"]
                thickness = 2
                isClosed = True
                userView = cv2.rectangle(flipped_frame, step["start_point"], step["start_point"], step["color"], 2)
                user_view = cv2.polyline(flipped_frame, [pts], isClosed, color, thickness)

    # Convert Flipped Frame to Grayscale
    grayFrame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)

    # Set threshold value and method
    ret, thresh = cv2.threshold(grayFrame, 240, 255, cv2.THRESH_TOZERO)

    # Find contours
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Register Centroids of Blobs
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            flipped_frame = cv2.circle(flipped_frame, (cX, cY), 20, (0, 0, 255), 3)
            mark_spell_progress(cX, cY)
        else:
            reset_progress()

    # Show Frame
    cv2.imshow('Camera', user_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
