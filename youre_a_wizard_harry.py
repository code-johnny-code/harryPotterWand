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


def target_dimensions_box(target):
    top_left = target["start_point"]
    top_right = ((top_left[0] + target_size_factor), top_left[1])
    bottom_right = (top_right[0], (top_left[1] + target_size_factor))
    bottom_left = (top_left[0], bottom_right[1])
    return [top_left, top_right, bottom_right, bottom_left]


def target_dimensions_hexagon(target):
    point_1 = target["start_point"]
    point_2 = [(point_1[0] + target_size_factor), point_1[1]]
    point_3 = [(point_2[0] + target_size_factor), (point_2[1] + target_size_factor)]
    point_4 = [point_3[0], (point_3[1] + target_size_factor)]
    point_5 = [(point_4[0] - target_size_factor), (point_4[1] + target_size_factor)]
    point_6 = [(point_5 - target_size_factor), point_5[1]]
    point_7 = [(point_6[0] - target_size_factor), (point_6[1] - target_size_factor)]
    point_8 = [point_7[0], (point_7[1] - target_size_factor)]
    return [point_1, point_2, point_3, point_4, point_5, point_6, point_7, point_8]


def spell_target_hit(x, y):
    for target in targets:
        # Get the target box vertices
        target_box = target_dimensions_box(target)
        top_edge = target_box[0][1]
        bottom_edge = target_box[2][0]
        left_edge = target_box[0][0]
        right_edge = target_box[1][0]
        # If the y coordinate is between the top and bottom edges,
        # and the x coordinate is between the left and right edges, it's in the box
        in_box = top_edge <= y <= bottom_edge and left_edge <= x <= right_edge
        if in_box:
            return target["name"]
        else:
            return False


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
    # Start with the assumption that the hit isn't valid
    valid_hit = False
    # Check to see if the coordinates intersect any targets
    target_name = spell_target_hit(x, y)
    # If a target was hit, it's not the most recent target to be hit, and it wasn't hit earlier
    if target_name and (target_name != user_progress[-1]) and (target_name not in user_progress):
        # Determine what the index of the next target will be
        potential_index = len(user_progress)
        # Check only spells that are possible given the user's current progress
        for possible_spell in possible_spells():
            # Use the potential index to determine valid next target
            next_target = possible_spell.sequence[potential_index]
            # If the correct target was hit
            if next_target == target_name:
                # Mark that there has been a successful hit
                valid_hit = True
                # Add the target to the current user progress
                user_progress.append(target_name)
                if show_targets:
                    # Update the color of the target to show that it was successfully registered
                    for target in targets:
                        if target["name"] == target_name:
                            target["color"] = (0, 255, 0)
                # Check to see if user has successfully completed a spell sequence
                if user_progress == possible_spell.sequence:
                    print(possible_spell['name'])
                    reset_progress()
        # If the coordinates hit a target but it wasn't in any of the possible spells, start over
        if not valid_hit:
            reset_progress()


def reset_progress():
    # Empty the user's progress
    user_progress.clear()
    # If the targets are drawn on the user's view, set the colors back to default
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
                start_point = step["start_point"]
                end_point = ((start_point[0] + target_size_factor), (start_point[1] + target_size_factor))
                pts = np.array(target_dimensions_hexagon(step), np.int32)
                pts = pts.reshape((-1, 1, 2))
                color = step["color"]
                thickness = 2
                isClosed = True
                userView = cv2.rectangle(flipped_frame, start_point, end_point, color, 2)
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
