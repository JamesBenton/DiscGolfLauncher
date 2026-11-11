
import cv2
import numpy as np
import csv

# ==========================================
# STEP 1: LOAD VIDEO
# ==========================================

video_path = r"C:\Users\Owner\Documents\disc_flight.mp4"

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise RuntimeError("Could not open video.")

fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise RuntimeError("Could not read FPS.")

delay = max(1, round(1000 / fps))

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ==========================================
# SAVE TRACKING VIDEO AS MP4
# ==========================================

output_video_path = r"C:\Users\Owner\Documents\disc_tracking_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (width, height)
)

if not video_writer.isOpened():
    video.release()
    raise RuntimeError("Could not create output video.")

# ==========================================
# STEP 2: DETECTION SETTINGS
# ==========================================

# White disc HSV range
lower_white = np.array([0, 0, 175])
upper_white = np.array([179, 70, 255])

# Detection area near the launcher
launch_x1 = int(width * 0.115)
launch_x2 = int(width * 0.285)

launch_y1 = int(height * 0.55)
launch_y2 = int(height * 0.90)

# Contour area limits
min_area = 3
max_area = 40

# Maximum movement allowed per frame
search_radius = 110

# Motion threshold
motion_threshold = 18

# ==========================================
# STEP 3: VARIABLES
# ==========================================

positions = []
trajectory = []

frame_number = 0
tracking = False

# NEW: Pause control
paused = False

previous_gray = None
previous_position = None
velocity = np.array([0.0, 0.0])

missed_frames = 0
max_missed_frames = 8

# ==========================================
# STEP 4: PROCESS VIDEO
# ==========================================

while True:

    # ======================================
    # NEW: PAUSE / RESUME CONTROL
    # ======================================

    if paused:

        key = cv2.waitKey(30) & 0xFF

        if key == 32:
            paused = False
            print("Video resumed")

        elif key == ord("q"):
            break

        elif key == ord("r"):

            tracking = False
            previous_position = None
            velocity = np.array([0.0, 0.0])
            missed_frames = 0

            trajectory = []
            positions = []

            print("Tracking reset.")

        continue

    success, frame = video.read()

    if not success:
        print("Video finished.")
        break

    time = frame_number / fps

    display = frame.copy()

    # Convert frame to HSV and grayscale
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # White detection
    white_mask = cv2.inRange(
        hsv,
        lower_white,
        upper_white
    )

    # ======================================
    # STEP 5: DETECT MOTION
    # ======================================

    motion_mask = np.zeros_like(gray)

    if previous_gray is not None:

        difference = cv2.absdiff(
            gray,
            previous_gray
        )

        _, motion_mask = cv2.threshold(
            difference,
            motion_threshold,
            255,
            cv2.THRESH_BINARY
        )

    previous_gray = gray.copy()

    # Find moving white objects
    moving_white = cv2.bitwise_and(
        white_mask,
        motion_mask
    )

    # ======================================
    # STEP 6: AUTOMATIC LAUNCH DETECTION
    # ======================================

    if not tracking:

        launch_mask = np.zeros_like(moving_white)

        launch_mask[
            launch_y1:launch_y2,
            launch_x1:launch_x2
        ] = 255

        detection_mask = cv2.bitwise_and(
            moving_white,
            launch_mask
        )

        contours, _ = cv2.findContours(
            detection_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if min_area <= area <= max_area:

                M = cv2.moments(contour)

                if M["m00"] != 0:

                    x = int(M["m10"] / M["m00"])
                    y = int(M["m01"] / M["m00"])

                    candidates.append((x, y, area))

        if candidates:

            # Initial selection favors rightmost object
            x, y, area = max(
                candidates,
                key=lambda c: c[0]
            )

            previous_position = (x, y)
            velocity = np.array([0.0, 0.0])


            tracking = True
            missed_frames = 0

            trajectory.append(previous_position)

# NEW: Save initial position to CSV
            positions.append([
                frame_number,
                round(time, 4),
                int(x),
                int(y),
                round(area, 2)
            ])

            print("Possible disc launch detected!")

    # ======================================
    # STEP 7: TRACK DISC
    # ======================================

    elif tracking:

        predicted = (
            np.array(previous_position, dtype=float)
            + velocity
        )

        search_mask = np.zeros_like(white_mask)

        cv2.circle(
            search_mask,
            tuple(np.round(predicted).astype(int)),
            search_radius,
            255,
            -1
        )

        local_mask = cv2.bitwise_and(
            white_mask,
            search_mask
        )

        contours, _ = cv2.findContours(
            local_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if min_area <= area <= max_area:

                M = cv2.moments(contour)

                if M["m00"] != 0:

                    x = M["m10"] / M["m00"]
                    y = M["m01"] / M["m00"]

                    distance = np.linalg.norm(
                        np.array([x, y]) - predicted
                    )

                    candidates.append(
                        (distance, x, y, area)
                    )

        if candidates:

            candidates.sort(key=lambda c: c[0])

            distance, x, y, area = candidates[0]

            current = np.array([x, y])

            new_velocity = (
                current -
                np.array(previous_position)
            )

            velocity = (
                0.6 * velocity +
                0.4 * new_velocity
            )

            previous_position = (int(x), int(y))

            trajectory.append(previous_position)

            positions.append([
                frame_number,
                round(time, 4),
                int(x),
                int(y),
                round(area, 2)
            ])

            missed_frames = 0

        else:

            missed_frames += 1

            previous_position = tuple(
                np.round(predicted).astype(int)
            )

            if missed_frames > max_missed_frames:

                print("Tracking lost. Waiting for new launch.")

                tracking = False
                previous_position = None
                velocity = np.array([0.0, 0.0])
                missed_frames = 0

    # ======================================
    # STEP 8: DRAW TRAJECTORY
    # ======================================

    for i in range(1, len(trajectory)):

        cv2.line(
            display,
            trajectory[i - 1],
            trajectory[i],
            (0, 255, 0),
            2
        )

    if tracking and previous_position is not None:

        cv2.circle(
            display,
            previous_position,
            12,
            (0, 0, 255),
            2
        )

        cv2.putText(
            display,
            "DISC",
            (
                previous_position[0] + 15,
                previous_position[1] - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # Draw launch detection area
    cv2.rectangle(
        display,
        (launch_x1, launch_y1),
        (launch_x2, launch_y2),
        (0, 255, 255),
        2
    )

    # Display status
    status = "TRACKING" if tracking else "WAITING FOR LAUNCH"

    cv2.putText(
        display,
        f"{status} | Time: {time:.2f} s",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Display video and motion mask
    cv2.imshow("Automatic Disc Tracking", display)
    cv2.imshow("Moving White Detection", moving_white)

    # ======================================
    # SAVE TRACKING VIDEO FRAME
    # ======================================

    video_writer.write(display)

    frame_number += 1

    # ======================================
    # NEW: KEYBOARD CONTROLS
    # ======================================

    key = cv2.waitKey(delay) & 0xFF

    if key == ord("q"):
        break

    elif key == 32:
        paused = True
        print("Video paused")

    elif key == ord("r"):

        tracking = False
        previous_position = None
        velocity = np.array([0.0, 0.0])
        missed_frames = 0

        trajectory = []
        positions = []

        print("Tracking reset.")

# ==========================================
# STEP 9: SAVE DATA AND VIDEO
# ==========================================

video.release()

video_writer.release()

cv2.destroyAllWindows()

output_file = r"C:\Users\Owner\Documents\disc_tracking_data.csv"

with open(output_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Frame",
        "Time (s)",
        "X (px)",
        "Y (px)",
        "Area (px^2)"
    ])

    writer.writerows(positions)

print("Tracking complete!")
print("Tracked positions:", len(positions))
print("CSV saved to:", output_file)
print("Tracking video saved to:", output_video_path)
