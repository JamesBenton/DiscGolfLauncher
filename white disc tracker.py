
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
# STEP 2A: REAL-WORLD CALIBRATION
# ==========================================

# Truck: 2006 Dodge Ram 2500 Mega Cab
#
# Assumed truck length = 247.7 inches
#
# Measured pixel coordinates:
# Point 1 = (50, 633)
# Point 2 = (180, 621)
#
# Pixel distance:
#
# d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
#
# Truck length in meters:
#
# length_m = length_inches * 0.0254
#
# Scale:
#
# meters_per_pixel = length_m / d

truck_length_inches = 247.7

truck_x1 = 50
truck_y1 = 633

truck_x2 = 180
truck_y2 = 621

# Convert inches to meters
truck_length_meters = truck_length_inches * 0.0254

# Calculate truck length in pixels
truck_length_pixels = np.sqrt(
    (truck_x2 - truck_x1)**2
    + (truck_y2 - truck_y1)**2
)

if truck_length_pixels == 0:
    raise RuntimeError("Truck calibration points must differ.")

# Calculate approximate meters per pixel
meters_per_pixel = (
    truck_length_meters / truck_length_pixels
)

print("===================================")
print("TRUCK CALIBRATION")
print("===================================")

print("Truck length (in):", truck_length_inches)
print("Truck length (m):", round(truck_length_meters, 5))
print("Truck length (pixels):", round(truck_length_pixels, 3))
print("Meters per pixel:", round(meters_per_pixel, 6))

print("===================================")

# ==========================================
# STEP 3: VARIABLES
# ==========================================

positions = []
trajectory = []

frame_number = 0
tracking = False
paused = False

# NEW: Keep the final frame open after playback
video_finished = False

previous_gray = None
previous_position = None
velocity = np.array([0.0, 0.0])

missed_frames = 0
max_missed_frames = 8

# ==========================================
# REAL-WORLD DISTANCE VARIABLES
# ==========================================

# Starting position of each detected flight
start_position = None

# Last measured disc position
last_measured_position = None

# Total distance along tracked path
total_distance_meters = 0.0

# Distances displayed on the video
display_linear_distance = 0.0
display_total_distance = 0.0

# ==========================================
# DRAW DASHED LINE
# ==========================================

def draw_dashed_line(
    image,
    start_point,
    end_point,
    color=(255, 0, 0),
    thickness=2,
    dash_length=12,
    gap_length=8
):

    start = np.array(start_point, dtype=float)
    end = np.array(end_point, dtype=float)

    direction = end - start
    length = np.linalg.norm(direction)

    if length == 0:
        return

    direction = direction / length

    for distance in np.arange(
        0,
        length,
        dash_length + gap_length
    ):

        dash_start = start + direction * distance

        dash_end = start + direction * min(
            distance + dash_length,
            length
        )

        cv2.line(
            image,
            tuple(np.round(dash_start).astype(int)),
            tuple(np.round(dash_end).astype(int)),
            color,
            thickness
        )

# ==========================================
# MOUSE PIXEL COORDINATES
# ==========================================

def get_pixel_coordinates(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel coordinates: X = {x}, Y = {y}")


cv2.namedWindow("Automatic Disc Tracking", cv2.WINDOW_NORMAL)

cv2.setMouseCallback(
    "Automatic Disc Tracking",
    get_pixel_coordinates
)

# ==========================================
# STEP 4: PROCESS VIDEO
# ==========================================

while True:

    # ======================================
    # NEW: KEEP VIDEO OPEN AFTER IT ENDS
    # ======================================

    if video_finished:

        # The final frame remains displayed.
        # No new frames are read or saved.
        # Press Q whenever you want to exit.

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            print("Closing video.")
            break

        continue

    # ======================================
    # PAUSE / RESUME CONTROL
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

            start_position = None
            last_measured_position = None
            total_distance_meters = 0.0

            display_linear_distance = 0.0
            display_total_distance = 0.0

            print("Tracking reset.")

        continue

    success, frame = video.read()

    if not success:

        print("===================================")
        print("VIDEO FINISHED")
        print("Final frame will remain on screen.")
        print("Press Q to exit.")
        print("===================================")

        # NEW: Do not break or close the window
        video_finished = True

        # Finalize the MP4 without closing the display
        video_writer.release()

        continue

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

            # Initial flight position
            start_position = (x, y)
            last_measured_position = (x, y)

            total_distance_meters = 0.0

            display_linear_distance = 0.0
            display_total_distance = 0.0

            # Start green trajectory immediately
            trajectory.append(previous_position)

            # Save initial point to CSV
            positions.append([
                frame_number,
                round(time, 4),
                int(x),
                int(y),
                round(area, 2),
                0.0,
                0.0,
                0.0,
                0.0
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

            # ==================================
            # CONVERT PIXELS TO METERS
            # ==================================

            x_meters = (
                (x - start_position[0])
                * meters_per_pixel
            )

            # Positive Y points upward in image
            y_meters = (
                (start_position[1] - y)
                * meters_per_pixel
            )

            # ==================================
            # LINEAR DISTANCE
            # ==================================

            # Straight-line distance from launch
            linear_distance_meters = np.sqrt(
                x_meters**2 + y_meters**2
            )

            # ==================================
            # TOTAL PATH DISTANCE
            # ==================================

            dx_pixels = (
                x - last_measured_position[0]
            )

            dy_pixels = (
                y - last_measured_position[1]
            )

            step_distance_pixels = np.sqrt(
                dx_pixels**2 + dy_pixels**2
            )

            step_distance_meters = (
                step_distance_pixels
                * meters_per_pixel
            )

            total_distance_meters += step_distance_meters

            last_measured_position = (x, y)

            # Update video measurements
            display_linear_distance = linear_distance_meters
            display_total_distance = total_distance_meters

            # Save measurements to CSV
            positions.append([
                frame_number,
                round(time, 4),
                int(x),
                int(y),
                round(area, 2),
                round(x_meters, 4),
                round(y_meters, 4),
                round(linear_distance_meters, 4),
                round(total_distance_meters, 4)
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

                start_position = None
                last_measured_position = None
                total_distance_meters = 0.0

                display_linear_distance = 0.0
                display_total_distance = 0.0

                # Clear trajectory for next launch
                trajectory = []

    # ======================================
    # STEP 8: DRAW TRAJECTORY
    # ======================================

    # Green line: actual tracked path
    for i in range(1, len(trajectory)):

        cv2.line(
            display,
            trajectory[i - 1],
            trajectory[i],
            (0, 255, 0),
            2
        )

    # ======================================
    # PURPLE DASHED LINE
    # ======================================

    # Draw a straight line from launch point
    # to latest confirmed disc position

    if start_position is not None and len(trajectory) > 0:

        final_point = trajectory[-1]

        draw_dashed_line(
            display,
            start_position,
            final_point,
            color=(0, 0, 255),
            thickness=2,
            dash_length=12,
            gap_length=8
        )

        # Blue dot marks launch position
        cv2.circle(
            display,
            start_position,
            6,
            (255, 0, 0),
            -1
        )

    # ======================================
    # DRAW RED CIRCLE AROUND DISC
    # ======================================

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

    # Draw yellow launch detection area
    cv2.rectangle(
        display,
        (launch_x1, launch_y1),
        (launch_x2, launch_y2),
        (0, 255, 255),
        2
    )

    # ======================================
    # DISPLAY TRACKING STATUS
    # ======================================

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

    # ======================================
    # DISPLAY DISTANCES ON VIDEO
    # ======================================

    # Purple linear distance from launch point
    cv2.putText(
        display,
        f"Linear Distance: {display_linear_distance:.2f} m",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # Green total distance along tracked path
    cv2.putText(
        display,
        f"Total Path Distance: {display_total_distance:.2f} m",
        (25, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ======================================
    # DISPLAY VIDEO
    # ======================================

    cv2.imshow("Automatic Disc Tracking", display)
    cv2.imshow("Moving White Detection", moving_white)

    # ======================================
    # SAVE TRACKING VIDEO FRAME
    # ======================================

    video_writer.write(display)

    frame_number += 1

    # ======================================
    # KEYBOARD CONTROLS
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

        start_position = None
        last_measured_position = None
        total_distance_meters = 0.0

        display_linear_distance = 0.0
        display_total_distance = 0.0

        print("Tracking reset.")

# ==========================================
# STEP 9: SAVE DATA AND VIDEO
# ==========================================

video.release()

# Only release if not already finalized at video end
if not video_finished:
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
        "Area (px^2)",
        "X (m)",
        "Y (m)",
        "Linear Distance (m)",
        "Total Path Distance (m)"
    ])

    writer.writerows(positions)

print("===================================")
print("TRACKING COMPLETE")
print("===================================")

print("Tracked positions:", len(positions))
print("Meters per pixel:", round(meters_per_pixel, 6))
print("CSV saved to:", output_file)
print("Tracking video saved to:", output_video_path)
