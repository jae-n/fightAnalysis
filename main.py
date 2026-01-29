import numpy as np
import cv2
import mediapipe as mp
import time
import os
import urllib.request

from utils import (
    StrikeAnalyzer,
    TakedownAnalyzer,
    KnockdownAnalyzer,
    HeadHit,
    BodyHit
)
from score import Fighter, get_score, get_winner
from color_tracker import ColorTracker

# Helper functions
def calculate_angle(results, frame):
    # Get body angle from hips
    if not results.pose_landmarks:
        return 0

    landmarks = results.pose_landmarks[0]
    h, w = frame.shape[:2]

    left_hip = landmarks[23]
    right_hip = landmarks[24]

    dx = (right_hip.x - left_hip.x) * w
    dy = (right_hip.y - left_hip.y) * h

    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)


def get_head_and_wrist(results, frame):
    # Get head and wrist positions
    if not results.pose_landmarks:
        return None, None
    #get the fighter
    landmarks = results.pose_landmarks[0]
    #get the head and wrist
    h, w, _ = frame.shape
    #mark it with the postion of the frame
    head = (landmarks[0].x * w, landmarks[0].y * h)
    wrist = (landmarks[16].x * w, landmarks[16].y * h)

    return head, wrist


def download_model():
    # Download pose model if needed
    model_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
    model_path = 'pose_landmarker_lite.task'

    if not os.path.exists(model_path):
        print("Downloading model...")
        urllib.request.urlretrieve(model_url, model_path)

    return model_path


def setup_pose_landmarker(model_path):
    #idk but this help
    # Initialize and return a MediaPipe PoseLandmarker for detecting human poses in images.
    # - `model_path` is the path to the pose detection model file.
    # - Sets the detector to IMAGE mode (single image processing).
    # - Returns a PoseLandmarker object ready to use for pose detection.
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )

    return PoseLandmarker.create_from_options(options)


def detect_people_in_frame(results):
    # Check if a person is detected
    return results.pose_landmarks and len(results.pose_landmarks) > 0


def main():
    # Setup model and video
    model_path = download_model()
    landmarker = setup_pose_landmarker(model_path)
    cap = cv2.VideoCapture('video.mp4')

    # Ask number of fighters
    num_fighters = input("Enter number of fighters (2 or 3): ").strip()
    while num_fighters not in ['2', '3']:
        num_fighters = input("Invalid input. Enter 2 or 3: ").strip()

    num_fighters = int(num_fighters)

    # Create fighters
    fighters = [Fighter(f"Fighter {i+1}") for i in range(num_fighters)]

    # Setup color tracker
    color_tracker = ColorTracker()

    # Create analyzers
    strike_analyzers = [StrikeAnalyzer() for _ in range(num_fighters)]
    takedown_analyzers = [TakedownAnalyzer() for _ in range(num_fighters)]
    knockdown_analyzers = [KnockdownAnalyzer() for _ in range(num_fighters)]
    head_hit_detectors = [HeadHit() for _ in range(num_fighters)]
    body_hit_analyzers = [BodyHit() for _ in range(num_fighters)]

    # Track time and previous positions
    prev_time = time.time()
    prev_positions = [{'wrist': None, 'head_y': None} for _ in range(num_fighters)]

    frame_count = 0
    frames_with_all_fighters = 0
    color_assignment_done = False

    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Split frame by fighter
        frame_width = width // num_fighters
        frames_list = [frame[:, i*frame_width:(i+1)*frame_width] for i in range(num_fighters)]

        # Detect poses
        results_list = []
        all_detected = True
        
    # Convert each fighter frame from OpenCV BGR format to MediaPipe Image format.
        for fighter_frame in frames_list:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(fighter_frame, cv2.COLOR_BGR2RGB)
            )

            # pose landmarker to detect poses in the current MediaPipe image
            result = landmarker.detect(mp_image)
            #append the image for later
            results_list.append(result)

            if not detect_people_in_frame(result):
                all_detected = False

        # Skip if not all detected
        if not all_detected:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frames_with_all_fighters += 1

        # Get positions
        positions = []
        for i, result in enumerate(results_list):
            head, wrist = get_head_and_wrist(result, frames_list[i])
            angle = calculate_angle(result, frames_list[i])

            positions.append({
                'head': head,
                'wrist': wrist,
                'angle': angle,
                'head_y': head[1] if head else None
            })

        # Assign colors once
        if not color_assignment_done:
            color_tracker.assign_fighter_colors(frames_list, positions)
            color_tracker.print_fighter_assignments()
            color_assignment_done = True

        # Time difference
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        # Skip if missing data
        if any(pos['head'] is None or pos['wrist'] is None for pos in positions):
            prev_positions = positions
            continue

        # Detect strikes
        #check for each fighter if they strike 
        #then add point to the striker
        #same for each funciton
        for attacker_idx in range(num_fighters):
            for defender_idx in range(num_fighters):
                if attacker_idx != defender_idx:
                    if strike_analyzers[attacker_idx].detect_strike(
                        prev_positions[attacker_idx]['wrist'],
                        positions[attacker_idx]['wrist'],
                        positions[defender_idx]['head'],
                        dt,
                        frame_count
                    ):
                        fighters[attacker_idx].strike += 1

        # Detect takedowns
        for attacker_idx in range(num_fighters):
            for defender_idx in range(num_fighters):
                if attacker_idx != defender_idx:
                    if takedown_analyzers[attacker_idx].detect_takedown(
                        positions[attacker_idx]['head'],
                        positions[defender_idx]['head'],
                        positions[attacker_idx]['head'],
                        positions[defender_idx]['head'],
                        frame_count
                    ):
                        fighters[attacker_idx].takedown += 1

        # Detect knockdowns
        for i in range(num_fighters):
            if knockdown_analyzers[i].detect_knockdown(
                positions[i]['angle'],
                positions[i]['head_y'],
                prev_positions[i]['head_y'],
                frame_count
            ):
                fighters[i].knockdown += 1

        # Detect head hits
        for attacker_idx in range(num_fighters):
            for defender_idx in range(num_fighters):
                if attacker_idx != defender_idx:
                    if head_hit_detectors[attacker_idx].detect(
                        positions[attacker_idx]['wrist'],
                        positions[defender_idx]['head'],
                        dt,
                        frame_count
                    ):
                        fighters[attacker_idx].score += 5

        # Detect body hits
        for attacker_idx in range(num_fighters):
            for defender_idx in range(num_fighters):
                if attacker_idx != defender_idx:
                    if body_hit_analyzers[attacker_idx].detect(
                        positions[attacker_idx]['wrist'],
                        positions[defender_idx]['head'],
                        positions[defender_idx]['head_y'],
                        dt,
                        frame_count
                    ):
                        fighters[attacker_idx].score += 3

        # Update scores
        for fighter in fighters:
            fighter.update_score()

        # Show score
        display_text = get_score(fighters)
        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame,
                    f"Frames with all fighters: {frames_with_all_fighters}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw wrist colors
        for i, pos in enumerate(positions):
            fighter_color = color_tracker.get_fighter_color(i)

            color_tracker.draw_color_indicator(
                frame,
                (pos['wrist'][0] + (i * frame_width), pos['wrist'][1]) if pos['wrist'] else None,
                fighter_color
            )

            if pos['wrist']:
                x = int(pos['wrist'][0]) + (i * frame_width)
                y = int(pos['wrist'][1]) - 20

                cv2.putText(frame,
                            f"Fighter {i+1}: {fighter_color.upper()}",
                            (x - 30, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1)

        cv2.imshow('UFC Fight Analysis', frame)

        prev_positions = positions

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    # Print results
    print("\nFIGHT RESULTS\n")

    for i, fighter in enumerate(fighters):
        fighter_color = color_tracker.get_fighter_color(i)
        print(f"{fighter.name} ({fighter_color.upper()}): "
              f"Score={fighter.score}, "
              f"Strikes={fighter.strike}, "
              f"Takedowns={fighter.takedown}, "
              f"Knockdowns={fighter.knockdown}")

    print("\nWinner:")
    print(get_winner(fighters))


if __name__ == "__main__":
    main()
