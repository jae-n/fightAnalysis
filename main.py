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


def calculate_angle(results, frame):
    """Calculate body angle from hip landmarks"""
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
    """Extract head and wrist positions from pose landmarks"""
    if not results.pose_landmarks:
        return None, None
    landmarks = results.pose_landmarks[0]
    h, w, _ = frame.shape
    head = (landmarks[0].x * w, landmarks[0].y * h)
    wrist = (landmarks[16].x * w, landmarks[16].y * h)
    return head, wrist


def download_model():
    """Download MediaPipe pose model if not exists"""
    model_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
    model_path = 'pose_landmarker_lite.task'
    
    if not os.path.exists(model_path):
        print("Downloading MediaPipe model...")
        urllib.request.urlretrieve(model_url, model_path)
    
    return model_path


def setup_pose_landmarker(model_path):
    """Initialize MediaPipe pose landmarker"""
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
    """Check if pose landmarks are detected"""
    return results.pose_landmarks and len(results.pose_landmarks) > 0


def main():
    # Setup
    model_path = download_model()
    landmarker = setup_pose_landmarker(model_path)
    cap = cv2.VideoCapture('video.mp4')
    
    # Ask user for number of fighters
    num_fighters = input("Enter number of fighters (2 or 3): ").strip()
    while num_fighters not in ['2', '3']:
        num_fighters = input("Invalid input. Enter 2 or 3: ").strip()
    
    num_fighters = int(num_fighters)
    
    # Initialize fighters
    fighters = [Fighter(f"Fighter {i+1}") for i in range(num_fighters)]
    
    # Initialize color tracker
    color_tracker = ColorTracker()
    
    # Initialize analyzers for each fighter
    strike_analyzers = [StrikeAnalyzer() for _ in range(num_fighters)]
    takedown_analyzers = [TakedownAnalyzer() for _ in range(num_fighters)]
    knockdown_analyzers = [KnockdownAnalyzer() for _ in range(num_fighters)]
    head_hit_detectors = [HeadHit() for _ in range(num_fighters)]
    body_hit_analyzers = [BodyHit() for _ in range(num_fighters)]

    # Tracking variables
    prev_time = time.time()
    prev_positions = [{'wrist': None, 'head_y': None} for _ in range(num_fighters)]
    
    frame_count = 0
    frames_with_all_fighters = 0

    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        
        # Split frame into sections based on number of fighters
        frame_width = width // num_fighters
        frames_list = [frame[:, i*frame_width:(i+1)*frame_width] for i in range(num_fighters)]

        # Detect poses for each fighter
        results_list = []
        all_detected = True
        
        for i, fighter_frame in enumerate(frames_list):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(fighter_frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_image)
            results_list.append(result)
            
            if not detect_people_in_frame(result):
                all_detected = False

        # Skip frame if not all fighters detected
        if not all_detected:
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frames_with_all_fighters += 1

        # Get positions for each fighter
        positions = []
        for i, result in enumerate(results_list):
            head, wrist = get_head_and_wrist(result, frames_list[i])
            angle = calculate_angle(result, frames_list[i])
            color = color_tracker.get_wrist_tape_color(frames_list[i], wrist)
            positions.append({
                'head': head,
                'wrist': wrist,
                'angle': angle,
                'head_y': head[1] if head else None,
                'color': color
            })

        # Calculate time delta
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        # Check all positions are valid
        if any(pos['head'] is None or pos['wrist'] is None for pos in positions):
            prev_positions = positions
            continue
        #note because i going to forget 
        # it detect action by previious and current wrist postion with head position knock back
        # ===== DETECT STRIKES =====
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
        
        # ===== DETECT TAKEDOWNS =====
        # ===== DETECT TAKEDOWNS =====
        # Check all fighter pairs for takedowns
        for attacker_idx in range(num_fighters):
            for defender_idx in range(num_fighters):
                if attacker_idx != defender_idx:
                    if takedown_analyzers[attacker_idx].detect_takedown(
                        positions[attacker_idx]['head'],
                        positions[defender_idx]['head'],
                        positions[attacker_idx]['head'],  # Using head as body proxy
                        positions[defender_idx]['head'],  # Using head as body proxy
                        frame_count
                    ):
                        fighters[attacker_idx].takedown += 1

        # ===== DETECT KNOCKDOWNS =====
        for i in range(num_fighters):
            if knockdown_analyzers[i].detect_knockdown(positions[i]['angle'], frame_count):
                fighters[i].knockdown += 1

        # ===== DETECT HEAD HITS =====
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

        # ===== DETECT BODY HITS =====
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

        # Update all scores
        for fighter in fighters:
            fighter.update_score()

        # Display results
        display_text = get_score(fighters)
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frames with all fighters: {frames_with_all_fighters}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw wrist tape colors
        for i, pos in enumerate(positions):
            color_tracker.draw_color_indicator(
                frame, 
                (pos['wrist'][0] + (i * frame_width), pos['wrist'][1]) if pos['wrist'] else None,
                pos['color']
            )
            # Display color label
            if pos['wrist']:
                x = int(pos['wrist'][0]) + (i * frame_width)
                y = int(pos['wrist'][1]) - 20
                cv2.putText(frame, f"Fighter {i+1}: {pos['color'].upper()}", 
                           (x - 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('UFC Fight Analysis', frame)

        # Update previous positions
        prev_positions = positions
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    # Print final results
    print("\n" + "="*50)
    print("FIGHT RESULTS")
    print("="*50)
    for fighter in fighters:
        print(f"{fighter.name}: Score={fighter.score}, Strikes={fighter.strike}, "
              f"Takedowns={fighter.takedown}, Knockdowns={fighter.knockdown}")
    print("="*50)
    print(get_winner(fighters))
    print("="*50)


if __name__ == "__main__":
    main()