import cv2
import numpy as np


class ColorTracker:
    """Track fighter colors based on wrist tape and assign consistently"""
    
    def __init__(self):
        # HSV range for red tape
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        # HSV range for blue tape
        self.blue_lower = np.array([100, 100, 100])
        self.blue_upper = np.array([130, 255, 255])
        
        # HSV range for green tape
        self.green_lower = np.array([40, 100, 100])
        self.green_upper = np.array([80, 255, 255])
        
        # Assign fighters to colors
        self.fighter_colors = {}  # {fighter_idx: color}
        self.color_assigned = False

    def get_dominant_color_at_wrist(self, frame, wrist_pos, roi_size=30):
        # get the color around the wrist 
        # Blue and Red only
        if wrist_pos is None:
            return 'unknown', 0
        #Extract Wrist Coordinates
        x, y = int(wrist_pos[0]), int(wrist_pos[1])
        h, w = frame.shape[:2]
        
        # create a area of interest around wrist
        # anything outside frame is clipped
        x1 = max(0, x - roi_size)
        x2 = min(w, x + roi_size)
        y1 = max(0, y - roi_size)
        y2 = min(h, y + roi_size)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 'unknown', 0
        
        # OpenCV reads images in BGR, but color detection works much better in HSV.
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Count pixels for each color and create a mask
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_count = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
        
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        blue_count = cv2.countNonZero(blue_mask)
        
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        green_count = cv2.countNonZero(green_mask)
        
        # Return dominant color
        counts = {
            'red': red_count,
            'blue': blue_count,
            'green': green_count
        }
        
        max_color = max(counts, key=counts.get)
        max_count = counts[max_color]
        
        # Debug print
        print(f"  Detected - Red: {red_count}, Blue: {blue_count}, Green: {green_count} -> {max_color.upper()}")
        
        # Only return color if significant pixels found
        if max_count > 10:
            return max_color, max_count
        
        return 'unknown', max_count

    def assign_fighter_colors(self, frames_list, positions):
        
        print("\n" + "="*50)
        print("DETECTING FIGHTER COLORS...")
        print("="*50)
        
        detected_colors = []
        color_counts = {}
        
        # Detect color for each fighter
        #split the frame and position of the fighter
        #append the color and count to the list
        for i, (frame, pos) in enumerate(zip(frames_list, positions)):
            print(f"Fighter {i+1}:")
            color, count = self.get_dominant_color_at_wrist(frame, pos['wrist'])
            detected_colors.append(color)
            color_counts[i] = (color, count)
            print(f"  -> Assigned: {color.upper()}\n")
        
        # Assign colors - prevent duplicates
        assigned = set()
        
        # First pass: assign detected colors
        #store the color if not assigned
        for i, (color, count) in color_counts.items():
            if color != 'unknown' and color not in assigned:
                self.fighter_colors[i] = color
                assigned.add(color)
        
        # Second pass: assign remaining fighters different colors
        #then loop through the remaining fighters and assign them colors not taken
        color_priority = ['blue', 'red', 'green']
        for i in range(len(frames_list)):
            if i not in self.fighter_colors:
                for color in color_priority:
                    if color not in assigned:
                        self.fighter_colors[i] = color
                        assigned.add(color)
                        break
        
        self.color_assigned = True
        return self.fighter_colors
    
    #helper to get the color of the fighter
    def get_fighter_color(self, fighter_idx):
        """Get assigned color for a fighter"""
        return self.fighter_colors.get(fighter_idx, 'unknown')

    def draw_color_indicator(self, frame, wrist_pos, color, radius=15):
       
        if wrist_pos is None:
            return
        
        x, y = int(wrist_pos[0]), int(wrist_pos[1])
        
        color_map = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'unknown': (128, 128, 128)
        }
        
        bgr_color = color_map.get(color, (128, 128, 128))
        cv2.circle(frame, (x, y), radius, bgr_color, -1)
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)

    def print_fighter_assignments(self):
        """Print the color assignments for debugging"""
        print("\n" + "="*50)
        print("FINAL FIGHTER COLOR ASSIGNMENTS")
        print("="*50)
        for fighter_idx, color in sorted(self.fighter_colors.items()):
            print(f"Fighter {fighter_idx + 1}: {color.upper()}")
        print("="*50 + "\n")