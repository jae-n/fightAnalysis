import cv2
import numpy as np


class ColorTracker:
    """Track fighter colors based on wrist tape"""
    
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

    def get_dominant_color_at_wrist(self, frame, wrist_pos, roi_size=30):
        """
        Get dominant color at wrist position
        
        Args:
            frame: image frame
            wrist_pos: (x, y) position of wrist
            roi_size: size of region of interest around wrist
            
        Returns:
            color string: 'red', 'blue', 'green', or 'unknown'
        """
        if wrist_pos is None:
            return 'unknown'
        
        x, y = int(wrist_pos[0]), int(wrist_pos[1])
        h, w = frame.shape[:2]
        
        # Define ROI around wrist
        x1 = max(0, x - roi_size)
        x2 = min(w, x + roi_size)
        y1 = max(0, y - roi_size)
        y2 = min(h, y + roi_size)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 'unknown'
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Count pixels for each color
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
        
        # Only return color if significant pixels found
        if max_count > 10:
            return max_color
        
        return 'unknown'

    def get_wrist_tape_color(self, frame, wrist_pos):
        """Simplified version - just get color at wrist"""
        return self.get_dominant_color_at_wrist(frame, wrist_pos)

    def draw_color_indicator(self, frame, wrist_pos, color, radius=15):
        """
        Draw a colored circle at wrist position
        
        Args:
            frame: image to draw on
            wrist_pos: (x, y) position
            color: color string ('red', 'blue', 'green')
            radius: circle radius
        """
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