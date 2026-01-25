import numpy as np


class StrikeAnalyzer:
    def __init__(self, velocity_threshold=500, distance_threshold=50):
        self.velocity_threshold = velocity_threshold
        self.distance_threshold = distance_threshold
        self.last_strike_frame = -100  # Track last strike detection

    def detect_strike(self, prev_point, current_point, opponent_head, time_diff, frame_count):
        if prev_point is None or current_point is None or time_diff <= 0:
            return False

        # Cooldown: don't detect same strike twice (minimum 15 frames between strikes)
        if frame_count - self.last_strike_frame < 5:
            return False

        velocity = np.linalg.norm(
            np.array(current_point) - np.array(prev_point)
        ) / time_diff

        distance = np.linalg.norm(
            np.array(current_point) - np.array(opponent_head)
        )

        # Both conditions must be met AND velocity must be significant
        if velocity > self.velocity_threshold and distance < self.distance_threshold:
            self.last_strike_frame = frame_count
            return True
        
        return False


class TakedownAnalyzer:
    def __init__(self, height_diff_threshold=80):
        self.height_diff_threshold = height_diff_threshold
        self.last_takedown_frame = -100
        self.in_takedown = False

    def detect_takedown(self, prev_height, current_height, frame_count):
        if prev_height is None or current_height is None:
            return False

        # Cooldown: don't detect same takedown twice
        if frame_count - self.last_takedown_frame < 30:
            return False

        height_diff = prev_height - current_height
        
        # Only detect if there's a significant height drop
        if height_diff > self.height_diff_threshold:
            self.last_takedown_frame = frame_count
            return True
        
        return False


class SubmissionAnalyzer:
    def __init__(self, pressure_threshold=200):
        self.pressure_threshold = pressure_threshold

    def is_submission_attempt(self, pressure):
        return pressure > self.pressure_threshold


class KnockdownAnalyzer:
    def __init__(self, angle_threshold=60):
        self.angle_threshold = angle_threshold
        self.last_knockdown_frame = -100

    def detect_knockdown(self, angle, frame_count):
        # Cooldown: don't detect knockdowns too frequently
        if frame_count - self.last_knockdown_frame < 25:
            return False

        # Body angle must be very tilted (knocked down)
        if angle > self.angle_threshold:
            self.last_knockdown_frame = frame_count
            return True
        
        return False


class HeadHit:
    def __init__(self):
        self.prev_wrist = None
        self.prev_head = None
        self.prev_head_velocity = None
        self.last_hit_frame = -100

    def detect(self, attacker_wrist, defender_head, dt, frame_count):
        if self.prev_wrist is None or self.prev_head is None or dt <= 0:
            self.prev_wrist = attacker_wrist
            self.prev_head = defender_head
            return False

        # Cooldown: prevent same hit from being counted multiple times
        if frame_count - self.last_hit_frame < 12:
            self.prev_wrist = attacker_wrist
            self.prev_head = defender_head
            return False

        # Wrist velocity to head
        velocity = np.linalg.norm(np.array(attacker_wrist) - np.array(self.prev_wrist)) / dt

        # Impact from wrist to head
        impact = velocity / dt

        # Head velocity (reaction)
        v_head = np.linalg.norm(np.array(defender_head) - np.array(self.prev_head)) / dt

        # Distance from wrist to head
        distance = np.linalg.norm(np.array(attacker_wrist) - np.array(defender_head))

        # Thresholds - much stricter now
        WRIST_VEL = 1200
        HEAD_ACCEL = 1500
        HIT_RADIUS = 70
        MIN_HEAD_VEL = 150

        hit = (
            velocity > WRIST_VEL and
            impact > HEAD_ACCEL and
            distance < HIT_RADIUS and
            v_head > MIN_HEAD_VEL
        )

        self.prev_wrist = attacker_wrist
        self.prev_head = defender_head
        self.prev_head_velocity = v_head

        if hit:
            self.last_hit_frame = frame_count
            return True

        return False


class BodyHit:
    def __init__(self):
        self.prev_wrist = None
        self.prev_body = None
        self.last_hit_frame = -100

    def detect(self, wrist, body, head_y, dt, frame_count):
        if self.prev_wrist is None or self.prev_body is None or dt <= 0:
            self.prev_wrist = wrist
            self.prev_body = body
            return False

        # Cooldown: prevent same hit from being counted multiple times
        if frame_count - self.last_hit_frame < 12:
            self.prev_wrist = wrist
            self.prev_body = body
            return False

        # Wrist velocity (px/s)
        wrist_velocity = np.linalg.norm(
            np.array(wrist) - np.array(self.prev_wrist)
        ) / dt

        # Body velocity (reaction)
        body_velocity = np.linalg.norm(
            np.array(body) - np.array(self.prev_body)
        ) / dt

        # Distance wrist → body
        wrist_body_dist = np.linalg.norm(
            np.array(wrist) - np.array(body)
        )

        # Thresholds - stricter
        WRIST_VEL = 800
        BODY_VEL = 400
        HIT_RADIUS = 100
        MIN_BODY_VEL = 150

        hit = (
            wrist_velocity > WRIST_VEL and
            body_velocity > BODY_VEL and
            body_velocity > MIN_BODY_VEL and
            wrist_body_dist < HIT_RADIUS and
            head_y is not None and
            wrist[1] > head_y
        )

        self.prev_wrist = wrist
        self.prev_body = body

        if hit:
            self.last_hit_frame = frame_count
            return True

        return False