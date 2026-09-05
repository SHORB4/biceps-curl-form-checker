import math
from types import SimpleNamespace

import mediapipe as mp

mp_pose = mp.solutions.pose


# =========================================================
# SETTINGS (unchanged from the original form.py values)
# =========================================================

FULL_CURL_ANGLE = 80
HALF_CURL_ANGLE = 110
EXTEND_ANGLE = 140

VISIBILITY_THRESHOLD = 0.5

MIN_REP_TIME = 1.0
MAX_REP_TIME = 4.0

MAX_ELBOW_MOVEMENT = 0.12


# =========================================================
# ANGLE CALCULATION
# =========================================================

def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# =========================================================
# LANDMARK COORDINATE CONVERSION
# =========================================================

def landmark_to_full_frame(landmark, roi, frame_w, frame_h):

    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1

    full_x = (roi_x1 + landmark.x * roi_w) / frame_w
    full_y = (roi_y1 + landmark.y * roi_h) / frame_h

    return SimpleNamespace(
        x=full_x,
        y=full_y,
        visibility=landmark.visibility
    )


# =========================================================
# SELECTED-ARM LANDMARK SELECTION
# =========================================================

def landmark_ids_for_arm(arm):

    # The camera frame is typically mirrored (e.g. cv2.flip) before
    # MediaPipe runs, but MediaPipe's LEFT_*/RIGHT_* landmarks are
    # anatomical (the subject's real side), not image-position based.
    # Mirroring the input inverts that mapping, so callers using a
    # mirrored feed get the opposite MediaPipe landmark set here to
    # match the user's actual selected arm.
    if arm == "left":

        return (
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST
        )

    return (
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST
    )


# =========================================================
# EXERCISE ANALYZER
# =========================================================

class ExerciseAnalyzer:
    """
    Holds the state and logic for one biceps-curl workout session.

    Consumes already-normalized shoulder/elbow/wrist landmarks (plain
    objects with .x, .y, .visibility, in full-frame-normalized
    coordinates) plus a timestamp, and returns per-frame analysis
    results. Has no dependency on any camera, window, or GUI library.
    """

    def __init__(self, arm):

        self.arm = arm

        (
            self.shoulder_id,
            self.elbow_id,
            self.wrist_id
        ) = landmark_ids_for_arm(arm)

        self.stage = "down"

        self.rep_start_time = None
        self.current_rep_min_angle = 180
        self.current_rep_elbow_start_x = None

        self.reps = 0
        self.good_reps = 0
        self.bad_reps = 0

        self.full_reps = 0
        self.partial_reps = 0
        self.very_short_reps = 0

        self.elbow_problems = 0
        self.tempo_problems = 0

        self.too_fast_reps = 0
        self.too_slow_reps = 0

        self.rep_tempos = []

        self.last_feedback = "Get ready..."

    def update(self, shoulder, elbow, wrist, timestamp):
        """
        shoulder/elbow/wrist: objects with .x, .y, .visibility in
        full-frame-normalized coordinates.
        timestamp: a float, e.g. time.time(), supplied by the caller.

        Returns a SimpleNamespace describing this frame's result:
        visible, missing_parts, angle, stage, reps, good_reps,
        bad_reps, last_feedback.
        """

        visible = (
            shoulder.visibility > VISIBILITY_THRESHOLD
            and elbow.visibility > VISIBILITY_THRESHOLD
            and wrist.visibility > VISIBILITY_THRESHOLD
        )

        if not visible:

            missing_parts = []

            if shoulder.visibility <= VISIBILITY_THRESHOLD:
                missing_parts.append("shoulder")

            if elbow.visibility <= VISIBILITY_THRESHOLD:
                missing_parts.append("elbow")

            if wrist.visibility <= VISIBILITY_THRESHOLD:
                missing_parts.append("wrist")

            return SimpleNamespace(
                visible=False,
                missing_parts=missing_parts,
                angle=None,
                stage=self.stage,
                reps=self.reps,
                good_reps=self.good_reps,
                bad_reps=self.bad_reps,
                last_feedback=self.last_feedback
            )

        shoulder_point = (shoulder.x, shoulder.y)
        elbow_point = (elbow.x, elbow.y)
        wrist_point = (wrist.x, wrist.y)

        angle = calculate_angle(
            shoulder_point,
            elbow_point,
            wrist_point
        )

        if angle < EXTEND_ANGLE:

            if self.stage == "down":

                self.stage = "up"

                self.rep_start_time = timestamp

                self.current_rep_min_angle = angle

                self.current_rep_elbow_start_x = elbow.x

            else:

                if angle < self.current_rep_min_angle:

                    self.current_rep_min_angle = angle

        elif angle > EXTEND_ANGLE:

            if self.stage == "up":

                self._complete_rep(elbow, timestamp)

        return SimpleNamespace(
            visible=True,
            missing_parts=[],
            angle=angle,
            stage=self.stage,
            reps=self.reps,
            good_reps=self.good_reps,
            bad_reps=self.bad_reps,
            last_feedback=self.last_feedback
        )

    def _complete_rep(self, elbow, timestamp):

        self.reps += 1

        self.stage = "down"

        if self.rep_start_time is not None:

            rep_time = timestamp - self.rep_start_time

        else:

            rep_time = 0

        self.rep_tempos.append(rep_time)

        if self.current_rep_min_angle <= FULL_CURL_ANGLE:

            rep_type = "FULL"

            self.full_reps += 1

        elif self.current_rep_min_angle <= HALF_CURL_ANGLE:

            rep_type = "PARTIAL"

            self.partial_reps += 1

        else:

            rep_type = "VERY SHORT"

            self.very_short_reps += 1

        elbow_good = True

        if self.current_rep_elbow_start_x is not None:

            elbow_movement = abs(
                elbow.x - self.current_rep_elbow_start_x
            )

            if elbow_movement > MAX_ELBOW_MOVEMENT:

                elbow_good = False

                self.elbow_problems += 1

        tempo_good = True

        if rep_time < MIN_REP_TIME:

            tempo_good = False

            self.tempo_problems += 1
            self.too_fast_reps += 1

        elif rep_time > MAX_REP_TIME:

            tempo_good = False

            self.tempo_problems += 1
            self.too_slow_reps += 1

        rep_feedback = []

        if rep_type == "PARTIAL":

            rep_feedback.append("Curl further")

        elif rep_type == "VERY SHORT":

            rep_feedback.append("Use more range")

        if not elbow_good:

            rep_feedback.append("Keep elbow stable")

        if not tempo_good:

            if rep_time < MIN_REP_TIME:

                rep_feedback.append("Slow down")

            else:

                rep_feedback.append("Use a steady tempo")

        if (
            rep_type == "FULL"
            and elbow_good
            and tempo_good
        ):

            self.good_reps += 1

        else:

            self.bad_reps += 1

        if len(rep_feedback) == 0:

            rep_feedback.append("Good rep!")

        self.last_feedback = " | ".join(rep_feedback)

        self.rep_start_time = None

        self.current_rep_min_angle = 180

        self.current_rep_elbow_start_x = None

    def get_results(self, target_reps):
        """
        Computes the final score and qualitative feedback lines,
        mirroring the original RESULTS-screen computation exactly.
        """

        reps = self.reps

        if reps > 0:

            good_rep_ratio = self.good_reps / reps

            base_score = good_rep_ratio * 10

            if len(self.rep_tempos) > 0:

                average_tempo = (
                    sum(self.rep_tempos) / len(self.rep_tempos)
                )

            else:

                average_tempo = 0

            good_tempo_reps = reps - self.tempo_problems

            tempo_ratio = good_tempo_reps / reps

            final_score = (
                base_score * 0.7
                +
                (tempo_ratio * 10) * 0.3
            )

        else:

            average_tempo = 0
            final_score = 0

        feedback_lines = []

        # Range
        if self.partial_reps > 0:

            feedback_lines.append(
                f"- {self.partial_reps} partial rep(s): curl further"
            )

        if self.very_short_reps > 0:

            feedback_lines.append(
                f"- {self.very_short_reps} very short rep(s): "
                "use more range"
            )

        if (
            self.partial_reps == 0
            and self.very_short_reps == 0
        ):

            feedback_lines.append("- Excellent range of motion")

        # Elbow
        if self.elbow_problems > 0:

            feedback_lines.append(
                f"- Elbow moved too much in {self.elbow_problems} rep(s)"
            )

        else:

            feedback_lines.append("- Good elbow stability")

        # Tempo
        if self.too_fast_reps > 0:

            feedback_lines.append(
                f"- {self.too_fast_reps} rep(s) were too fast: slow down"
            )

        if self.too_slow_reps > 0:

            feedback_lines.append(
                f"- {self.too_slow_reps} rep(s) were too slow"
            )

        if self.tempo_problems == 0:

            feedback_lines.append("- Tempo was consistent")

        # Overall
        if final_score >= 9:

            feedback_lines.append("- Excellent overall technique!")

        elif final_score >= 8:

            feedback_lines.append(
                "- Great job! Your technique was mostly consistent."
            )

        elif final_score >= 7:

            feedback_lines.append(
                "- Good workout. A few areas can be improved."
            )

        elif final_score >= 5:

            feedback_lines.append(
                "- Decent effort. Focus on the feedback above."
            )

        else:

            feedback_lines.append(
                "- Focus on controlled movement and full range."
            )

        return SimpleNamespace(
            reps=reps,
            target_reps=target_reps,
            good_reps=self.good_reps,
            bad_reps=self.bad_reps,
            full_reps=self.full_reps,
            partial_reps=self.partial_reps,
            very_short_reps=self.very_short_reps,
            tempo_problems=self.tempo_problems,
            average_tempo=average_tempo,
            final_score=final_score,
            feedback_lines=feedback_lines
        )
