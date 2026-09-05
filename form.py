import cv2
import mediapipe as mp
import time
import sys
import numpy as np
from types import SimpleNamespace

from exercise_analyzer import ExerciseAnalyzer, landmark_to_full_frame


# =========================================================
# MEDIAPIPE
# =========================================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=2
)

# Preserved from original setup (index 1). Change here if your
# camera is at a different index.
CAMERA_INDEX = 1

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"ERROR: Could not open camera at index {CAMERA_INDEX}.")
    sys.exit(1)


# =========================================================
# SETTINGS
# =========================================================

# Exercise-analysis thresholds (angle bands, visibility, tempo,
# elbow-stability) now live in exercise_analyzer.py. MIN_ROI_SIZE is
# a desktop ROI-selection concept and stays here.
MIN_ROI_SIZE = 60


# =========================================================
# RESPONSIVE UI SCALING
# =========================================================

# All UI text/button positions and sizes below are authored against
# this virtual reference resolution, then mapped onto whatever the
# actual camera frame size is. This keeps a single, centralized place
# (make_ui_scale + ui_point/ui_size/ui_thickness) responsible for
# scaling, instead of ad-hoc math scattered across every draw call.
REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 720


def make_ui_scale(frame_w, frame_h):

    return SimpleNamespace(
        sx=frame_w / REFERENCE_WIDTH,
        sy=frame_h / REFERENCE_HEIGHT,
        s=min(frame_w / REFERENCE_WIDTH, frame_h / REFERENCE_HEIGHT)
    )


def ui_point(ui, x, y):

    return (
        int(round(x * ui.sx)),
        int(round(y * ui.sy))
    )


def ui_size(ui, size):

    return size * ui.s


def ui_thickness(ui, thickness):

    if thickness < 0:
        return thickness

    return max(1, int(round(thickness * ui.s)))


# =========================================================
# DRAW TEXT HELPER
# =========================================================

def draw_text(
    frame,
    text,
    ui,
    position,
    size=0.7,
    color=(255, 255, 255),
    thickness=2
):

    cv2.putText(
        frame,
        text,
        ui_point(ui, *position),
        cv2.FONT_HERSHEY_SIMPLEX,
        ui_size(ui, size),
        color,
        ui_thickness(ui, thickness),
        cv2.LINE_AA
    )


# =========================================================
# DRAW RECT HELPER (reference-space UI boxes)
# =========================================================

def draw_rect(frame, ui, pt1, pt2, color, thickness=2):

    cv2.rectangle(
        frame,
        ui_point(ui, *pt1),
        ui_point(ui, *pt2),
        color,
        ui_thickness(ui, thickness)
    )


# =========================================================
# INSTRUCTIONS SCREEN CONTENT
# =========================================================

# One entry per page: (subtitle, [(line_text, color), ...]).
# Static reference content only - no exercise logic here.
INSTRUCTIONS_PAGES = [
    (
        "1. Before You Start",
        [
            ("For best accuracy, follow these setup tips:", (0, 255, 255)),
            ("", (220, 220, 220)),
            ("Make sure your device has a working camera.", (220, 220, 220)),
            ("Keep the camera lens clean.", (220, 220, 220)),
            ("Place the camera on a stable surface.", (220, 220, 220)),
            ("Position yourself so your upper body and the", (220, 220, 220)),
            ("selected arm are clearly visible.", (220, 220, 220)),
            ("Only one person should be in the camera frame.", (220, 220, 220)),
        ]
    ),
    (
        "2. Lighting",
        [
            ("Good lighting can help reduce detection errors.", (220, 220, 220)),
            ("", (220, 220, 220)),
            ("Recommended:", (0, 255, 255)),
            ("Use a well-lit environment.", (220, 220, 220)),
            ("Make sure your body is clearly illuminated.", (220, 220, 220)),
            ("Prefer light from in front of or slightly to the side.", (220, 220, 220)),
            ("Avoid strong shadows over your body.", (220, 220, 220)),
            ("Avoid very dark rooms.", (220, 220, 220)),
            ("Avoid an extremely bright light directly behind you.", (220, 220, 220)),
        ]
    ),
    (
        "3. Background and Clothing",
        [
            ("Use a relatively uncluttered background.", (220, 220, 220)),
            ("Avoid moving objects or other people behind you.", (220, 220, 220)),
            (
                "Clothing should contrast reasonably with the background.",
                (220, 220, 220)
            ),
            (
                "Avoid clothing that blends completely into the background.",
                (220, 220, 220)
            ),
        ]
    ),
    (
        "4. ROI - Region of Interest",
        [
            (
                "ROI tells the system which part of the camera frame",
                (220, 220, 220)
            ),
            ("to focus on.", (220, 220, 220)),
            ("", (220, 220, 220)),
            (
                "Selecting an appropriate ROI can help reduce",
                (220, 220, 220)
            ),
            (
                "interference from other objects or people in the frame.",
                (220, 220, 220)
            ),
            ("", (220, 220, 220)),
            ("See the next page for how to set it up.", (0, 255, 255)),
        ]
    ),
    (
        "4. ROI - Region of Interest (continued)",
        [
            ("How to use it:", (0, 255, 255)),
            ("1. Select the arm you want to train.", (220, 220, 220)),
            (
                "2. On the ROI screen, drag a rectangle around your arm.",
                (220, 220, 220)
            ),
            (
                "3. Make sure it contains the shoulder, elbow, and wrist.",
                (220, 220, 220)
            ),
            (
                "4. Keep the entire arm inside the ROI during the exercise.",
                (220, 220, 220)
            ),
            ("5. Do not make the ROI unnecessarily large.", (220, 220, 220)),
            ("6. Do not crop out the shoulder, elbow, or wrist.", (220, 220, 220)),
            ("7. If the ROI is wrong, use reset and select it again.", (220, 220, 220)),
            ("", (220, 220, 220)),
            (
                "Important: ROI does not guarantee accurate detection -",
                (0, 165, 255)
            ),
            (
                "it is intended to improve focus and reduce possible errors.",
                (0, 165, 255)
            ),
        ]
    ),
    (
        "5. During the Exercise",
        [
            ("Keep the selected arm visible.", (220, 220, 220)),
            (
                "Stay in approximately the same position relative to the camera.",
                (220, 220, 220)
            ),
            ("Keep the camera stationary.", (220, 220, 220)),
            ("Perform controlled biceps curls.", (220, 220, 220)),
            ("Avoid excessive body movement.", (220, 220, 220)),
            ("Avoid moving the elbow unnecessarily.", (220, 220, 220)),
            ("Keep the selected arm within the ROI.", (220, 220, 220)),
        ]
    ),
    (
        "6. Best Accuracy Checklist",
        [
            ("- Good lighting", (0, 255, 0)),
            ("- Stable camera", (0, 255, 0)),
            ("- Clear background", (0, 255, 0)),
            ("- One person in frame", (0, 255, 0)),
            ("- Selected arm clearly visible", (0, 255, 0)),
            ("- Shoulder, elbow, and wrist visible", (0, 255, 0)),
            ("- Correct ROI", (0, 255, 0)),
            ("- Controlled movement", (0, 255, 0)),
            ("- Camera position remains stable", (0, 255, 0)),
        ]
    ),
]


# =========================================================
# VARIABLES
# =========================================================

screen = "INSTRUCTIONS"

instructions_page = 0

selected_arm = None
target_reps = 10

rep_input = ""

# Display-only color per arm. Which MediaPipe landmarks are used for
# a given arm is now owned by exercise_analyzer.ExerciseAnalyzer.
ARM_COLORS = {
    "left": (0, 255, 0),
    "right": (255, 0, 255)
}

arm_color = ARM_COLORS["left"]

# The active workout session's analysis state. Created fresh each
# time an ROI is confirmed (see the ROI SELECTION keyboard handler).
analyzer = None


# =========================================================
# ROI VARIABLES
# =========================================================

roi_selecting = False

roi_drag_start = None
roi_drag_current = None

roi_pending = None
roi_confirmed = None

roi_message = ""

# Maps mouse coordinates from the (possibly letterboxed/resized)
# displayed window back onto the real camera frame. Updated once
# per frame in the DISPLAY section, consumed by roi_mouse_callback.
display_scale = 1.0
display_offset_x = 0
display_offset_y = 0


# =========================================================
# FUNCTION TO SET ARM
# =========================================================

def set_arm(arm):

    global selected_arm
    global arm_color

    selected_arm = arm
    arm_color = ARM_COLORS[arm]


# =========================================================
# ROI SELECTION
# =========================================================

WINDOW_NAME = "Biceps Curl Form Checker"


def normalize_rect(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    return (left, top, right, bottom)


def roi_mouse_callback(event, x, y, flags, param):

    global roi_selecting
    global roi_drag_start
    global roi_drag_current
    global roi_pending
    global roi_message

    if screen != "ROI_SELECT":
        return

    # The window may be showing a resized/letterboxed version of the
    # camera frame (see fit_frame_to_window). Map the raw window
    # coordinates back onto the underlying camera frame before using
    # them, so ROI selection logic keeps working in real camera pixels.
    cam_x = int((x - display_offset_x) / display_scale)
    cam_y = int((y - display_offset_y) / display_scale)

    cam_x = max(0, min(cam_x, w - 1))
    cam_y = max(0, min(cam_y, h - 1))

    if event == cv2.EVENT_LBUTTONDOWN:

        roi_selecting = True
        roi_drag_start = (cam_x, cam_y)
        roi_drag_current = (cam_x, cam_y)
        roi_pending = None
        roi_message = ""

    elif event == cv2.EVENT_MOUSEMOVE:

        if roi_selecting:
            roi_drag_current = (cam_x, cam_y)

    elif event == cv2.EVENT_LBUTTONUP:

        if roi_selecting:

            roi_selecting = False
            roi_drag_current = (cam_x, cam_y)
            roi_pending = normalize_rect(roi_drag_start, roi_drag_current)


def fit_frame_to_window(frame, window_name):

    frame_h, frame_w = frame.shape[:2]

    try:
        rect = cv2.getWindowImageRect(window_name)
    except cv2.error:
        rect = None

    if not rect or rect[2] <= 0 or rect[3] <= 0:
        return frame, 1.0, 0, 0

    win_w, win_h = rect[2], rect[3]

    scale = min(win_w / frame_w, win_h / frame_h)

    if scale <= 0:
        return frame, 1.0, 0, 0

    new_w = max(1, int(frame_w * scale))
    new_h = max(1, int(frame_h * scale))

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((win_h, win_w, 3), dtype=frame.dtype)

    offset_x = (win_w - new_w) // 2
    offset_y = (win_h - new_h) // 2

    canvas[
        offset_y:offset_y + new_h,
        offset_x:offset_x + new_w
    ] = resized

    return canvas, scale, offset_x, offset_y


cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, roi_mouse_callback)


# =========================================================
# MAIN LOOP
# =========================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # Mirror camera
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        ui = make_ui_scale(w, h)


        # =====================================================
        # INSTRUCTIONS SCREEN
        # =====================================================

        if screen == "INSTRUCTIONS":

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (w, h),
                (20, 20, 20),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.85,
                frame,
                0.15,
                0
            )


            draw_text(
                frame,
                "HOW TO USE",
                ui,
                (REFERENCE_WIDTH // 2 - 135, 50),
                1.05,
                (0, 255, 255),
                3
            )

            draw_text(
                frame,
                f"Page {instructions_page + 1}/{len(INSTRUCTIONS_PAGES)}",
                ui,
                (REFERENCE_WIDTH - 180, 45),
                0.55,
                (180, 180, 180)
            )


            subtitle, lines = INSTRUCTIONS_PAGES[instructions_page]

            draw_text(
                frame,
                subtitle,
                ui,
                (80, 115),
                0.9,
                (255, 255, 255),
                2
            )

            line_y = 160

            for line_text, line_color in lines:

                draw_text(
                    frame,
                    line_text,
                    ui,
                    (80, line_y),
                    0.6,
                    line_color,
                    1
                )

                line_y += 34


            draw_text(
                frame,
                "N: Next Page    B: Previous Page",
                ui,
                (80, REFERENCE_HEIGHT - 75),
                0.55,
                (180, 180, 180)
            )

            draw_text(
                frame,
                "ENTER: Continue to Arm Selection",
                ui,
                (80, REFERENCE_HEIGHT - 40),
                0.6,
                (0, 255, 0),
                2
            )

            draw_text(
                frame,
                "Press Q to quit",
                ui,
                (REFERENCE_WIDTH - 200, REFERENCE_HEIGHT - 40),
                0.55,
                (180, 180, 180)
            )


        # =====================================================
        # ARM SELECTION SCREEN
        # =====================================================

        elif screen == "ARM_SELECT":

            # Dark overlay
            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (w, h),
                (20, 20, 20),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.75,
                frame,
                0.25,
                0
            )


            draw_text(
                frame,
                "BICEPS CURL FORM CHECKER",
                ui,
                (REFERENCE_WIDTH // 2 - 250, 100),
                1.1,
                (0, 255, 255),
                3
            )

            draw_text(
                frame,
                "Choose which arm you want to train",
                ui,
                (REFERENCE_WIDTH // 2 - 220, 170),
                0.7
            )

            # Left box
            draw_rect(
                frame,
                ui,
                (150, 230),
                (REFERENCE_WIDTH // 2 - 30, 380),
                (0, 255, 0),
                3
            )

            draw_text(
                frame,
                "LEFT ARM",
                ui,
                (220, 315),
                1,
                (0, 255, 0),
                3
            )

            draw_text(
                frame,
                "Press L",
                ui,
                (250, 350),
                0.6
            )


            # Right box
            draw_rect(
                frame,
                ui,
                (REFERENCE_WIDTH // 2 + 30, 230),
                (REFERENCE_WIDTH - 150, 380),
                (255, 0, 255),
                3
            )

            draw_text(
                frame,
                "RIGHT ARM",
                ui,
                (REFERENCE_WIDTH // 2 + 80, 315),
                1,
                (255, 0, 255),
                3
            )

            draw_text(
                frame,
                "Press R",
                ui,
                (REFERENCE_WIDTH // 2 + 110, 350),
                0.6
            )


            draw_text(
                frame,
                "Press Q to quit",
                ui,
                (REFERENCE_WIDTH // 2 - 80, REFERENCE_HEIGHT - 50),
                0.6,
                (180, 180, 180)
            )


        # =====================================================
        # REP INPUT SCREEN
        # =====================================================

        elif screen == "REP_INPUT":

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (w, h),
                (20, 20, 20),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.75,
                frame,
                0.25,
                0
            )


            draw_text(
                frame,
                "HOW MANY REPS?",
                ui,
                (REFERENCE_WIDTH // 2 - 160, 120),
                1.1,
                (0, 255, 255),
                3
            )


            # Input box
            draw_rect(
                frame,
                ui,
                (REFERENCE_WIDTH // 2 - 150, 180),
                (REFERENCE_WIDTH // 2 + 150, 270),
                (255, 255, 255),
                2
            )


            input_display = rep_input

            if input_display == "":
                input_display = "_"


            draw_text(
                frame,
                input_display,
                ui,
                (REFERENCE_WIDTH // 2 - 20, 240),
                1.2,
                (255, 255, 255),
                2
            )


            draw_text(
                frame,
                "Type the number using your keyboard",
                ui,
                (REFERENCE_WIDTH // 2 - 180, 320),
                0.6
            )

            draw_text(
                frame,
                "Press ENTER to start",
                ui,
                (REFERENCE_WIDTH // 2 - 130, 360),
                0.7,
                (0, 255, 0),
                2
            )

            draw_text(
                frame,
                "Press BACKSPACE to correct",
                ui,
                (REFERENCE_WIDTH // 2 - 150, 400),
                0.6
            )

            draw_text(
                frame,
                "Press Q to quit",
                ui,
                (REFERENCE_WIDTH // 2 - 80, REFERENCE_HEIGHT - 50),
                0.6,
                (180, 180, 180)
            )


        # =====================================================
        # ROI SELECTION SCREEN
        # =====================================================

        elif screen == "ROI_SELECT":

            draw_text(
                frame,
                "SELECT REGION OF INTEREST",
                ui,
                (REFERENCE_WIDTH // 2 - 250, 40),
                0.9,
                (0, 255, 255),
                2
            )

            draw_text(
                frame,
                "Click and drag to draw a box around your arm",
                ui,
                (REFERENCE_WIDTH // 2 - 260, 70),
                0.6
            )

            if (
                roi_selecting
                and roi_drag_start is not None
                and roi_drag_current is not None
            ):

                cv2.rectangle(
                    frame,
                    roi_drag_start,
                    roi_drag_current,
                    (20, 20, 20),
                    2
                )

            elif roi_pending is not None:

                px1, py1, px2, py2 = roi_pending

                cv2.rectangle(
                    frame,
                    (px1, py1),
                    (px2, py2),
                    (0, 255, 0),
                    2
                )

                draw_text(
                    frame,
                    "Press ENTER/SPACE to confirm, R to reset",
                    ui,
                    (REFERENCE_WIDTH // 2 - 260, REFERENCE_HEIGHT - 70),
                    0.6,
                    (0, 255, 0)
                )

            else:

                draw_text(
                    frame,
                    "No region selected yet",
                    ui,
                    (REFERENCE_WIDTH // 2 - 150, REFERENCE_HEIGHT - 70),
                    0.6,
                    (200, 200, 200)
                )

            if roi_message != "":

                draw_text(
                    frame,
                    roi_message,
                    ui,
                    (REFERENCE_WIDTH // 2 - 200, REFERENCE_HEIGHT - 100),
                    0.6,
                    (0, 0, 255),
                    2
                )

            draw_text(
                frame,
                "Press Q to quit",
                ui,
                (REFERENCE_WIDTH // 2 - 80, REFERENCE_HEIGHT - 30),
                0.6,
                (180, 180, 180)
            )


        # =====================================================
        # WORKOUT SCREEN
        # =====================================================

        elif screen == "WORKOUT":

            if roi_confirmed is None:

                draw_text(
                    frame,
                    "No ROI selected",
                    ui,
                    (REFERENCE_WIDTH // 2 - 130, 60),
                    0.8,
                    (0, 0, 255),
                    2
                )


            else:

                roi_x1, roi_y1, roi_x2, roi_y2 = roi_confirmed

                roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

                rgb_frame = cv2.cvtColor(
                    roi_frame,
                    cv2.COLOR_BGR2RGB
                )

                results = pose.process(rgb_frame)

                cv2.rectangle(
                    frame,
                    (roi_x1, roi_y1),
                    (roi_x2, roi_y2),
                    (255, 255, 0),
                    1
                )


                # =================================================
                # POSE FOUND
                # =================================================

                if results.pose_landmarks:

                    landmarks = results.pose_landmarks.landmark

                    shoulder = landmark_to_full_frame(
                        landmarks[analyzer.shoulder_id], roi_confirmed, w, h
                    )
                    elbow = landmark_to_full_frame(
                        landmarks[analyzer.elbow_id], roi_confirmed, w, h
                    )
                    wrist = landmark_to_full_frame(
                        landmarks[analyzer.wrist_id], roi_confirmed, w, h
                    )

                    result = analyzer.update(
                        shoulder, elbow, wrist, time.time()
                    )


                    # =============================================
                    # VISIBILITY
                    # =============================================

                    if result.visible:

                        # =========================================
                        # DRAW ARM
                        # =========================================

                        shoulder_pixel = (
                            int(shoulder.x * w),
                            int(shoulder.y * h)
                        )

                        elbow_pixel = (
                            int(elbow.x * w),
                            int(elbow.y * h)
                        )

                        wrist_pixel = (
                            int(wrist.x * w),
                            int(wrist.y * h)
                        )


                        cv2.line(
                            frame,
                            shoulder_pixel,
                            elbow_pixel,
                            arm_color,
                            4
                        )

                        cv2.line(
                            frame,
                            elbow_pixel,
                            wrist_pixel,
                            arm_color,
                            4
                        )


                        cv2.circle(
                            frame,
                            shoulder_pixel,
                            9,
                            arm_color,
                            -1
                        )

                        cv2.circle(
                            frame,
                            elbow_pixel,
                            9,
                            arm_color,
                            -1
                        )

                        cv2.circle(
                            frame,
                            wrist_pixel,
                            9,
                            arm_color,
                            -1
                        )


                        # =========================================
                        # TOP INFO
                        # =========================================

                        draw_text(
                            frame,
                            f"{selected_arm.upper()} ARM",
                            ui,
                            (30, 40),
                            0.7,
                            arm_color,
                            2
                        )

                        draw_text(
                            frame,
                            f"Reps: {result.reps}/{target_reps}",
                            ui,
                            (30, 80),
                            0.9,
                            (255, 255, 255),
                            2
                        )

                        draw_text(
                            frame,
                            f"Angle: {int(result.angle)}",
                            ui,
                            (30, 120),
                            0.7,
                            arm_color,
                            2
                        )


                        # =========================================
                        # STAGE
                        # =========================================

                        if result.stage == "up":

                            stage_text = "CURLING"

                        else:

                            stage_text = "EXTENDED"


                        draw_text(
                            frame,
                            f"Stage: {stage_text}",
                            ui,
                            (30, 160),
                            0.7,
                            (255, 255, 255),
                            2
                        )


                        # =========================================
                        # GOOD / BAD
                        # =========================================

                        draw_text(
                            frame,
                            f"Good: {result.good_reps}",
                            ui,
                            (30, 200),
                            0.65,
                            (0, 255, 0),
                            2
                        )

                        draw_text(
                            frame,
                            f"Bad: {result.bad_reps}",
                            ui,
                            (30, 235),
                            0.65,
                            (0, 0, 255),
                            2
                        )

                        draw_text(
                            frame,
                            f"Partial: {analyzer.partial_reps}",
                            ui,
                            (30, 270),
                            0.65,
                            (0, 165, 255),
                            2
                        )


                        # =========================================
                        # LIVE TEMPO
                        # =========================================

                        if (
                            analyzer.stage == "up"
                            and analyzer.rep_start_time is not None
                        ):

                            current_tempo = (
                                time.time()
                                - analyzer.rep_start_time
                            )

                            draw_text(
                                frame,
                                f"Tempo: {current_tempo:.1f}s",
                                ui,
                                (30, 305),
                                0.65,
                                (255, 255, 0),
                                2
                            )


                        # =========================================
                        # LAST FEEDBACK
                        # =========================================

                        draw_text(
                            frame,
                            "Last:",
                            ui,
                            (REFERENCE_WIDTH - 300, 50),
                            0.65,
                            (255, 255, 255),
                            2
                        )

                        draw_text(
                            frame,
                            result.last_feedback,
                            ui,
                            (REFERENCE_WIDTH - 300, 85),
                            0.55,
                            (0, 255, 255),
                            2
                        )


                    else:

                        draw_text(
                            frame,
                            "ARM NOT FULLY VISIBLE: "
                            + ", ".join(result.missing_parts),
                            ui,
                            (REFERENCE_WIDTH // 2 - 260, 60),
                            0.7,
                            (0, 0, 255),
                            2
                        )


                else:

                    draw_text(
                        frame,
                        "NO PERSON DETECTED",
                        ui,
                        (REFERENCE_WIDTH // 2 - 150, 60),
                        0.8,
                        (0, 0, 255),
                        2
                    )


            # =============================================
            # WORKOUT INSTRUCTIONS
            # =============================================

            draw_text(
                frame,
                "Press Q to quit",
                ui,
                (REFERENCE_WIDTH - 180, REFERENCE_HEIGHT - 25),
                0.5,
                (180, 180, 180),
                1
            )


            # =================================================
            # WORKOUT FINISHED
            # =================================================

            if analyzer.reps >= target_reps:

                screen = "RESULTS"


        # =====================================================
        # RESULTS SCREEN
        # =====================================================

        elif screen == "RESULTS":

            # Dark background
            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (w, h),
                (15, 15, 15),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.85,
                frame,
                0.15,
                0
            )


            # =================================================
            # CALCULATE SCORE
            # =================================================

            summary = analyzer.get_results(target_reps)


            # =================================================
            # TITLE
            # =================================================

            draw_text(
                frame,
                "WORKOUT COMPLETE!",
                ui,
                (REFERENCE_WIDTH // 2 - 220, 65),
                1.0,
                (0, 255, 255),
                3
            )


            draw_text(
                frame,
                f"{selected_arm.upper()} ARM",
                ui,
                (REFERENCE_WIDTH // 2 - 60, 105),
                0.65,
                arm_color,
                2
            )


            # =================================================
            # SCORE
            # =================================================

            draw_text(
                frame,
                f"{summary.final_score:.1f}/10",
                ui,
                (REFERENCE_WIDTH // 2 - 80, 175),
                1.3,
                (0, 255, 0),
                3
            )

            draw_text(
                frame,
                "FORM SCORE",
                ui,
                (REFERENCE_WIDTH // 2 - 75, 210),
                0.55,
                (180, 180, 180),
                1
            )


            # =================================================
            # STATISTICS
            # =================================================

            left_x = 100
            right_x = 500

            y = 275

            draw_text(
                frame,
                f"Completed: {summary.reps}/{summary.target_reps}",
                ui,
                (left_x, y),
                0.65
            )

            draw_text(
                frame,
                f"Good reps: {summary.good_reps}",
                ui,
                (left_x, y + 35),
                0.65,
                (0, 255, 0)
            )

            draw_text(
                frame,
                f"Bad reps: {summary.bad_reps}",
                ui,
                (left_x, y + 70),
                0.65,
                (0, 0, 255)
            )

            draw_text(
                frame,
                f"Partial reps: {summary.partial_reps}",
                ui,
                (left_x, y + 105),
                0.65,
                (0, 165, 255)
            )


            draw_text(
                frame,
                f"Full reps: {summary.full_reps}",
                ui,
                (right_x, y),
                0.65
            )

            draw_text(
                frame,
                f"Very short: {summary.very_short_reps}",
                ui,
                (right_x, y + 35),
                0.65
            )

            draw_text(
                frame,
                f"Average tempo: {summary.average_tempo:.2f}s",
                ui,
                (right_x, y + 70),
                0.65,
                (255, 255, 0)
            )

            draw_text(
                frame,
                f"Tempo issues: {summary.tempo_problems}",
                ui,
                (right_x, y + 105),
                0.65
            )


            # =================================================
            # FEEDBACK
            # =================================================

            draw_text(
                frame,
                "FEEDBACK",
                ui,
                (100, 450),
                0.75,
                (0, 255, 255),
                2
            )


            # =================================================
            # DRAW FEEDBACK
            # =================================================

            feedback_y = 485

            for line in summary.feedback_lines[:5]:

                draw_text(
                    frame,
                    line,
                    ui,
                    (100, feedback_y),
                    0.55,
                    (230, 230, 230),
                    1
                )

                feedback_y += 30


            # =================================================
            # CONTROLS
            # =================================================

            draw_text(
                frame,
                "Press R to restart",
                ui,
                (REFERENCE_WIDTH // 2 - 120, REFERENCE_HEIGHT - 60),
                0.65,
                (0, 255, 0),
                2
            )

            draw_text(
                frame,
                "Press Q to quit",
                ui,
                (REFERENCE_WIDTH // 2 - 100, REFERENCE_HEIGHT - 25),
                0.55,
                (180, 180, 180),
                1
            )


        # =====================================================
        # DISPLAY
        # =====================================================

        display_frame, display_scale, display_offset_x, display_offset_y = (
            fit_frame_to_window(frame, WINDOW_NAME)
        )

        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )


        # =====================================================
        # KEYBOARD INPUT
        # =====================================================

        key = cv2.waitKey(1) & 0xFF


        # =====================================================
        # QUIT
        # =====================================================

        if key == ord("q"):

            break


        # =====================================================
        # INSTRUCTIONS
        # =====================================================

        if screen == "INSTRUCTIONS":

            if key == ord("n"):

                if instructions_page < len(INSTRUCTIONS_PAGES) - 1:

                    instructions_page += 1

            elif key == ord("b"):

                if instructions_page > 0:

                    instructions_page -= 1

            elif key == 13:

                screen = "ARM_SELECT"


        # =====================================================
        # ARM SELECTION
        # =====================================================

        elif screen == "ARM_SELECT":

            if key == ord("l"):

                set_arm("left")

                screen = "REP_INPUT"

                rep_input = ""


            elif key == ord("r"):

                set_arm("right")

                screen = "REP_INPUT"

                rep_input = ""


        # =====================================================
        # REP INPUT
        # =====================================================

        elif screen == "REP_INPUT":

            # Numbers 0-9
            if ord("0") <= key <= ord("9"):

                rep_input += chr(key)


            # Backspace
            elif key == 8:

                rep_input = rep_input[:-1]


            # Enter
            elif key == 13:

                if rep_input != "":

                    try:

                        target_reps = int(
                            rep_input
                        )

                        if target_reps > 0:

                            roi_selecting = False
                            roi_drag_start = None
                            roi_drag_current = None
                            roi_pending = None
                            roi_confirmed = None
                            roi_message = ""

                            screen = "ROI_SELECT"

                    except ValueError:

                        rep_input = ""


        # =====================================================
        # ROI SELECTION
        # =====================================================

        elif screen == "ROI_SELECT":

            # Confirm (ENTER or SPACE)
            if key == 13 or key == 32:

                if roi_pending is not None:

                    px1, py1, px2, py2 = roi_pending

                    px1 = max(0, min(px1, w))
                    px2 = max(0, min(px2, w))
                    py1 = max(0, min(py1, h))
                    py2 = max(0, min(py2, h))

                    if (
                        (px2 - px1) >= MIN_ROI_SIZE
                        and (py2 - py1) >= MIN_ROI_SIZE
                    ):

                        roi_confirmed = (px1, py1, px2, py2)

                        analyzer = ExerciseAnalyzer(selected_arm)

                        screen = "WORKOUT"

                    else:

                        roi_message = "ROI too small - drag a larger box"

                else:

                    roi_message = "Draw a region first"


            # Reset
            elif key == ord("r"):

                roi_selecting = False
                roi_drag_start = None
                roi_drag_current = None
                roi_pending = None
                roi_message = ""


        # =====================================================
        # RESULTS RESTART
        # =====================================================

        elif screen == "RESULTS":

            if key == ord("r"):

                screen = "ARM_SELECT"

                rep_input = ""

                selected_arm = None


finally:

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
