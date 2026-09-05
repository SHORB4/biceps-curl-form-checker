"""
Standalone test proving exercise_analyzer.py operates independently
of the desktop UI (form.py) - no camera, no OpenCV window/GUI calls.

Run directly:
    python test_analyzer.py

This test does NOT import form.py (form.py opens the camera and a
window as soon as it's imported, since it has no __main__ guard).
"""

import sys
import math


# ---------------------------------------------------------------
# INDEPENDENCE CHECK, part 1: exercise_analyzer.py's own source must
# not reference any OpenCV GUI/camera function.
# ---------------------------------------------------------------

with open("exercise_analyzer.py", encoding="utf-8") as f:
    _analyzer_source = f.read()

_forbidden_calls = [
    "cv2.VideoCapture", "cv2.imshow", "cv2.waitKey",
    "cv2.namedWindow", "cv2.setMouseCallback", "import cv2",
]
_found = [c for c in _forbidden_calls if c in _analyzer_source]
assert not _found, (
    f"exercise_analyzer.py source references forbidden GUI/camera "
    f"calls: {_found}"
)

print("[OK] exercise_analyzer.py source contains no cv2 import and "
      "no camera/GUI/window function references.")


# ---------------------------------------------------------------
# INDEPENDENCE CHECK, part 2: actually running the analyzer must
# never CALL a camera/GUI function.
#
# Note: `import mediapipe` alone already pulls the cv2 *module* into
# memory as a transitive dependency of MediaPipe's Solutions API -
# this happens even before exercise_analyzer.py is touched, and is
# not something this extraction introduced or can avoid while using
# mediapipe.solutions.pose. That is a different, much weaker claim
# than "opens a camera" or "uses OpenCV GUI functions", which is what
# actually matters here - so instead of checking whether cv2 was
# imported, we monkeypatch the actual GUI/camera entry points and
# assert none of them are ever invoked while exercising the analyzer.
# ---------------------------------------------------------------

import cv2  # noqa: E402  (already loaded transitively via mediapipe)

_gui_calls_made = []


def _forbid(name):
    def _blocked(*args, **kwargs):
        _gui_calls_made.append(name)
        raise AssertionError(f"cv2.{name} was called - not independent!")
    return _blocked


cv2.VideoCapture = _forbid("VideoCapture")
cv2.imshow = _forbid("imshow")
cv2.waitKey = _forbid("waitKey")
cv2.namedWindow = _forbid("namedWindow")
cv2.setMouseCallback = _forbid("setMouseCallback")

import exercise_analyzer as ea  # noqa: E402
from exercise_analyzer import ExerciseAnalyzer  # noqa: E402

assert not _gui_calls_made, (
    f"Importing exercise_analyzer triggered: {_gui_calls_made}"
)

print("[OK] Importing exercise_analyzer calls none of "
      "VideoCapture/imshow/waitKey/namedWindow/setMouseCallback "
      "(these are now trip-wired to fail loudly if ever invoked).")


from types import SimpleNamespace  # noqa: E402


# ---------------------------------------------------------------
# Synthetic landmark geometry helpers
# ---------------------------------------------------------------
# Fixed shoulder/elbow, with the wrist placed on a circle around the
# elbow so the included angle at the elbow (shoulder-elbow-wrist) is
# exactly the requested angle in degrees. This exercises the REAL
# calculate_angle() math through the public update() interface,
# rather than poking internal state directly.

SHOULDER = (0.5, 0.2)
ELBOW = (0.5, 0.5)


def wrist_for_angle(angle_deg, elbow_x=ELBOW[0], radius=0.3):
    """Returns a wrist (x, y) such that calculate_angle(shoulder,
    (elbow_x, ELBOW[1]), wrist) == angle_deg. Computed relative to the
    ACTUAL elbow position passed in (not a fixed constant), since a
    test case may deliberately move the elbow to simulate drift."""
    elbow = (elbow_x, ELBOW[1])
    shoulder_theta = math.degrees(
        math.atan2(SHOULDER[1] - elbow[1], SHOULDER[0] - elbow[0])
    )
    theta = math.radians(shoulder_theta + angle_deg)
    return (
        elbow[0] + radius * math.cos(theta),
        elbow[1] + radius * math.sin(theta)
    )


def lm(x, y, visibility=0.9):
    return SimpleNamespace(x=x, y=y, visibility=visibility)


def shoulder_lm(visibility=0.9):
    return lm(SHOULDER[0], SHOULDER[1], visibility)


def elbow_lm(x=ELBOW[0], visibility=0.9):
    return lm(x, ELBOW[1], visibility)


def wrist_lm(angle_deg, elbow_x=ELBOW[0], visibility=0.9):
    x, y = wrist_for_angle(angle_deg, elbow_x=elbow_x)
    return lm(x, y, visibility)


def check_angle(actual, expected, tol=0.5):
    assert abs(actual - expected) < tol, (
        f"angle mismatch: got {actual}, expected ~{expected}"
    )


# =================================================================
# 1. STARTING / RESTING STATE
# =================================================================

analyzer = ExerciseAnalyzer("left")

assert analyzer.reps == 0
assert analyzer.good_reps == 0
assert analyzer.bad_reps == 0
assert analyzer.stage == "down"
assert analyzer.last_feedback == "Get ready..."

print("[OK] Fresh session starts at rest: reps=0, stage='down'.")


# =================================================================
# 2. ARM VISIBILITY (insufficient visibility should not progress
#    the state machine at all)
# =================================================================

r = analyzer.update(
    shoulder_lm(visibility=0.9),
    elbow_lm(visibility=0.9),
    wrist_lm(180, visibility=0.1),   # wrist barely visible
    timestamp=0.0
)

assert r.visible is False
assert r.missing_parts == ["wrist"]
assert r.angle is None
assert analyzer.stage == "down"      # unchanged
assert analyzer.reps == 0            # unchanged

print("[OK] Low-visibility frame reports visible=False, "
      f"missing_parts={r.missing_parts}, and leaves state untouched.")


# =================================================================
# 3. EXTENSION (resting, fully visible) - stage should remain "down"
# =================================================================

r = analyzer.update(shoulder_lm(), elbow_lm(), wrist_lm(180), timestamp=0.1)
check_angle(r.angle, 180)
assert r.visible is True
assert r.stage == "down"
assert analyzer.reps == 0

print(f"[OK] Extended arm (angle={r.angle:.1f}) keeps stage='down'.")


# =================================================================
# 4. FULL CURL then RETURN TO EXTENSION -> one FULL, GOOD rep
# =================================================================

r = analyzer.update(shoulder_lm(), elbow_lm(x=0.50), wrist_lm(10), timestamp=0.5)
check_angle(r.angle, 10)
assert r.stage == "up"

r = analyzer.update(shoulder_lm(), elbow_lm(x=0.50), wrist_lm(180), timestamp=2.5)
check_angle(r.angle, 180)
assert r.stage == "down"
assert r.reps == 1
assert analyzer.good_reps == 1
assert analyzer.full_reps == 1
assert analyzer.elbow_problems == 0
assert analyzer.tempo_problems == 0
assert r.last_feedback == "Good rep!"

print(f"[OK] Full curl -> extend produced rep #1: GOOD, FULL, "
      f"feedback='{r.last_feedback}'.")


# =================================================================
# 5. MULTIPLE REPS with varied quality (partial ROM + too-fast tempo,
#    then very-short ROM + elbow drift + too-slow tempo)
# =================================================================

# Rep 2: partial ROM (~95 deg), too fast (0.4s)
analyzer.update(shoulder_lm(), elbow_lm(x=0.50), wrist_lm(95), timestamp=2.6)
r = analyzer.update(shoulder_lm(), elbow_lm(x=0.50), wrist_lm(180), timestamp=3.0)
assert r.reps == 2
assert analyzer.partial_reps == 1
assert analyzer.too_fast_reps == 1
assert "Curl further" in r.last_feedback
assert "Slow down" in r.last_feedback
assert analyzer.bad_reps == 1

print(f"[OK] Partial + too-fast rep #2 classified correctly: "
      f"'{r.last_feedback}'.")

# Rep 3: very-short ROM (~125 deg), elbow drifts 0.20 (> 0.12 threshold),
# too slow (5.0s)
analyzer.update(
    shoulder_lm(), elbow_lm(x=0.50), wrist_lm(125, elbow_x=0.50),
    timestamp=3.1
)
r = analyzer.update(
    shoulder_lm(), elbow_lm(x=0.70), wrist_lm(180, elbow_x=0.70),
    timestamp=8.1
)
assert r.reps == 3
assert analyzer.very_short_reps == 1
assert analyzer.elbow_problems == 1
assert analyzer.too_slow_reps == 1
assert "Use more range" in r.last_feedback
assert "Keep elbow stable" in r.last_feedback
assert "Use a steady tempo" in r.last_feedback
assert analyzer.bad_reps == 2

print(f"[OK] Very-short + unstable-elbow + too-slow rep #3 classified "
      f"correctly: '{r.last_feedback}'.")

print(f"[OK] Multiple reps accumulate correctly: reps={analyzer.reps}, "
      f"good={analyzer.good_reps}, bad={analyzer.bad_reps}.")


# =================================================================
# 6. get_results() - final scoring, callable repeatedly (no side effects)
# =================================================================

summary_1 = analyzer.get_results(target_reps=3)
summary_2 = analyzer.get_results(target_reps=3)

assert summary_1.reps == 3
assert summary_1.target_reps == 3
assert summary_1.good_reps == 1
assert summary_1.bad_reps == 2
assert summary_1.full_reps == 1
assert summary_1.partial_reps == 1
assert summary_1.very_short_reps == 1
assert summary_1.tempo_problems == 2

expected_avg_tempo = (2.0 + 0.4 + 5.0) / 3
assert abs(summary_1.average_tempo - expected_avg_tempo) < 1e-9

expected_score = (
    (1 / 3 * 10) * 0.7
    + ((3 - 2) / 3 * 10) * 0.3
)
assert abs(summary_1.final_score - expected_score) < 1e-9

# calling it twice must not change analyzer state or the result
assert summary_1.reps == summary_2.reps
assert summary_1.final_score == summary_2.final_score
assert analyzer.reps == 3  # untouched by get_results()

print(f"[OK] get_results(): score={summary_1.final_score:.2f}/10, "
      f"avg_tempo={summary_1.average_tempo:.2f}s, is idempotent, "
      f"and feedback_lines has {len(summary_1.feedback_lines)} entries.")

for line in summary_1.feedback_lines:
    print("       ", line)


# =================================================================
# 7. SESSION INDEPENDENCE between two analyzer instances
# =================================================================

analyzer_a = ExerciseAnalyzer("left")
analyzer_b = ExerciseAnalyzer("right")

# Different arms must resolve to different (mirror-corrected) landmark ids
assert analyzer_a.shoulder_id != analyzer_b.shoulder_id
assert analyzer_a.elbow_id != analyzer_b.elbow_id
assert analyzer_a.wrist_id != analyzer_b.wrist_id

# Drive analyzer_a through one full rep; analyzer_b must stay untouched
analyzer_a.update(shoulder_lm(), elbow_lm(), wrist_lm(10), timestamp=0.0)
analyzer_a.update(shoulder_lm(), elbow_lm(), wrist_lm(180), timestamp=2.0)

assert analyzer_a.reps == 1
assert analyzer_b.reps == 0
assert analyzer_b.stage == "down"
assert analyzer_b.last_feedback == "Get ready..."

print("[OK] Two ExerciseAnalyzer instances are fully independent - "
      f"analyzer_a.reps={analyzer_a.reps}, analyzer_b.reps={analyzer_b.reps} "
      "(no shared/module-level workout state).")


# =================================================================
# 8. "RESET" = constructing a new instance
# =================================================================

analyzer_c = ExerciseAnalyzer("left")

assert analyzer_c.reps == 0
assert analyzer_a.reps == 1  # the old session is unaffected

print("[OK] Starting a new session (new ExerciseAnalyzer instance) "
      "is unaffected by a prior session's accumulated state - this "
      "is the reset mechanism (no dedicated .reset() method exists).")


# =================================================================
# Final independence re-check
# =================================================================

assert not _gui_calls_made, (
    f"Something during the test triggered forbidden GUI/camera "
    f"calls: {_gui_calls_made}"
)

print("[OK] No camera/GUI/window function was called at any point "
      "during the entire synthetic workout above.")

print()
print("ALL CHECKS PASSED - exercise_analyzer.py operates correctly "
      "without a camera, without OpenCV GUI, and independently of form.py.")
