import unittest

from src.exercise_weights import EXERCISE_WEIGHTS, get_weights, resolve_exercise_name


class ExerciseAliasResolutionTests(unittest.TestCase):
    def test_shoulder_up_aliases_resolve_to_shoulder_press(self) -> None:
        self.assertEqual(resolve_exercise_name("shoulder_up"), "shoulder_press")
        self.assertEqual(resolve_exercise_name("Shoulder Up"), "shoulder_press")
        self.assertEqual(resolve_exercise_name("shoulderup"), "shoulder_press")
        self.assertEqual(resolve_exercise_name("shoulder raise"), "shoulder_press")
        self.assertEqual(resolve_exercise_name("arm_raises"), "shoulder_press")

    def test_shoulder_up_uses_curated_shoulder_press_weights(self) -> None:
        self.assertEqual(get_weights("shoulder_up"), EXERCISE_WEIGHTS["shoulder_press"])


if __name__ == "__main__":
    unittest.main()
