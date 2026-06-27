""" Test the spacemouse output. """
import time
import numpy as np
from franka_env.spacemouse.spacemouse_expert import (
    SpaceMouseExpert,
    normalize_button_state,
)


def test_spacemouse():
    """Test the SpaceMouseExpert class.

    This interactive test prints the action and buttons of the spacemouse at a rate of 10Hz.
    The user is expected to move the spacemouse and press its buttons while the test is running.
    It keeps running until the user stops it.

    """
    spacemouse0 = SpaceMouseExpert()
    with np.printoptions(precision=3, suppress=True):
        while True:
            action, buttons = spacemouse0.get_action()
            logical_buttons = normalize_button_state(buttons, expected_buttons=2)
            print(
                "Spacemouse action:"
                f" {action}, raw buttons: {buttons}, logical buttons: {logical_buttons}"
            )
            time.sleep(0.1)


def main():
    """Call spacemouse test."""
    test_spacemouse()


if __name__ == "__main__":
    main()
