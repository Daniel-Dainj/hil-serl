import atexit
import multiprocessing
import numpy as np
from franka_env.spacemouse import pyspacemouse
from typing import Sequence, Tuple


UNIVERSAL_RECEIVER_BUTTON_COUNT = 15


def normalize_button_state(
    buttons: Sequence[int], expected_buttons: int
) -> Tuple[bool, ...]:
    """Map device-specific button arrays to logical left/right button pairs.

    Directly connected SpaceMouse models usually expose two buttons. Wireless
    devices behind the 3Dconnexion Universal Receiver report a 15-button layout
    where the physical left/right buttons appear as the first and last entries.
    """
    normalized = [bool(value) for value in buttons]
    if expected_buttons <= 0:
        return tuple()
    if not normalized:
        return tuple(False for _ in range(expected_buttons))

    if (
        expected_buttons % 2 == 0
        and len(normalized) >= UNIVERSAL_RECEIVER_BUTTON_COUNT
        and len(normalized) % UNIVERSAL_RECEIVER_BUTTON_COUNT == 0
    ):
        logical_buttons = []
        for start in range(0, len(normalized), UNIVERSAL_RECEIVER_BUTTON_COUNT):
            chunk = normalized[start : start + UNIVERSAL_RECEIVER_BUTTON_COUNT]
            logical_buttons.extend((chunk[0], chunk[-1]))
        if len(logical_buttons) >= expected_buttons:
            return tuple(logical_buttons[:expected_buttons])

    padded = normalized[:expected_buttons]
    if len(padded) < expected_buttons:
        padded.extend(False for _ in range(expected_buttons - len(padded)))
    return tuple(padded)


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # Using lists for compatibility
        self.latest_data["buttons"] = [0, 0, 0, 0]
        self.stop_event = multiprocessing.Event()
        self._closed = False

        # Start a process to continuously read the SpaceMouse state
        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()
        atexit.register(self.close)

    def _read_spacemouse(self):
        try:
            while not self.stop_event.is_set():
                state = pyspacemouse.read_all()
                action = [0.0] * 6
                buttons = [0, 0, 0, 0]

                if len(state) == 2:
                    action = [
                        -state[0].y, state[0].x, state[0].z,
                        -state[0].roll, -state[0].pitch, -state[0].yaw,
                        -state[1].y, state[1].x, state[1].z,
                        -state[1].roll, -state[1].pitch, -state[1].yaw
                    ]
                    buttons = state[0].buttons + state[1].buttons
                elif len(state) == 1:
                    action = [
                        -state[0].y, state[0].x, state[0].z,
                        -state[0].roll, -state[0].pitch, -state[0].yaw
                    ]
                    buttons = state[0].buttons

                try:
                    self.latest_data["action"] = action
                    self.latest_data["buttons"] = buttons
                except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
                    break
        finally:
            try:
                pyspacemouse.close()
            except Exception:
                pass

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons
    
    def close(self):
        if self._closed:
            return

        self._closed = True
        self.stop_event.set()
        if hasattr(self, "process") and self.process.is_alive():
            self.process.join(timeout=0.5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=0.5)
        if hasattr(self, "manager"):
            self.manager.shutdown()
        try:
            pyspacemouse.close()
        except Exception:
            pass
