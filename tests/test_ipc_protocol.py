from __future__ import annotations

import unittest

from runtime.ipc.protocol import OMY_JOINT_NAMES, decode_message, encode_message


class TestIpcProtocol(unittest.TestCase):
    def test_round_trip_includes_trigger_and_gripper(self) -> None:
        payload = encode_message(
            sequence=7,
            source_timestamp_ns=11,
            sender_monotonic_ns=13,
            joint_names=OMY_JOINT_NAMES,
            position=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            trigger_position=-0.8,
            gripper_command=255.0,
        )
        message = decode_message(payload)
        self.assertEqual(message.sequence, 7)
        self.assertEqual(message.trigger_position, -0.8)
        self.assertEqual(message.gripper_command, 255.0)
        self.assertEqual(message.position, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5))


if __name__ == "__main__":
    unittest.main()

