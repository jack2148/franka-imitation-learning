import type { ConsoleState, Point } from "./types";

const line = (points: Array<[number, number]>): Point[] => points.map(([x, y]) => ({ x, y }));

export const MOCK_STATE: ConsoleState = {
  source: "mock",
  view: "work",
  header: {
    omy: { state: "connected", detail: "mock" }, fr3: { state: "connected", detail: "mock" },
    teleop: { state: "active", detail: "mock" }, recording: { state: "idle", detail: "mock" },
    latency_ms: 8, session: "024", task: "Pick-and-Place", region: "A",
  },
  work: {
    phase: "STANDBY", message: "Mock data is explicitly enabled.", active_region: "A", task: "pick_and_place", dataset_root: null,
    teleop_mode: "position", mode_change_events: [],
    episode: 17, frame: 0, elapsed_s: 0, recording: false,
    status: { omy: { state: "connected", detail: "Connected" }, fr3: { state: "connected", detail: "Connected" }, teleop: { state: "active", detail: "Active" }, recording: { state: "idle", detail: "IDLE" }, latency_ms: 8 },
    task_status: { reason: null, success: null, terminated: false, block_goal_error_xy: null },
    camera: { available: false, message: "Mock camera preview" }, force: { available: false, source: "MuJoCo push-tool/block contact", fx: null, fy: null, fz: null }, haptic: { available: false, enabled: false, message: "Haptic unavailable" },
    trajectory: { available: true, source: "mock trajectory", points: line([[0.1, 0.2], [0.3, 0.38], [0.52, 0.31], [0.78, 0.7], [0.95, 0.82]]), message: "" },
    counts: [
      { key: "A", target: 20, success: 12, attempts: 16, failure: 3, abort: 1, available: true },
      { key: "B", target: 20, success: 7, attempts: 13, failure: 4, abort: 2, available: true },
      { key: "C", target: 20, success: 15, attempts: 18, failure: 2, abort: 1, available: true },
    ],
  },
};
