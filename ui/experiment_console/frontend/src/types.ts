export type Mode = "work" | "train" | "eval" | "analyze";
export type TeleopMode = "position" | "rotation_z" | "rotation" | "6dof";

export type Status = { state: string; detail: string };
export type LeaderStatus = Status & {
  name: string; label: string; connected: boolean; fresh: boolean;
  active: boolean; mode: TeleopMode; mode_label?: string;
};

export type Point = { x: number; y: number };

export type GraspGuidance = {
  available: boolean;
  state: "NOT_READY" | "NEAR" | "READY" | "UNAVAILABLE";
  label: string;
  wall_time?: number;
  sim_time?: number;
  safe?: boolean;
  reason?: string;
  tolerance_status?: string;
  translation_error_local_mm?: number[];
  orientation_error_deg?: number;
  yaw_error_deg?: number;
  clearance_mm?: Record<string, number | null>;
  corrections?: Array<{ label: string; error: number; unit: "mm" | "deg"; direction: string }>;
};

export type WorkState = {
  selected_output_folder?: string | null;
  collection_stats?: { folder: string; success: number; failure: number; executions: number; other_saved: number; active: number } | null;
  worker_pid?: number | null;
  weight?: { current_weight: number; recording_id: string; request_id: string | null; frames: number };
  workspace_validation?: {
    plan_name: string; workspace_status: "DRAFT" | "FROZEN";
    workspace: { x_min: number; x_max: number; y_min: number; y_max: number };
    point: { name: string; x: number; y: number; status: "pending" | "pass" | "fail" | "NOT TESTED" | "PASS" | "MARGINAL" | "FAIL"; attempt_number: number };
    points: Array<{ name: string; x: number; y: number; status: string; attempt_number: number }>;
    active_point?: string | null; index: number; total: number;
    passed: number; failed: number; pending: number; can_freeze: boolean; recording_enabled: false;
  } | null;
  collection?: {
    distribution_id: string; title: string; label: string; current_seed: number | null;
    completed: number; target: number; remaining: number; complete: boolean;
    status: string; last_outcome: string | null; seed_start: number; seed_end: number;
    world_xy_bounds: [[number, number], [number, number]];
    yaw_center: number; yaw_half_range: number;
    plan_status?: "DRAFT" | "FROZEN";
    planned_state?: { planned_state_id: string; plan_index: number; collection_order: number;
      cell_row: number; cell_col: number; x: number; y: number } | null;
    attempt_number?: number | null;
  } | null;
  grasp_guidance?: GraspGuidance;
  phase: string;
  message: string;
  latency_capture?: { directory: string; state: 'recording' | 'saving' | 'saved' | 'error'; error: string | null; stopped?: string } | null;
  teleop_mode: TeleopMode;
  mode_change_events?: Array<{ timestamp: number; old_mode: string; new_mode: string; clutch_count: number }>;
  active_region: string | null;
  task: string | null;
  dataset_root?: string | null;
  dataset_output_root?: string | null;
  dataset_validation?: any;
  next_attempt_index?: number | null;
  episode: number | null;
  frame: number | null;
  elapsed_s: number | null;
  recording: boolean;
  status: { leader?: LeaderStatus | null; omy: Status; fr3: Status; teleop: Status; recording: Status; latency_ms: number | null };
  control_metrics?: { available: boolean; measured_hz?: number; p95_period_s?: number; max_period_s?: number; message?: string };
  mujoco?: { available?: boolean; running: boolean; pid: number | null; teleop_connected: boolean; endpoint: string; log: string };
  task_status: { reason: string | null; success: boolean | null; terminated: boolean; block_goal_error_xy: number | null; initial_robot_table_contact?: boolean | null };
  camera: { available: boolean; message: string; main_label?: string; secondary_label?: string; layout_description?: string; static_preview?: boolean; primary_view?: "operator" | "task_camera"; task_camera_frame_url?: string; wrist_frame_url?: string; bottom_right_view?: "front"; front_frame_url?: string; operator_frame_url?: string; front_operator_frame_url?: string; left_side_frame_url?: string; frame_seq?: number; preview_hz?: number; preview_measured_hz?: number | null; region_overlay?: any; collection_boundary?: { camera: string; image_size: [number, number]; world_xy_bounds: [[number, number], [number, number]]; polygon: number[][] } | null };
  force: { available: boolean; source: string | null; fx: number | null; fy: number | null; fz: number | null };
  initial_object_pose?: number[] | null;
  initial_ee_pose?: number[] | null;
  goal_pose?: number[] | null;
  camera_config_version?: string | null;
  dataset_version?: string | null;
  haptic: { available: boolean; enabled: boolean; message: string; contact?: "none" | "external" | "self"; renderable_contact?: boolean; force_target_n?: number; rendered_force_n?: number; watchdog?: string; device_connected?: boolean };
  trajectory: { available: boolean; source: string | null; points: Point[]; message: string };
  counts: Array<{ key: string; target: number | null; success: number | null; attempts: number | null; failure: number | null; abort: number | null; available: boolean }>;
};

export type ConsoleState = {
  source: "live" | "mock";
  view: Mode;
  header: { leader?: LeaderStatus | null; omy: Status; fr3: Status; teleop: Status; recording: Status; latency_ms: number | null; session: string | null; task: string | null; region: string | null };
  work: WorkState;
  train?: any;
  conversion?: any;
  eval?: any;
  analyze?: any;
};
