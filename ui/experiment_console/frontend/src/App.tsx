import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { apiUrl, getDatasets, getState, post, selectNativeFolder, validateExperimentPath, type FolderPickerPurpose } from "./api";
import { MOCK_STATE } from "./mock";
import type { ConsoleState, Mode, Point, Status, WorkState } from "./types";
import { GraspGuidanceHUD } from "./GraspGuidanceHUD";
import { useConsoleLifecycle } from "./useConsoleLifecycle";
import { SampleWeightPanel, TeleopModeButtons } from "./SampleWeightPanel";

const MODES: Array<{ key: Mode; label: string }> = [
  { key: "work", label: "WORK" }, { key: "train", label: "TRAIN" },
  { key: "eval", label: "EVAL" }, { key: "analyze", label: "ANALYZE" },
];
const REGION_COLORS: Record<string, string> = { A: "#91b9ee", B: "#d9b06b", C: "#82c4ae" };
const off = (detail: string): Status => ({ state: "unavailable", detail });
const UI_LEADER = ((import.meta as any).env?.VITE_MIMIC_LEADER ?? "omy") as string;
const IS_HAPTIC_LEADER = UI_LEADER === "geomagic";
const teleopModeLabel = (mode: WorkState["teleop_mode"]) => mode === "position" ? "Position" : mode === "rotation_z" ? "Rotation Z" : mode === "6dof" ? "6DoF" : IS_HAPTIC_LEADER ? "Rotation 3D" : "Rotation";
const LEADER_LABEL = IS_HAPTIC_LEADER ? "Geomagic Touch" : "OMY-L100";
const UI_INPUT_BACKEND = ((import.meta as any).env?.VITE_MIMIC_INPUT_BACKEND ?? (IS_HAPTIC_LEADER ? "geomagic" : "zmq")) as string;
const UI_TASK = ((import.meta as any).env?.VITE_MIMIC_TASK ?? "pick_and_place") as string;
const IS_WEIGHT_COLLECTION = (import.meta as any).env?.VITE_MIMIC_WEIGHT_COLLECTION === "1";
const UI_EXPERIMENT = ((import.meta as any).env?.VITE_MIMIC_EXPERIMENT ?? UI_TASK) as string;
const UI_DATASET_VERSION = ((import.meta as any).env?.VITE_MIMIC_DATASET_VERSION ?? "omy_fr3_operator_wrist_20hz_v1") as string;
// Dataset/training paths are sent relative to the backend's workspace
// ``experiments/`` root.  This keeps the UI portable when the checkout moves.
const EXPERIMENT_ROOT = UI_EXPERIMENT;

function unavailableWork(): WorkState {
  return {
    phase: "UNAVAILABLE", message: "Live API is not connected. No mock fallback is enabled.",
    teleop_mode: "position", mode_change_events: [],
    active_region: null, task: null, episode: null, frame: null, elapsed_s: null, recording: false,
    status: { leader: { ...off("waiting for leader input"), name: UI_LEADER, label: LEADER_LABEL, connected: false, fresh: false, active: false, mode: "position" }, omy: off("no live OMY state source"), fr3: { state: "available", detail: "MuJoCo model available" }, teleop: off("no live teleop runtime"), recording: off("IDLE"), latency_ms: null },
    task_status: { reason: null, success: null, terminated: false, block_goal_error_xy: null },
    camera: { available: false, message: "No camera stream is configured." },
    force: { available: false, source: "MuJoCo push-tool/block contact", fx: null, fy: null, fz: null },
    haptic: { available: false, enabled: false, message: "Haptic is isolated in omy_franka_teleop." },
    trajectory: { available: false, source: null, points: [], message: "No trajectory available" },
    counts: ["A", "B", "C"].map((key) => ({ key, target: null, success: null, attempts: null, failure: null, abort: null, available: false })),
  };
}

export default function App() {
  const [mode, setMode] = useState<Mode>("work");
  const [liveState, setLiveState] = useState<ConsoleState | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [selectedRegion, setSelectedRegion] = useState("A");
  const [gripperOpen, setGripperOpen] = useState(false);
  const defaultRaw = IS_WEIGHT_COLLECTION ? "" : ((import.meta as any).env?.VITE_MIMIC_DATASET_OUTPUT_ROOT as string | undefined) || `${EXPERIMENT_ROOT}/datasets/raw${UI_TASK === "mimicgen_square" ? (IS_HAPTIC_LEADER ? "/restricted30_geomagic" : "/restricted30") : (IS_HAPTIC_LEADER ? "/geomagic" : "")}`;
  const defaultTraining = (import.meta as any).env?.VITE_MIMIC_TRAINING_ROOT || `${EXPERIMENT_ROOT}/training`;
  const defaultProcessed = (import.meta as any).env?.VITE_MIMIC_PROCESSED_ROOT || `${EXPERIMENT_ROOT}/datasets/processed`;
  const [datasetOutput, setDatasetOutput] = useState(defaultRaw);
  const [rawDatasetInput, setRawDatasetInput] = useState(defaultRaw);
  const [trainingOutput, setTrainingOutput] = useState(defaultTraining);
  const [trainingDataset, setTrainingDataset] = useState(defaultProcessed);
  const [processedOutput, setProcessedOutput] = useState(defaultProcessed);
  const [datasetOutputInfo, setDatasetOutputInfo] = useState<any>(null);
  const [folderPickerBusy, setFolderPickerBusy] = useState<FolderPickerPurpose | null>(null);
  const folderPickerBusyRef = useRef(false);
  const gripperOpenRef = useRef(false);
  const gripperBusyRef = useRef(false);
  const consoleMode = ((import.meta as any).env?.VITE_CONSOLE_MODE ?? "live") === "mock" ? "mock" : "live";
  useConsoleLifecycle(consoleMode === 'live');
  const liveApiReady = consoleMode === "mock" || (liveState !== null && apiError === null);
  useEffect(() => {
    if (consoleMode === "mock") return;
    if (!liveApiReady || !datasetOutput) { setDatasetOutputInfo(null); return; }
    let cancelled = false;
    validateExperimentPath(datasetOutput)
      .then((value) => { if (!cancelled) setDatasetOutputInfo(value); })
      .catch(() => {
        if (cancelled) return;
        if (!IS_WEIGHT_COLLECTION && datasetOutput !== defaultRaw) setDatasetOutput(defaultRaw);
        else setDatasetOutputInfo(null);
      });
    return () => { cancelled = true; };
  }, [datasetOutput, consoleMode, liveApiReady]);
  useEffect(() => {
    if (consoleMode === "mock") return;
    validateExperimentPath(rawDatasetInput).catch(() => { if (rawDatasetInput !== defaultRaw) setRawDatasetInput(defaultRaw); });
  }, [rawDatasetInput, consoleMode]);
  useEffect(() => {
    if (consoleMode === "mock") return;
    validateExperimentPath(trainingOutput).catch(() => { if (trainingOutput !== defaultTraining) setTrainingOutput(defaultTraining); });
  }, [trainingOutput, consoleMode]);
  useEffect(() => {
    if (consoleMode === "mock") return;
    validateExperimentPath(trainingDataset).catch(() => { if (trainingDataset !== defaultProcessed) setTrainingDataset(defaultProcessed); });
  }, [trainingDataset, consoleMode]);
  useEffect(() => {
    if (consoleMode === "mock" || trainingDataset !== defaultProcessed) return;
    let cancelled = false;
    getDatasets().then((items) => {
      if (cancelled) return;
      const prefix = `/experiments/${EXPERIMENT_ROOT}/datasets/processed/`;
      const matching = items.filter((item) => typeof item.path === "string" && item.path.includes(prefix));
      matching.sort((left, right) => Number(right.converted_at ?? 0) - Number(left.converted_at ?? 0));
      if (matching[0]) setTrainingDataset(matching[0].path);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [trainingDataset, consoleMode]);
  useEffect(() => {
    if (consoleMode === "mock") return;
    validateExperimentPath(processedOutput).catch(() => { if (processedOutput !== defaultProcessed) setProcessedOutput(defaultProcessed); });
  }, [processedOutput, consoleMode]);

  const shutdownConsole = async () => {
    if (consoleMode === "mock") { setToast("Mock mode: console was not shut down."); return; }
    try {
      await post("/api/shutdown", {});
      setToast("Console shutting down.");
      window.close();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "Console shutdown failed");
    }
  };

  useEffect(() => {
    if (consoleMode === "mock") return;
    let cancelled = false;
    let timer: number | undefined;
    let inFlight = false;
    const refresh = async () => {
      if (inFlight) return;
      inFlight = true;
      const started = performance.now();
      try {
        const next = await getState(mode);
        if (!cancelled) { setLiveState(next); setApiError(null); }
      } catch (error) {
        if (!cancelled) { setLiveState(null); setApiError(error instanceof Error ? error.message : "API unavailable"); }
      } finally {
        inFlight = false;
        if (!cancelled) timer = window.setTimeout(refresh, Math.max(0, 1000 / 60 - (performance.now() - started)));
      }
    };
    refresh();
    return () => { cancelled = true; if (timer !== undefined) window.clearTimeout(timer); };
  }, [mode, consoleMode, toast]);

  useEffect(() => {
    if (consoleMode === "mock") return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (target?.isContentEditable || (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))) return;
      if (event.code !== "Space") return;
      event.preventDefault();
      if (gripperBusyRef.current) return;
      const nextOpen = !gripperOpenRef.current;
      gripperBusyRef.current = true;
      post("/api/gripper", { command: nextOpen ? 255 : 0 })
        .then(() => {
          gripperOpenRef.current = nextOpen;
          setGripperOpen(nextOpen);
          setToast(`Gripper ${nextOpen ? "OPEN" : "CLOSED"} · Space`);
        })
        .catch((error) => setToast(error instanceof Error ? error.message : "Gripper command failed"))
        .finally(() => { gripperBusyRef.current = false; });
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [consoleMode]);

  useEffect(() => {
    if (consoleMode === "mock") return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() !== "q" || event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (target?.isContentEditable || (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))) return;
      event.preventDefault();
      void shutdownConsole();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [consoleMode]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 4200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const state = useMemo<ConsoleState>(() => {
    if (consoleMode === "mock") return { ...MOCK_STATE, view: mode };
    const empty = unavailableWork();
    return liveState ?? {
      source: "live", view: mode,
      header: { ...empty.status, session: null, task: UI_TASK, region: null }, work: { ...empty, task: UI_TASK },
    };
  }, [consoleMode, liveState, mode]);

  useEffect(() => {
    if ((IS_WEIGHT_COLLECTION) && liveState?.work.selected_output_folder) setDatasetOutput(liveState.work.selected_output_folder);
  }, [liveState?.work.selected_output_folder]);

  const workAction = async (action: string) => {
    if (consoleMode === "mock") { setToast("Mock mode: action not sent to a runtime."); return; }
    try {
      if (action === "mujoco_start" || action === "mujoco_stop") {
        await post(`/api/mujoco/${action === "mujoco_start" ? "start" : "stop"}`, {});
        setToast(action === "mujoco_start" ? "MuJoCo viewer started." : "MuJoCo viewer stopped.");
      } else if (action === "start") {
        await post("/api/work/start", {
          region: selectedRegion, task: UI_TASK, input_backend: UI_INPUT_BACKEND,
          omy_endpoint: "tcp://127.0.0.1:5557", episodes: state.work.collection?.target ?? 1, seed: 0, operator_id: "operator_01",
          collection_plan_id: state.work.collection?.distribution_id ?? null,
          dataset_version: UI_DATASET_VERSION,
          dataset_output_root: datasetOutput,
        });
      } else {
        await post("/api/work/action", { action });
      }
      if (action !== "mujoco_start" && action !== "mujoco_stop") setToast(action === "start" ? "Work runtime requested." : `Work action: ${action}`);
      setLiveState(await getState("work"));
    } catch (error) {
      setToast(error instanceof Error ? error.message : "Work action failed");
    }
  };

  const browseForFolder = async (
    purpose: FolderPickerPurpose,
    initialPath: string,
    onSelect: (path: string) => void,
  ) => {
    if (folderPickerBusyRef.current) return;
    if (consoleMode === "mock") { setToast("Native folder selection requires the live local backend."); return; }
    folderPickerBusyRef.current = true;
    setFolderPickerBusy(purpose);
    try {
      const result = await selectNativeFolder(purpose, initialPath);
      if (!result.cancelled) {
        if ((IS_WEIGHT_COLLECTION) && purpose === "dataset_output") await post("/api/work/output-folder", { path: result.path });
        onSelect(result.path);
        if (purpose === "dataset_output") {
          setDatasetOutputInfo({ writable: result.writable, free_bytes: result.free_bytes });
        }
        setToast(`Folder selected · ${result.writable ? "writable" : "read-only"}`);
      }
    } catch (error) {
      setToast(error instanceof Error ? error.message : "Native folder selection failed");
    } finally {
      folderPickerBusyRef.current = false;
      setFolderPickerBusy(null);
    }
  };

  return (
    <div className="app-shell">
      <Header state={state} mode={mode} onMode={setMode} source={consoleMode} onStop={() => workAction("stop_and_save")} onQuit={shutdownConsole} />
      {apiError && consoleMode === "live" && <div className="api-banner">LIVE API UNAVAILABLE · {apiError}</div>}
      <div className="mode-content">
        {mode === "work" && <WorkView state={state.work} selectedRegion={selectedRegion} setSelectedRegion={setSelectedRegion} onAction={workAction} source={consoleMode} gripperOpen={gripperOpen} datasetOutput={datasetOutput} datasetOutputInfo={datasetOutputInfo} browseBusy={folderPickerBusy !== null} onBrowse={() => browseForFolder("dataset_output", datasetOutput, setDatasetOutput)} />}
        {mode === "train" && <TrainView state={state.train} conversion={state.conversion} source={consoleMode} onToast={setToast} rawDataset={rawDatasetInput} dataset={trainingDataset} setDataset={setTrainingDataset} processedOutput={processedOutput} trainingOutput={trainingOutput} browseBusy={folderPickerBusy !== null} onBrowseRaw={() => browseForFolder("raw_input", rawDatasetInput, setRawDatasetInput)} onBrowseDataset={() => browseForFolder("training_input", trainingDataset, setTrainingDataset)} onBrowseProcessed={() => browseForFolder("processed_output", processedOutput, setProcessedOutput)} onBrowseOutput={() => browseForFolder("training_output", trainingOutput, setTrainingOutput)} />}
        {mode === "eval" && <EvalView state={state.eval} source={consoleMode} onToast={setToast} />}
        {mode === "analyze" && <AnalyzeView state={state.analyze} source={consoleMode} />}
      </div>
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

function Header({ state, mode, onMode, source, onStop, onQuit }: { state: ConsoleState; mode: Mode; onMode: (mode: Mode) => void; source: string; onStop: () => void; onQuit: () => void }) {
  const h = state.header;
  return (
    <header className="topbar">
      <div className="brand">{LEADER_LABEL} <span>Console</span></div><div className="divider" />
      <div className="header-status">
        <HeaderStatus label={LEADER_LABEL} status={IS_HAPTIC_LEADER ? (h.leader ?? off("waiting for Geomagic input")) : h.omy} /><HeaderStatus label="FR3/MuJoCo" status={h.fr3} /><HeaderStatus label="Teleop" status={h.teleop} /><HeaderStatus label="Recording" status={h.recording} />
        <span className="header-item"><em>Session</em><b>{h.session ?? "—"}</b></span>
        <span className="header-item"><em>Task</em><b>{h.task ?? "—"}</b></span>
        <span className="header-item"><em>Region</em><b>{h.region ?? "—"}</b></span>
      </div>
      <div className="pipeline">COLLECT → TRAIN → EVALUATE → ANALYZE → COLLECT</div>
      <nav className="mode-nav">{MODES.map((item) => <button className={item.key === mode ? "active" : ""} key={item.key} onClick={() => onMode(item.key)}>{item.label}</button>)}</nav>
      <div className="header-tools">
        <FrequencyBadge work={state.work} />
        {source === "live" && state.work.recording && <button className="topbar-stop" onClick={onStop}>■ STOP</button>}
        <button className="topbar-quit" onClick={onQuit}>Q QUIT</button>
      </div>
      <span className={`source-badge ${source}`}>{source.toUpperCase()}</span>
    </header>
  );
}

function FrequencyBadge({ work }: { work: WorkState }) {
  const controlHz = work.control_metrics?.available && work.control_metrics.measured_hz != null
    ? `${work.control_metrics.measured_hz.toFixed(0)} Hz`
    : "—";
  const previewTarget = work.camera.preview_hz ?? 60;
  const previewHz = work.camera.preview_measured_hz != null
    ? work.camera.preview_measured_hz.toFixed(0)
    : previewTarget;
  return <span className="frequency-badge" title={`Live control-loop and display-preview frequency (preview target ${previewTarget} Hz)`}><i /> CTRL {controlHz} · PREVIEW {previewHz} Hz</span>;
}

function HeaderStatus({ label, status }: { label: string; status: Status }) {
  const good = ["connected", "active", "recording"].includes(status.state);
  return <span className="header-item"><i className={`dot ${good ? "good" : status.state === "available" ? "neutral" : "off"}`} /><em>{label}</em><b>{status.state.toUpperCase()}</b></span>;
}

function WorkView({ state, selectedRegion, setSelectedRegion, onAction, source, gripperOpen, datasetOutput, datasetOutputInfo, browseBusy, onBrowse }: { state: WorkState; selectedRegion: string; setSelectedRegion: (region: string) => void; onAction: (action: string) => void; source: string; gripperOpen: boolean; datasetOutput: string; datasetOutputInfo: any; browseBusy: boolean; onBrowse: () => void }) {
  const frontPanel = IS_HAPTIC_LEADER && state.camera.bottom_right_view === "front";
  const validation = state.workspace_validation;
  const validationPoint = validation?.points.find(p => p.name === validation.active_point) ?? validation?.point;
  return (
    <div className="work-layout">
      <aside className="left-rail">
        {validation && validationPoint && <Card title="Workspace validation"><strong>{validation.plan_name}</strong><InfoRow label="Point" value={validationPoint.name} /><InfoRow label="XY [m]" value={`${validationPoint.x.toFixed(4)}, ${validationPoint.y.toFixed(4)}`} /><InfoRow label="Point index" value={`${validation.points.findIndex(p => p.name === validationPoint.name)+1} / 9`} /><InfoRow label="Assessment" value={validationPoint.status.toUpperCase()} /><InfoRow label="PASS" value={`${validation.passed} / 9`} /><InfoRow label="Attempt" value={String(validationPoint.attempt_number)} />{validation.active_point && validation.active_point !== validation.point.name && <InfoRow label="Next selected" value={validation.point.name} />}<small>Assess A–E manually. Recording disabled. STOP before selecting or retrying a point.</small></Card>}
        <Card title="System Status">
          <div className="latency"><span>Latency</span><b>{state.status.latency_ms == null ? "—" : `${state.status.latency_ms} ms`}</b></div>
          <StatusRow label={IS_HAPTIC_LEADER ? LEADER_LABEL : "Arm Controller"} status={IS_HAPTIC_LEADER ? (state.status.leader ?? off("waiting for Geomagic input")) : state.status.omy} /><StatusRow label="Sim Backend" status={state.status.fr3} /><StatusRow label="Teleop Input" status={state.status.teleop} />
          {IS_HAPTIC_LEADER ? <><StatusRow label="Haptics" status={{ state: state.haptic.enabled ? "available" : "off", detail: state.haptic.message }} /><StatusRow label="Safety Watchdog" status={{ state: state.haptic.watchdog === "OK" ? "connected" : "off", detail: state.haptic.watchdog ?? "OFF" }} /></> : <><StatusRow label="F/T Sensor" status={state.haptic.available ? { state: "available", detail: "available" } : off("haptic isolated")} /><StatusRow label="Safety Watchdog" status={off("not exposed")} /></>}
        </Card>
        {(IS_WEIGHT_COLLECTION ? <Card title="선택한 폴더의 수집 현황"><InfoRow label="성공" value={String(state.collection_stats?.success ?? 0)} /><InfoRow label="실패" value={String(state.collection_stats?.failure ?? 0)} /><InfoRow label="실행 횟수" value={String(state.collection_stats?.executions ?? 0)} /><small>{state.collection_stats ? "개수 제한 없음 · 종료는 직접 선택" : "저장 폴더를 선택하세요"}</small></Card> : state.collection ? <Card title={state.collection.title} className="progress-card"><div className="collection-plan-progress"><strong>{state.collection.label}</strong><span>{state.collection.complete ? "COLLECTION COMPLETE" : state.collection.planned_state ? `State ${state.collection.planned_state.plan_index} · cell (${state.collection.planned_state.cell_row},${state.collection.planned_state.cell_col}) · attempt ${state.collection.attempt_number}` : `Seed ${state.collection.current_seed}`}</span>{state.collection.planned_state && <small>XY ({state.collection.planned_state.x.toFixed(5)}, {state.collection.planned_state.y.toFixed(5)}) m</small>}<b>{state.collection.completed} / {state.collection.target}</b>{state.collection.planned_state && <small>Workspace {((state.collection.world_xy_bounds[0][1]-state.collection.world_xy_bounds[0][0])*1000).toFixed(0)} × {((state.collection.world_xy_bounds[1][1]-state.collection.world_xy_bounds[1][0])*1000).toFixed(0)} mm</small>}<small>Remaining {state.collection.remaining}</small><em>{state.collection.planned_state ? state.collection.status : state.collection.status === "SUCCESS" ? `SUCCESS · Next seed: ${state.collection.current_seed}` : state.collection.status}</em></div></Card> : <Card title="Collection Progress" className="progress-card"><div className="progress-list">{state.counts.map((item) => <ProgressRow key={item.key} item={item} />)}</div></Card>)}
      </aside>
      <section className="camera-panel">
        <div className="camera-toolbar"><span className="tag">TRIPLE CAMERA PREVIEW — {source === "mock" ? "MOCK" : state.camera.static_preview ? "FIXED PREVIEW" : state.camera.available ? "LIVE" : "UNAVAILABLE"}</span><span className="mono">{state.camera.layout_description ?? (frontPanel ? "Operator View + Top-Down Alignment · EE Point + Front View" : state.camera.available ? "operator + front + left side" : state.camera.message)}</span></div>
        <GraspGuidanceHUD guidance={state.grasp_guidance} />
        <div className="dual-camera-grid">
          <CameraViewport title={state.camera.main_label ?? "Operator View"} camera="operator" state={state} source={source} />
          <div className="camera-side-stack">
            <CameraViewport title="Top-Down Alignment · EE Point" camera="front_operator" state={state} source={source} />
            <CameraViewport title={state.camera.secondary_label ?? (frontPanel ? "Front View" : "Left Side View")} camera={frontPanel ? "front" : "left_side"} state={state} source={source} />
          </div>
        </div>
      </section>
      <aside className="right-rail">
        <SampleWeightPanel state={state} source={source} />
        <Card title="Dataset Output"><div className="path-picker"><input value={datasetOutput} readOnly placeholder="BROWSE로 저장 폴더 선택" /><button disabled={browseBusy || state.recording || state.phase === "STARTING" || state.phase === "SAVING"} onClick={onBrowse}>{browseBusy ? "SELECTING…" : "BROWSE"}</button></div><InfoRow label="Selected root" value={datasetOutputInfo ? "VALID" : "INVALID"} /><InfoRow label="Writable" value={datasetOutputInfo?.writable ? "YES" : "NO"} /><InfoRow label="Free disk" value={datasetOutputInfo ? `${Math.round(datasetOutputInfo.free_bytes / 1073741824)} GB` : "—"} /></Card>
        <Card title="Collection Preflight"><InfoRow label="Task" value={state.task ?? UI_TASK} /><InfoRow label="Demo batch" value={state.collection ? `${state.collection.target} · ${state.collection.label}` : "manual"} /><InfoRow label="Control" value="1000 Hz" /><InfoRow label="Dataset" value="20 Hz / 50 steps" /><InfoRow label="Action" value="8D" /><InfoRow label="Cameras" value="operator + wrist" /><InfoRow label="Teleop mode" value={teleopModeLabel(state.teleop_mode)} /><InfoRow label="Initial table contact" value={state.task_status.initial_robot_table_contact === false ? "PASS" : state.task_status.initial_robot_table_contact === true ? "FAIL" : "CHECKED AT START"} /></Card>
        {state.dataset_validation && <Card title="Dataset Saved"><InfoRow label="Attempt" value={formatNumber(state.dataset_validation.attempt_index)} /><InfoRow label="Path" value={state.dataset_root ?? "—"} /><InfoRow label="Samples" value={String(state.dataset_validation.samples)} /><InfoRow label="Duration" value={`${Number(state.dataset_validation.duration_s ?? 0).toFixed(2)} s`} /><InfoRow label="20 Hz" value={state.dataset_validation.checks?.timestamps ? "PASS" : "FAIL"} /><InfoRow label="Action 8D" value={state.dataset_validation.checks?.action_shape ? "PASS" : "FAIL"} /><InfoRow label="Operator" value={`${state.dataset_validation.images?.operator}/${state.dataset_validation.samples}`} /><InfoRow label="Wrist" value={`${state.dataset_validation.images?.wrist}/${state.dataset_validation.samples}`} /><InfoRow label="NaN/Inf" value={state.dataset_validation.checks?.finite ? "PASS" : "FAIL"} /><InfoRow label="Outcome" value={state.dataset_validation.outcome ?? "—"} /><InfoRow label="Training-ready" value={state.dataset_validation.training_ready ? "YES" : state.dataset_validation.readiness} /></Card>}
        <Card title="Teleop Mode"><div className={`teleop-mode-display ${state.teleop_mode === "position" ? "position" : "rotation"}`}><strong>{teleopModeLabel(state.teleop_mode)}</strong></div><TeleopModeButtons state={state} source={source} geomagic={IS_HAPTIC_LEADER} /></Card>
        <Card title={validation ? "Validation Control" : "Episode Control"}><button className={`primary-action ${state.recording ? "danger" : ""}`} disabled={source === "live" && (state.phase === "UNAVAILABLE" || state.phase === "STARTING" || state.phase === "SAVING" || state.latency_capture?.state === "saving" || !datasetOutputInfo?.writable || state.collection?.complete || state.collection?.plan_status === "DRAFT")} onClick={() => onAction(validation ? (state.phase === "RECORDING" ? "discard" : "start") : state.recording ? "stop_and_save" : "start")}>{validation ? (state.phase === "RECORDING" ? "■ STOP VALIDATION" : "▶ START VALIDATION") : state.recording ? "■ STOP & SAVE" : (state.collection?.complete) ? "COLLECTION COMPLETE" : "● START RECORDING"}</button>{!IS_HAPTIC_LEADER && <button className={`mujoco-button ${state.mujoco?.running ? "running" : ""}`} onClick={() => onAction(state.mujoco?.running ? "mujoco_stop" : "mujoco_start")}>{state.mujoco?.running ? "■ STOP MUJOCO" : "▶ START MUJOCO"}</button>}{state.recording && <button className="secondary-action initial-state-button" onClick={() => onAction("reset_initial")}>↺ INITIAL STATE</button>}<button className="link-button" onClick={() => onAction("reset")}>{false ? "Abort & preserve attempt" : "Discard episode"}</button><div className="mono" style={{ marginTop: "0.75rem" }}>GRIPPER · {gripperOpen ? "OPEN" : "CLOSED"} · SPACE TOGGLE</div></Card>
        <Card title="Active Region"><div className="region-selector">{["A", "B", "C"].map((key) => <button key={key} className={selectedRegion === key ? "selected" : ""} style={{ "--region": REGION_COLORS[key] } as CSSProperties} disabled={state.recording} onClick={() => setSelectedRegion(key)}>{key}<small>{countText(state.counts.find((item) => item.key === key)?.success)}/{countText(state.counts.find((item) => item.key === key)?.target)}</small></button>)}</div></Card>
        {state.latency_capture && <Card title="Latency Capture"><InfoRow label="Logs" value={state.latency_capture.state.toUpperCase()} /><div className="mono" style={{ overflowWrap: "anywhere" }}>{state.latency_capture.directory}</div>{state.latency_capture.error && <p>{state.latency_capture.error}</p>}{state.latency_capture.stopped && state.latency_capture.stopped !== "session_end" && <p>Capture limit reached: {state.latency_capture.stopped}</p>}</Card>}
        <Card title="Episode Info" className="episode-card"><InfoRow label="Attempt" value={formatNumber(state.dataset_validation?.attempt_index ?? state.next_attempt_index)} /><InfoRow label="Episode" value={formatNumber(state.episode)} /><InfoRow label="Frame" value={formatNumber(state.frame)} /><InfoRow label="Elapsed" value={formatTime(state.elapsed_s)} /><InfoRow label="Region" value={state.active_region ?? selectedRegion} /><InfoRow label="Camera" value={state.camera_config_version ?? "UNAVAILABLE"} /><InfoRow label="Phase" value={state.phase} /><InfoRow label="Recording" value={state.recording ? "RECORDING" : "IDLE"} /><div className="pose-grid"><span>INITIAL OBJECT<br /><b>{formatPose(state.initial_object_pose)}</b></span><span>GOAL POSE<br /><b>{formatPose(state.goal_pose)}</b></span></div></Card>
        {state.recording && <Card title="Outcome"><div className="outcome-buttons"><button onClick={() => onAction("success")}>SUCCESS</button><button onClick={() => onAction("failure")}>FAILURE</button><button onClick={() => onAction("abort")}>ABORT</button></div></Card>}
        <div className="work-control-bottom"><ForceTerminal force={state.force} /><FastLoopPanel trajectory={state.trajectory} /></div>
        <div className="haptic-note">{IS_HAPTIC_LEADER ? <>Haptics: {state.haptic.message}<br /><small>contact: {state.haptic.contact ?? "none"} · force: {(state.haptic.rendered_force_n ?? state.haptic.force_target_n ?? 0).toFixed(2)} N · watchdog: {state.haptic.watchdog ?? "OFF"}</small></> : <>HAPTIC · UNAVAILABLE<br /><small>Independent development track</small></>}</div>
      </aside>
    </div>
  );
}

function CameraViewport({ title, camera, state, source, regions = false }: { title: string; camera: "task_camera" | "wrist" | "operator" | "front_operator" | "left_side" | "front"; state: WorkState; source: string; regions?: boolean }) {
  const frameUrl = camera === "task_camera" ? state.camera.task_camera_frame_url : camera === "wrist" ? state.camera.wrist_frame_url : camera === "operator" ? state.camera.operator_frame_url : camera === "front_operator" ? state.camera.front_operator_frame_url : camera === "front" ? state.camera.front_frame_url : state.camera.left_side_frame_url;
  const overlay = regions ? state.camera.region_overlay : null;
  const boundary = state.camera.collection_boundary?.camera === camera ? state.camera.collection_boundary : null;
  const aspect = boundary ? boundary.image_size[0] / boundary.image_size[1] : 1;
  return <div className="work-camera-view"><div className="work-camera-title"><b>{title}</b><span className="mono">CONTROL MONITOR</span></div><div className="work-camera-image">{source === "mock" ? <div className="scene-message">PREVIEW (DEMO)</div> : frameUrl ? <LiveCameraImage src={apiUrl(frameUrl)} alt={title} /> : <div className="scene-message">WAITING FOR CAMERA FRAME</div>}{source === "live" && boundary?.polygon?.length === 4 && <svg className="camera-projection-overlay" viewBox={`0 0 ${aspect} 1`} preserveAspectRatio="xMidYMid meet" aria-label="Blue collection boundary"><polygon points={boundary.polygon.map(([u, v]) => `${u * aspect},${v}`).join(" ")} fill="none" stroke="#208dff" strokeWidth="2" vectorEffect="non-scaling-stroke" /></svg>}{regions && overlay?.regions?.length ? <svg className="camera-projection-overlay" viewBox="0 0 1 1" preserveAspectRatio="xMidYMid meet" aria-label="Projected Region boundaries">{overlay.regions.map((region: any) => <polygon key={region.id} points={region.polygon.map((point: number[]) => point.join(",")).join(" ")} fill={REGION_COLORS[region.id]} fillOpacity={region.active ? 0.16 : 0.035} stroke={REGION_COLORS[region.id]} strokeOpacity={region.active ? 0.95 : 0.55} strokeWidth="0.0025" vectorEffect="non-scaling-stroke" />)}{overlay.regions.map((region: any) => <text key={`${region.id}-label`} x={region.polygon[0][0] + 0.01} y={region.polygon[0][1] + 0.025} fill={REGION_COLORS[region.id]} fontSize="0.035" fontWeight="700">{region.label}</text>)}{overlay.object?.visible && <circle cx={overlay.object.pixel[0]} cy={overlay.object.pixel[1]} r="0.012" fill="#24b7ff" stroke="#fff" strokeWidth="0.003" vectorEffect="non-scaling-stroke" />}{overlay.goal?.visible && <circle cx={overlay.goal.pixel[0]} cy={overlay.goal.pixel[1]} r="0.014" fill="none" stroke="#52e38a" strokeWidth="0.004" vectorEffect="non-scaling-stroke" />}</svg> : regions && <div className="scene-message">REGION OVERLAY UNAVAILABLE</div>}</div></div>;
}

function LiveCameraImage(props: { src: string; alt: string; className?: string }) {
  return <LegacyLiveCameraImage {...props} />;
}

function LegacyLiveCameraImage({ src, alt, className }: { src: string; alt: string; className?: string }) {
  const imageRef = useRef<HTMLImageElement>(null);
  useEffect(() => {
    let frame = 0;
    let cancelled = false;
    let loading = false;
    let lastRequest = -Infinity;
    let currentUrl: string | undefined;
    const controller = new AbortController();
    const load = async () => {
      let nextUrl: string | undefined;
      try {
        const response = await fetch(src, { cache: "no-store", signal: controller.signal });
        if (!response.ok) return;
        nextUrl = URL.createObjectURL(await response.blob());
        const nextImage = new Image();
        nextImage.src = nextUrl;
        await nextImage.decode();
        if (cancelled || !imageRef.current) return;
        imageRef.current.src = nextUrl;
        if (currentUrl) URL.revokeObjectURL(currentUrl);
        currentUrl = nextUrl;
        nextUrl = undefined;
      } catch {
        // Keep the last complete frame while the preview is unavailable.
      } finally {
        if (nextUrl) URL.revokeObjectURL(nextUrl);
        loading = false;
      }
    };
    const tick = (now: number) => {
      if (!loading && now - lastRequest >= 1000 / 60 - 0.5) {
        loading = true;
        lastRequest = now;
        void load();
      }
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      controller.abort();
      window.cancelAnimationFrame(frame);
      if (currentUrl) URL.revokeObjectURL(currentUrl);
    };
  }, [src]);
  return <img ref={imageRef} className={className} alt={alt} />;
}

function FastLoopPanel({ trajectory }: { trajectory: WorkState["trajectory"] }) {
  const points = normalizePoints(trajectory.points);
  return <div className="fast-loop-panel"><div className="work-camera-title"><b>Fast Loop Path</b><span className="mono">{trajectory.available ? "LIVE" : "IDLE"}</span></div><div className="fast-loop-plot"><svg viewBox="0 0 160 96" preserveAspectRatio="none">{points.length > 1 && <polyline points={points.map((point) => `${point.x},${point.y}`).join(" ")} fill="none" stroke="#91b9ee" strokeWidth="2.4" />}</svg>{points.length === 0 && <span>Waiting for EE path</span>}</div><small className="mono">{trajectory.available ? trajectory.source : "Fast loop path not available"}</small></div>;
}

function ForceTerminal({ force }: { force: WorkState["force"] }) {
  const value = (item: number | null | undefined) => item == null ? "UNAVAILABLE" : `${item.toFixed(3)} N`;
  return <div className="force-terminal"><div className="work-camera-title"><b>MuJoCo Contact Force</b><span className="mono">{force.available ? "LIVE" : "IDLE"}</span></div><pre>{`Fx = ${value(force.fx)}\nFy = ${value(force.fy)}\nFz = ${value(force.fz)}`}</pre><small className="mono">{force.source ?? "Force source unavailable"}</small></div>;
}

function Scene({ state, source }: { state: WorkState; source: string }) {
  return <div className="scene"><div className="scene-message">{source === "mock" ? "PREVIEW (DEMO)" : state.camera.available ? "LIVE CAMERA" : "NO CAMERA SOURCE"}</div>{["A", "B", "C"].map((key) => <div key={key} className={`scene-region ${state.active_region === key ? "active" : ""}`} style={{ "--region": REGION_COLORS[key] } as CSSProperties}><b>{key}</b></div>)}{state.trajectory.available && <div className="scene-point" />}</div>;
}

function ProgressRow({ item }: { item: WorkState["counts"][number] }) {
  const pct = item.available && item.target ? Math.min(100, ((item.success ?? 0) / item.target) * 100) : 0;
  return <div className="progress-row"><div><strong><i style={{ background: REGION_COLORS[item.key] }} />Region {item.key}</strong><b>{countText(item.success)} <small>/ {countText(item.target)}</small></b></div><div className="bar"><i style={{ width: `${pct}%`, background: REGION_COLORS[item.key] }} /></div><span className="mono">Attempts {countText(item.attempts)} · Fail {countText(item.failure)} · Abort {countText(item.abort)}</span></div>;
}

function TrajectoryInset({ trajectory, title }: { trajectory: WorkState["trajectory"]; title: string }) {
  const points = normalizePoints(trajectory.points);
  return <div className="trajectory-inset"><div className="inset-title"><b>{title}</b><span><button className="active">Latest</button><button>3×</button><button>Region</button></span></div><div className="plot"><svg viewBox="0 0 160 96" preserveAspectRatio="none">{points.length > 1 && <polyline points={points.map((point) => `${point.x},${point.y}`).join(" ")} fill="none" stroke="#91b9ee" strokeWidth="2.4" />}</svg>{points.length === 0 && <span>{trajectory.message || "No trajectory available"}</span>}</div><small className="mono">{trajectory.available ? trajectory.source : "No trajectory available"}</small></div>;
}

function parseTrainingLog(lines: string[], fallbackSteps: string) {
  const text = lines.join("\n");
  const progressMatches = text.match(/Training:\s+(\d+)%[^\n]*?\|\s*(\d+)\/(\d+)\s+\[([^<]+)<([^,]+),/g);
  const progress = progressMatches ? progressMatches[progressMatches.length - 1] : undefined;
  const match = progress?.match(/(\d+)%[^\n]*?\|\s*(\d+)\/(\d+)\s+\[([^<]+)<([^,]+),/);
  const lossMatches = [...text.matchAll(/loss:([0-9.eE+-]+)/g)];
  const loss = lossMatches.length ? lossMatches[lossMatches.length - 1][1] : undefined;
  const lrMatches = [...text.matchAll(/\blr:([0-9.eE+-]+)/g)];
  const learningRate = lrMatches.length ? lrMatches[lrMatches.length - 1][1] : undefined;
  return {
    step: match ? `${match[2]} / ${match[3]}` : `— / ${fallbackSteps}`,
    progress: match ? `${match[1]}%` : "—",
    elapsed: match?.[4]?.trim() ?? "—",
    eta: match?.[5]?.trim() ?? "—",
    loss: loss ?? "NOT AVAILABLE",
    learningRate: learningRate ?? "NOT AVAILABLE",
  };
}

function TrainView({ state, conversion, source, onToast, rawDataset, dataset, setDataset, processedOutput, trainingOutput, browseBusy, onBrowseRaw, onBrowseDataset, onBrowseProcessed, onBrowseOutput }: { state: any; conversion: any; source: string; onToast: (message: string) => void; rawDataset: string; dataset: string; setDataset: (path: string) => void; processedOutput: string; trainingOutput: string; browseBusy: boolean; onBrowseRaw: () => void; onBrowseDataset: () => void; onBrowseProcessed: () => void; onBrowseOutput: () => void }) {
  const [policy, setPolicy] = useState("act");
  const [batchSize, setBatchSize] = useState("8");
  const [learningRate, setLearningRate] = useState("0.00001");
  const [saveFreq, setSaveFreq] = useState("20000");
  const [chunkSize, setChunkSize] = useState("64");
  const [horizon, setHorizon] = useState("64");
  const [obsSteps, setObsSteps] = useState("2");
  const [actionSteps, setActionSteps] = useState("32");
  const [rawDatasetValidation, setRawDatasetValidation] = useState<any>(null);
  const [datasetValidation, setDatasetValidation] = useState<any>(null);
  const [steps, setSteps] = useState("5000");
  const [seed, setSeed] = useState("0");
  const [device, setDevice] = useState("cuda");
  const status = state?.status?.state ?? "IDLE";
  const training = parseTrainingLog(state?.log_lines ?? [], steps);
  const rawFolder = rawDataset.split(/[\\/]/).filter(Boolean).pop() ?? "raw";
  const datasetFolder = dataset.split(/[\\/]/).filter(Boolean).pop() ?? "dataset";
  const runId = `${datasetFolder}_${policy === "act" ? "ACT" : "Diffusion"}`;
  const convertedOutputPath = `${processedOutput.replace(/\/$/, "")}/${rawFolder}_convert`;
  const trainingRunPath = `${trainingOutput.replace(/\/$/, "")}/${policy === "act" ? "act" : "diffusion_policy"}/${runId}`;

  useEffect(() => {
    if (source !== "mock") validateExperimentPath(rawDataset).then((value) => setRawDatasetValidation(value.dataset)).catch(() => setRawDatasetValidation(null));
  }, [rawDataset, source]);
  useEffect(() => {
    if (source === "mock") return;
    let cancelled = false;
    setDatasetValidation(null);
    validateExperimentPath(dataset)
      .then((value) => { if (!cancelled) setDatasetValidation(value.dataset); })
      .catch(() => { if (!cancelled) setDatasetValidation(null); });
    return () => { cancelled = true; };
  }, [dataset, source]);
  useEffect(() => {
    if (conversion?.validation?.training_ready && conversion.output) setDataset(conversion.output);
  }, [conversion?.validation?.training_ready, conversion?.output]);

  const convert = async () => {
    if (source === "mock") return onToast("Mock mode: conversion was not started.");
    try {
      await post("/api/conversion/start", { input_root: rawDataset, output_root: processedOutput, task: UI_TASK });
      onToast("Raw-to-LeRobot conversion started.");
    } catch (error) { onToast(error instanceof Error ? error.message : "Conversion failed"); }
  };

  const start = async () => {
    if (source === "mock") return onToast("Mock mode: training was not started.");
    try {
      const overrides = {
        batch_size: Number(batchSize),
        "optimizer.lr": Number(learningRate),
        save_freq: Number(saveFreq),
        "policy.n_action_steps": Number(actionSteps),
        ...(policy === "act"
          ? { "policy.chunk_size": Number(chunkSize) }
          : { "policy.horizon": Number(horizon), "policy.n_obs_steps": Number(obsSteps) }),
      };
      // Trainer commands run from the checkout root, while folder APIs resolve
      // user paths relative to ``experiments/``.
      const config = `configs/square/${policy === "act" ? "act300k" : "dp480k"}/wrapper.json`;
      await post("/api/train/start", {
        policy, dataset_root: dataset, training_output_root: trainingOutput, steps: Number(steps),
        seed: Number(seed), device, config, overrides,
      });
      onToast("Training start requested.");
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Training start failed");
    }
  };

  const stop = async () => {
    if (source === "mock") return onToast("Mock mode: training was not stopped.");
    try {
      await post("/api/train/stop", {});
      onToast("Training stop requested.");
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Training stop failed");
    }
  };

  return (
    <div className="train-layout">
      <aside className="setup-rail">
        <Card title="Training Setup">
          <Field label="Raw Dataset Input">
            <div className="path-picker"><input value={rawDataset} readOnly /><button disabled={browseBusy} onClick={onBrowseRaw}>{browseBusy ? "SELECTING…" : "BROWSE"}</button></div>
          </Field>
          <Field label="Raw Dataset Type"><input value={rawDatasetValidation?.type ?? "UNKNOWN"} readOnly /></Field>
          <Field label="Processed Dataset Output"><div className="path-picker"><input value={processedOutput} readOnly /><button disabled={browseBusy} onClick={onBrowseProcessed}>{browseBusy ? "SELECTING…" : "BROWSE"}</button></div></Field>
          <InfoRow label="Converted folder" value={convertedOutputPath} />
          <button className="primary-action" disabled={!rawDatasetValidation?.valid || !["RAW", "RAW_MIMIC"].includes(rawDatasetValidation?.type) || conversion?.status?.state === "RUNNING"} onClick={convert}>CONVERT TO LEROBOT</button>
          {conversion?.validation && <>
            <Field label="Conversion"><input value={conversion.validation.training_ready ? "PASS" : "FAIL"} readOnly /></Field>
            <InfoRow label="Converted dataset" value={conversion.output ?? "—"} />
            <InfoRow label="Episodes / frames" value={`${conversion.validation.episodes ?? "—"} / ${conversion.validation.frames ?? "—"}`} />
            <InfoRow label="Task / action" value={`${conversion.validation.task ?? "—"} / ${conversion.validation.action_shape?.[0] ?? "—"}D`} />
            <InfoRow label="Camera schema" value={conversion.validation.camera_schema ?? "UNKNOWN"} />
            <InfoRow label="Operator / wrist" value={`${conversion.validation.operator_equivalent ? "PASS" : "N/A"} / ${conversion.validation.wrist_equivalent ? "PASS" : "FAIL"}`} />
            <InfoRow label="LeRobot load" value={conversion.validation.lerobot_load ? "PASS" : "FAIL"} />
            <InfoRow label="Training-ready" value={conversion.validation.training_ready ? "YES" : "NO"} />
          </>}
          <Field label="Dataset Input">
            <div className="path-picker"><input value={dataset} readOnly /><button disabled={browseBusy} onClick={onBrowseDataset}>{browseBusy ? "SELECTING…" : "BROWSE"}</button></div>
          </Field>
          <Field label="Dataset Type"><input value={datasetValidation?.type ?? "UNKNOWN"} readOnly /></Field>
          <Field label="Episodes"><input value={datasetValidation?.episodes ?? "—"} readOnly /></Field>
          <Field label="Samples"><input value={datasetValidation?.samples ?? "—"} readOnly /></Field>
          <Field label="Images"><input value={datasetValidation ? `operator ${datasetValidation.images?.operator ?? 0} · wrist ${datasetValidation.images?.wrist ?? 0}` : "—"} readOnly /></Field>
          <Field label="Task"><input value={datasetValidation?.task ?? "—"} readOnly /></Field>
          <Field label="Validation"><input value={datasetValidation?.valid ? "PASS" : "FAIL"} readOnly /></Field>
          <Field label="Readiness"><input value={datasetValidation?.training_ready ? "READY" : datasetValidation?.readiness ?? "NOT VALIDATED"} readOnly /></Field>
          <Field label="Training Output"><div className="path-picker"><input value={trainingOutput} readOnly /><button disabled={browseBusy} onClick={onBrowseOutput}>{browseBusy ? "SELECTING…" : "BROWSE"}</button></div></Field>
          <InfoRow label="Run folder" value={trainingRunPath} />
          <Field label="Policy">
            <div className="toggle">
              <button className={policy === "act" ? "selected" : ""} onClick={() => setPolicy("act")}>ACT</button>
              <button className={policy === "diffusion_policy" ? "selected" : ""} onClick={() => setPolicy("diffusion_policy")}>Diffusion<br />Policy</button>
            </div>
          </Field>
          <Field label="Config"><input value={`configs/square/${policy === "act" ? "act300k" : "dp480k"}/wrapper.json`} readOnly /></Field>
          <Field label="Batch Size"><input type="number" min="1" value={batchSize} onChange={(event) => setBatchSize(event.target.value)} /></Field>
          <Field label="Learning Rate"><input type="number" step="0.000001" value={learningRate} onChange={(event) => setLearningRate(event.target.value)} /></Field>
          <Field label="Save Frequency"><input type="number" min="1" value={saveFreq} onChange={(event) => setSaveFreq(event.target.value)} /></Field>
          {policy === "act" ? (
            <Field label="ACT Chunk Size"><input type="number" min="1" value={chunkSize} onChange={(event) => setChunkSize(event.target.value)} /></Field>
          ) : (
            <>
              <Field label="DP Horizon"><input type="number" min="1" value={horizon} onChange={(event) => setHorizon(event.target.value)} /></Field>
              <Field label="DP Observation Steps"><input type="number" min="1" value={obsSteps} onChange={(event) => setObsSteps(event.target.value)} /></Field>
            </>
          )}
          <Field label="Action Steps"><input type="number" min="1" value={actionSteps} onChange={(event) => setActionSteps(event.target.value)} /></Field>
          <Field label="Seed"><input value={seed} onChange={(event) => setSeed(event.target.value)} /></Field>
          <Field label="Device">
            <select value={device} onChange={(event) => setDevice(event.target.value)}>
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </select>
          </Field>
          <Field label="Training Steps"><input value={steps} onChange={(event) => setSteps(event.target.value)} /></Field>
          <Field label="Run ID"><input value={runId} readOnly /></Field>
        </Card>
      </aside>
      <section className="train-main">
        <Card title="Training" action={<Pill text={status} tone={status === "RUNNING" ? "good" : "neutral"} />}>
          <div className="stat-grid">
            <Stat label="Step" value={source === "mock" ? "100,000 / 100,000" : training.step} />
            <Stat label="Progress" value={source === "mock" ? "100%" : training.progress} />
            <Stat label="Elapsed" value={source === "mock" ? "—" : training.elapsed} />
            <Stat label="ETA" value={source === "mock" ? "—" : training.eta} />
          </div>
          <div className="bar large"><i style={{ width: source === "mock" ? "100%" : training.progress }} /></div>
        </Card>
        <Card title="Metrics" className="metrics-card">
          <div className="metric-grid">
            <Metric title="Train Loss" value={source === "mock" ? "0.222" : training.loss} color="#a996eb" />
            <Metric title="Validation Loss" value={source === "mock" ? "0.256" : "NOT AVAILABLE"} color="#91b9ee" />
            <Metric title="Learning Rate" value={source === "mock" ? "1.0e-4" : training.learningRate} color="#d9b06b" />
          </div>
          <div className="resource-row">
            <span>GPU Utilization <b>{state?.gpu?.available ? `${state.gpu.utilization}%` : "UNAVAILABLE"}</b></span>
            <span>GPU Memory <b>{state?.gpu?.available ? `${state.gpu.memory_used} / ${state.gpu.memory_total} MiB` : "UNAVAILABLE"}</b></span>
          </div>
        </Card>
        <div className="bottom-grid">
          <Card title="Validation Snapshot">
            <InfoRow label="Best Val Loss" value={source === "mock" ? "0.141" : "NOT AVAILABLE"} />
            <InfoRow label="Best Checkpoint" value={source === "mock" ? "60k" : "NOT AVAILABLE"} />
            <InfoRow label="Last Val Step" value={source === "mock" ? "65,000" : "NOT AVAILABLE"} />
          </Card>
          <Card title="Dataset Summary">
            <InfoRow label="Total Demos" value={state?.dataset_summary?.available ? `${state.dataset_summary.episodes} episodes` : "NOT AVAILABLE"} />
            <InfoRow label="Frames" value={state?.dataset_summary?.available ? String(state.dataset_summary.frames) : "NOT AVAILABLE"} />
            <InfoRow label="Regions" value="NOT COMPUTED" />
          </Card>
          <Card title="Event Log">
            <pre className="event-log">{source === "mock" ? "Mock event stream" : (state?.log_lines ?? []).join("\n") || "No training job has run."}</pre>
          </Card>
        </div>
      </section>
      <aside className="right-rail">
          <Card title="Run Control">
            <button className={"primary-action " + (status === "RUNNING" ? "danger" : "")} disabled={source === "live" && status !== "RUNNING" && !datasetValidation?.training_ready} onClick={status === "RUNNING" ? stop : start}>
              {status === "RUNNING" ? "STOP" : "START TRAINING"}
            </button>
          </Card>
          <Card title="Run">
            <InfoRow label="Run" value={runId} />
            <InfoRow label="Dataset" value={dataset} />
            <InfoRow label="Status" value={status} />
          </Card>
          <Card title="Checkpoints">
            <p className="muted">Checkpoint overwrite protection is enabled.</p>
            <InfoRow label="latest" value="UNAVAILABLE" />
            <InfoRow label="best" value="UNAVAILABLE" />
          </Card>
      </aside>
    </div>
  );
}
function EvalView({ state, source, onToast }: { state: any; source: string; onToast: (message: string) => void }) {
  const [experiment, setExperiment] = useState(UI_EXPERIMENT);
  const [policy, setPolicy] = useState("act");
  const [trainedRun, setTrainedRun] = useState("");
  const [checkpoint, setCheckpoint] = useState("");
  const [runId, setRunId] = useState("EVAL-console");
  const [region, setRegion] = useState("overall");
  const [seed, setSeed] = useState("0");
  const [episodes, setEpisodes] = useState("5");
  const [maxSteps, setMaxSteps] = useState("2400");
  const [device, setDevice] = useState("cuda");
  const experiments: string[] = state?.experiments ?? [];
  const trainingRuns: any[] = state?.training_runs ?? [];
  const runs = trainingRuns.filter((item: any) => item.experiment === experiment && item.policy === policy);
  const selectedRun = runs.find((item: any) => item.name === trainedRun);
  const checkpoints: any[] = selectedRun?.checkpoints ?? [];
  const dataset = selectedRun?.dataset?.path ?? "";
  const selectedCheckpoint = checkpoints.find((item: any) => item.path === checkpoint);
  const entrypointAvailable = Boolean(state?.entrypoint_available?.[policy]);
  const status = state?.status?.state ?? "IDLE";

  useEffect(() => {
    if (experiments.length > 0 && !experiments.includes(experiment)) setExperiment(experiments[0]);
  }, [experiment, experiments]);

  useEffect(() => {
    if (!runs.some((item: any) => item.name === trainedRun)) setTrainedRun(runs[0]?.name ?? "");
  }, [trainedRun, runs]);

  useEffect(() => {
    if (!checkpoints.some((item: any) => item.path === checkpoint)) {
      setCheckpoint(selectedRun?.default_checkpoint ?? checkpoints[checkpoints.length - 1]?.path ?? "");
    }
  }, [checkpoint, checkpoints, selectedRun]);

  const start = async () => {
    if (source === "mock") return onToast("Mock mode: evaluation was not started.");
    try {
      await post("/api/eval/start", {
        policy, checkpoint, experiment, trained_run: trainedRun, dataset_root: dataset, run_id: runId,
        episodes: Number(episodes), max_steps: Number(maxSteps),
        device, seed: Number(seed),
      });
      onToast("Evaluation start requested.");
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Evaluation start failed");
    }
  };

  const stop = async () => {
    if (source === "mock") return onToast("Mock mode: evaluation was not stopped.");
    try {
      await post("/api/eval/action", { action: "stop" });
      onToast("Evaluation stop requested.");
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Evaluation stop failed");
    }
  };

  return (
    <div className="eval-layout">
      <aside className="left-rail">
        <Card title="Evaluation Setup">
          <Field label="Experiment">
            <select value={experiment} onChange={(event) => { setExperiment(event.target.value); setTrainedRun(""); setCheckpoint(""); }}>
              {experiments.length === 0 && <option value={experiment}>No experiment found</option>}
              {experiments.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </Field>
          <Field label="Policy">
            <select value={policy} onChange={(event) => { setPolicy(event.target.value); setTrainedRun(""); setCheckpoint(""); }}>
              <option value="act">ACT</option>
              <option value="diffusion_policy">Diffusion Policy</option>
            </select>
          </Field>
          <Field label="Trained Run">
            <select value={trainedRun} onChange={(event) => { setTrainedRun(event.target.value); setCheckpoint(""); }}>
              {runs.length === 0 && <option value="">No loadable run found</option>}
              {runs.map((item: any) => <option key={item.path} value={item.name}>{item.name}</option>)}
            </select>
          </Field>
          <Field label="Checkpoint">
            <select value={checkpoint} onChange={(event) => setCheckpoint(event.target.value)}>
              {checkpoints.length === 0 && <option value="">No checkpoint found</option>}
              {checkpoints.map((item: any) => <option key={item.path} value={item.path}>{item.name}</option>)}
            </select>
          </Field>
          <Field label="Training Dataset"><input value={dataset || "No dataset provenance found"} readOnly /></Field>
          <Field label="Evaluation Distribution"><select value={region} onChange={(event) => setRegion(event.target.value)}><option value="overall">Overall (balanced A/B/C)</option><option value="A">Region A</option><option value="B">Region B</option><option value="C">Region C</option></select></Field>
          <Field label="Episodes"><input type="number" min="1" value={episodes} onChange={(event) => setEpisodes(event.target.value)} /></Field>
          <Field label="Max Steps"><input type="number" min="1" value={maxSteps} onChange={(event) => setMaxSteps(event.target.value)} /></Field>
          <Field label="Seed"><input type="number" value={seed} onChange={(event) => setSeed(event.target.value)} /></Field>
          <Field label="Device">
            <select value={device} onChange={(event) => setDevice(event.target.value)}><option value="cuda">cuda</option><option value="cpu">cpu</option></select>
          </Field>
          <Field label="Eval Run ID"><input value={runId} onChange={(event) => setRunId(event.target.value)} /></Field>
          <button className="primary-action" disabled={source === "live" && status !== "RUNNING" && (!checkpoint || !dataset || !entrypointAvailable)} onClick={status === "RUNNING" ? stop : start}>{status === "RUNNING" ? "■ STOP EVAL" : "START EVAL"}</button>
          {!entrypointAvailable && <p className="muted">Closed-loop evaluation entrypoint is not available. Discovery and checkpoint selection remain usable.</p>}
        </Card>
      </aside>
      <section className="camera-panel">
        <div className="camera-stage">
          <div className="camera-toolbar"><span className="tag">POLICY ROLLOUT · FRONT / OPERATOR VIEW · 1280×960 · 60 Hz — {source === "mock" ? "MOCK" : status}</span><span className="mono">{state?.status?.log_path ?? "No rollout active"}</span></div>
          <div className="legend"><span>● EE</span><span>⌗ Object</span><span>◉ Goal</span></div>
          <div className="scene">
            {state?.live?.frame_url ? (
              <LiveCameraImage className="rollout-frame" src={apiUrl(state.live.frame_url)} alt="Latest policy rollout front/operator camera frame" />
            ) : (
              <div className="scene-message">{source === "mock" ? "RUNNING · GRASP" : status === "RUNNING" ? "WAITING FOR ROLLOUT FRAME" : "NO ROLLOUT SOURCE"}</div>
            )}
          </div>
          <TrajectoryInset trajectory={{ available: Array.isArray(state?.live?.trajectory) && state.live.trajectory.length > 0, source: state?.live?.trajectory ? "live rollout EE XY" : null, points: (state?.live?.trajectory ?? []).map((point: any) => ({ x: point[0], y: point[1] })), message: "No rollout trajectory available" }} title="Live EE Trajectory" />
        </div>
        <pre className="event-log">{source === "mock" ? "Mock event stream" : (state?.log_lines ?? []).join("\n") || "No evaluation job has run."}</pre>
      </section>
      <aside className="right-rail">
        <Card title="Current Rollout">
          <InfoRow label="Status" value={status} />
          <InfoRow label="Policy" value={String(state?.live?.policy ?? policy).toUpperCase()} />
          <InfoRow label="Trained Run" value={trainedRun || "—"} />
          <InfoRow label="Checkpoint" value={state?.live?.checkpoint ?? selectedCheckpoint?.name ?? "—"} />
          <InfoRow label="Load Status" value={selectedRun?.status ?? "NOT AVAILABLE"} />
          <InfoRow label="Training Dataset" value={selectedRun?.dataset?.name ?? "—"} />
          <InfoRow label="Episode" value={state?.live ? `${state.live.episode ?? "—"}/${state.live.total_episodes ?? episodes}` : "—"} />
          <InfoRow label="Step" value={state?.live?.step == null ? "—" : String(state.live.step)} />
          <InfoRow label="Sim Time" value={state?.live?.simulation_time == null ? "—" : `${Number(state.live.simulation_time).toFixed(2)} s`} />
          <InfoRow label="Result" value={state?.result?.available ? `${state.result.success_rate ?? "—"} (${state.result.episodes ?? "—"} ep)` : state?.live?.termination_reason ?? "NOT AVAILABLE"} />
        </Card>
        <Card title="Evaluation Output">
          <p className="muted">Resolved checkpoint: {checkpoint || "—"}</p>
          <p className="muted">{state?.status?.log_path ?? "Run an evaluation to create rollout logs."}</p>
        </Card>
      </aside>
    </div>
  );
}

function AnalyzeView({ state, source }: { state: any; source: string }) {
  const runs = source === "mock" ? [] : (state?.runs ?? []);
  const details = source === "mock" ? {} : (state?.details ?? {});
  const [leftId, setLeftId] = useState("");
  const [rightId, setRightId] = useState("");
  const [episodeId, setEpisodeId] = useState(0);
  const [failureFilter, setFailureFilter] = useState("ALL");
  useEffect(() => { if (runs.length && !leftId) setLeftId(runs[0].run_id); }, [runs, leftId]);
  const left = details[leftId];
  const right = details[rightId];
  const selectedEpisode = left?.episode_results?.find((item: any) => item.episode_id === episodeId) ?? left?.episode_results?.[0];
  const firstPredictedAction = selectedEpisode?.predicted_actions?.[0];
  const firstExecutedAction = selectedEpisode?.executed_actions?.[0];
  const fmt = (value: any, suffix = "") => value == null ? "NOT AVAILABLE" : `${typeof value === "number" ? value.toFixed(4) : value}${suffix}`;
  const trajectory = selectedEpisode?.object_trajectory ?? [];
  const points = trajectory.map((point: any) => ({ x: point.x, y: point.y }));
  const failureRows = (left?.failure_counts ?? []).filter((row: any) => failureFilter === "ALL" || row.label === failureFilter);
  const comparison = right && left ? { compatible: left.dataset === right.dataset && left.max_steps === right.max_steps, warning: left.dataset !== right.dataset || left.max_steps !== right.max_steps ? "Dataset or evaluation protocol differs." : "" } : null;
  const paired = right && left ? left.episode_results.map((episode: any) => { const match = right.episode_results.find((item: any) => item.eval_seed === episode.eval_seed); if (!match) return null; const a = episode.success === true, b = match.success === true; return a && b ? "Both Success" : !a && !b ? "Both Failure" : a ? "ACT Success / DP Failure" : "ACT Failure / DP Success"; }).filter(Boolean) : [];
  const pairCounts = ["Both Success", "Both Failure", "ACT Success / DP Failure", "ACT Failure / DP Success"].map((category) => ({ category, count: paired.filter((value: string) => value === category).length }));
  return <div className="analyze-layout">
    <div className="lineage"><span>EXPERIMENT</span><select value={leftId} onChange={(event) => { setLeftId(event.target.value); setEpisodeId(0); }}>{runs.length === 0 && <option value="">No evaluation runs</option>}{runs.map((run: any) => <option key={run.run_id} value={run.run_id}>{run.run_id} · {String(run.policy ?? "unknown").toUpperCase()}</option>)}</select><select value={rightId} onChange={(event) => setRightId(event.target.value)}><option value="">Single run</option>{runs.map((run: any) => <option key={run.run_id} value={run.run_id}>{run.run_id}</option>)}</select></div>
    <div className="next-collection"><span>NEXT COLLECTION</span><strong>NOT COMPUTED</strong><small>Requires region-conditioned evaluation, coverage statistics, and feasibility estimates.</small></div>
    <div className="analysis-grid"><Card title="Baseline Summary" className="region-comparison">{!left && <div className="empty-state">No evaluation run selected.</div>}{left && <><InfoRow label="Policy" value={String(left.policy ?? "NOT AVAILABLE").toUpperCase()} /><InfoRow label="Checkpoint" value={left.checkpoint ?? "NOT AVAILABLE"} /><InfoRow label="Dataset" value={left.dataset ?? "NOT AVAILABLE"} /><InfoRow label="Episodes" value={String(left.episodes)} /><InfoRow label="Success" value={`${left.success_count}/${left.episodes}`} /><InfoRow label="Success Rate" value={fmt(left.success_rate == null ? null : left.success_rate * 100, "%")} /><InfoRow label="Mean Steps" value={fmt(left.summary.mean_steps)} /><InfoRow label="Median Steps" value={fmt(left.summary.median_steps)} /><InfoRow label="Mean Final Goal Distance" value={fmt(left.summary.mean_final_goal_distance, " m")} /><InfoRow label="EE Trajectory Length" value="NOT AVAILABLE" /></>}</Card><Card title="ACT vs DP Comparison"><p className="muted">{!comparison ? "Select a second evaluation run to compare." : comparison.warning || "Compatible baseline runs."}</p>{right && left && <><InfoRow label="Left Success Rate" value={fmt(left.success_rate == null ? null : left.success_rate * 100, "%")} /><InfoRow label="Right Success Rate" value={fmt(right.success_rate == null ? null : right.success_rate * 100, "%")} /><InfoRow label="Left Mean Steps" value={fmt(left.summary.mean_steps)} /><InfoRow label="Right Mean Steps" value={fmt(right.summary.mean_steps)} /><InfoRow label="Left Final Distance" value={fmt(left.summary.mean_final_goal_distance, " m")} /><InfoRow label="Right Final Distance" value={fmt(right.summary.mean_final_goal_distance, " m")} /><div className="muted">Paired episodes: {pairCounts.map((item) => `${item.category} ${item.count}`).join(" · ")}</div></>}</Card></div>
    <div className="analysis-grid"><Card title="Failure Analysis"><div className="filter-row"><button className={failureFilter === "ALL" ? "selected" : ""} onClick={() => setFailureFilter("ALL")}>ALL</button>{(left?.failure_counts ?? []).map((row: any) => <button key={row.label} className={failureFilter === row.label ? "selected" : ""} onClick={() => setFailureFilter(row.label)}>{row.label}</button>)}</div>{failureRows.map((row: any) => <FailureBar key={row.label} {...row} />)}{!left && <div className="empty-state">No evaluation run selected.</div>}</Card><Card title="Region Analysis">{left?.region_summary?.available ? <><div className="muted">{left.region_summary.version}</div>{left.region_summary.regions.map((row: any) => <div key={row.region} className="progress-row"><div><strong>Region {row.region}</strong><b>{row.success_count}/{row.episodes}</b></div><div className="bar"><i style={{ width: row.success_rate == null ? "0%" : `${row.success_rate * 100}%`, background: REGION_COLORS[row.region] }} /></div><span className="mono">Sπ({row.region}) {row.success_rate == null ? "NOT AVAILABLE" : `${(row.success_rate * 100).toFixed(1)}%`}</span></div>)}</> : <div className="empty-state">REGION NOT AVAILABLE<br />This run has no Region v1 metadata.</div>}</Card></div>
    <Card title="Rollout Inspector" className="inspector"><div className="inspector-layout"><div className="rollout-list">{(left?.episode_results ?? []).map((episode: any) => <button key={episode.episode_id} onClick={() => setEpisodeId(episode.episode_id)}>{String(episode.episode_id).padStart(3, "0")} {episode.success ? "SUCCESS" : "FAILURE"}<small> seed {episode.eval_seed ?? "—"} · {episode.steps ?? "—"} steps</small></button>)}</div><div className="inspector-main"><div className="inspector-legend">Object trajectory / goal-relative inspection</div><div className="inspector-plot"><TrajectoryInset trajectory={{ available: points.length > 1, source: selectedEpisode ? `episode ${selectedEpisode.episode_id}` : null, points, message: "No object trajectory available" }} title={selectedEpisode ? `Episode ${selectedEpisode.episode_id}` : "No episode selected"} /></div></div><div className="diagnosis"><Card title="Selected Episode"><InfoRow label="Policy" value={String(left?.policy ?? "NOT AVAILABLE").toUpperCase()} /><InfoRow label="Seed" value={String(selectedEpisode?.eval_seed ?? "NOT AVAILABLE")} /><InfoRow label="Steps" value={String(selectedEpisode?.steps ?? "NOT AVAILABLE")} /><InfoRow label="Final Distance" value={fmt(selectedEpisode?.final_goal_distance, " m")} /><InfoRow label="Termination" value={selectedEpisode?.termination_reason ?? "NOT AVAILABLE"} /><InfoRow label="Action Delta Mean" value={fmt(selectedEpisode?.action_execution_delta_mean)} /><InfoRow label="Action Delta Max" value={fmt(selectedEpisode?.action_execution_delta_max)} /><div className="action-inspector"><b>Step 0 · Predicted Action</b><code>{firstPredictedAction ? JSON.stringify(firstPredictedAction) : "NOT AVAILABLE"}</code><b>Step 0 · Executed Action</b><code>{firstExecutedAction ? JSON.stringify(firstExecutedAction) : "NOT AVAILABLE"}</code></div></Card></div></div></Card>
  </div>;
}

function Card({ title, children, className = "", action }: { title: string; children: ReactNode; className?: string; action?: ReactNode }) { return <section className={`card ${className}`}><div className="card-title">{title}{action}</div>{children}</section>; }
function StatusRow({ label, status }: { label: string; status: Status }) { const good = ["connected", "active", "available", "recording"].includes(status.state); return <div className="status-row"><span>{label}</span><Pill text={status.state === "unavailable" ? "Unavailable" : status.state} tone={good ? "good" : "neutral"} /></div>; }
function InfoRow({ label, value }: { label: string; value: string }) { return <div className="info-row"><span>{label}</span><b>{value}</b></div>; }
function Field({ label, children }: { label: string; children: ReactNode }) { return <label className="field"><span>{label}</span>{children}</label>; }
function Stat({ label, value }: { label: string; value: string }) { return <div><span>{label}</span><b>{value}</b></div>; }
function Metric({ title, value, color }: { title: string; value: string; color: string }) { return <div className="metric"><div><span>{title}</span><b>{value}</b></div><svg viewBox="0 0 120 36" preserveAspectRatio="none"><polyline points="0,5 18,12 36,17 54,22 72,26 90,29 120,32" fill="none" stroke={color} strokeWidth="1.5" /></svg></div>; }
function MetricBar({ label, value, color }: { label: string; value: string; color: string }) { const numeric = Number.parseFloat(value); return <div className="metric-bar"><span>{label}</span><div className="bar"><i style={{ width: Number.isFinite(numeric) ? `${numeric * 100}%` : "0%", background: color }} /></div><b>{value}</b></div>; }
function FailureBar({ label, count }: { label: string; count: number }) { return <div className="failure-bar"><span>{label}</span><div className="bar"><i style={{ width: `${Math.min(100, count * 5)}%` }} /></div><b>{count}</b></div>; }
function Pill({ text, tone }: { text: string; tone: "good" | "neutral" }) { return <span className={`pill ${tone}`}>{text}</span>; }
function formatNumber(value: number | null | undefined) { return value == null ? "—" : String(value).padStart(3, "0"); }
function formatTime(value: number | null | undefined) { if (value == null) return "—"; const seconds = Math.floor(value); return `00:${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`; }
function countText(value: number | null | undefined) { return value == null ? "—" : String(value); }
function formatPose(value: number[] | null | undefined) { return value && value.length >= 3 ? value.slice(0, 3).map((item) => Number(item).toFixed(3)).join(", ") : "UNAVAILABLE"; }
function normalizePoints(points: Point[]) { if (points.length < 2) return []; const xs = points.map((point) => point.x); const ys = points.map((point) => point.y); const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys); return points.map((point) => ({ x: 8 + ((point.x - minX) / Math.max(1e-9, maxX - minX)) * 144, y: 88 - ((point.y - minY) / Math.max(1e-9, maxY - minY)) * 80 })); }
