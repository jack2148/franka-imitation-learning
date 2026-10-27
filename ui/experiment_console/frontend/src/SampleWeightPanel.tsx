import { useEffect, useRef, useState } from "react";
import { post } from "./api";
import type { WorkState } from "./types";

// Codes deliberately exclude punctuation, modifiers, repeat and typing targets.
export function weightKey(event: KeyboardEvent): number | null {
  const target = event.target instanceof HTMLElement ? event.target : null;
  if (event.defaultPrevented || event.repeat || event.isComposing || event.ctrlKey || event.altKey || event.metaKey || event.shiftKey
      || target?.isContentEditable || target?.closest("input,textarea,select,[contenteditable]:not([contenteditable='false'])")) return null;
  const match = /^(?:Digit|Numpad)([123])$/.exec(event.code);
  return match ? Number(match[1]) : null;
}

export function SampleWeightPanel({ state, source }: { state: WorkState; source: string }) {
  const [display, setDisplay] = useState(state.weight?.current_weight ?? 1);
  const [error, setError] = useState("");
  const [waiting, setWaiting] = useState(false);
  const latest = useRef({ state, source });
  latest.current = { state, source };
  const pending = useRef<string | null>(null);
  const episode = useRef(state.weight?.recording_id);
  const queue = useRef(Promise.resolve());
  const timeout = useRef<number>();
  const mounted = useRef(true);
  const clearPending = () => {
    pending.current = null;
    window.clearTimeout(timeout.current);
    setWaiting(false);
  };
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; window.clearTimeout(timeout.current); };
  }, []);
  useEffect(() => {
    const annotation = state.weight;
    if (episode.current !== annotation?.recording_id || !state.recording) {
      episode.current = annotation?.recording_id;
      clearPending();
      setError("");
    } else if (pending.current === annotation?.request_id) {
      clearPending();
    }
    if (!pending.current) setDisplay(annotation?.current_weight ?? 1);
  }, [state.recording, state.weight?.recording_id, state.weight?.current_weight, state.weight?.request_id]);

  const select = (value: number) => {
    const { state: current, source: currentSource } = latest.current;
    if (!current.recording || !current.weight || currentSource === "mock") return;
    const recording_id = current.weight.recording_id;
    const request_id = crypto.randomUUID();
    pending.current = request_id;
    setDisplay(value); // Optimistic UI only; the worker owns persisted labels.
    setError(""); setWaiting(true);
    window.clearTimeout(timeout.current);
    const reconcileFailure = (message: string) => {
      if (!mounted.current || pending.current !== request_id) return;
      clearPending();
      setDisplay(latest.current.state.weight?.current_weight ?? 1);
      setError(message);
    };
    timeout.current = window.setTimeout(() => reconcileFailure("Weight 확인 지연 · 서버 값으로 표시합니다."), 2000);
    // Preserve key order across HTTP requests, including rapid 1 -> 3 -> 2.
    queue.current = queue.current.catch(() => {}).then(async () => {
      if (!latest.current.state.recording || latest.current.state.weight?.recording_id !== recording_id) return;
      const controller = new AbortController();
      const requestTimeout = window.setTimeout(() => controller.abort(), 2000);
      try { await post("/api/work/weight", { weight: value, recording_id, request_id }, controller.signal); }
      catch (failure) { reconcileFailure(failure instanceof Error ? failure.message : "Weight update failed"); }
      finally { window.clearTimeout(requestTimeout); }
    });
  };
  const selectRef = useRef(select);
  selectRef.current = select;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const value = weightKey(event);
      const current = latest.current;
      if (value === null || !current.state.recording || !current.state.weight || current.source === "mock") return;
      event.preventDefault();
      selectRef.current(value);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return <section className="card sample-weight-panel" aria-label="Current weight">
    <h3>CURRENT WEIGHT</h3>
    <output className="current-weight" aria-label="Current weight value">{display}</output>
    <div className="weight-buttons">{[1, 2, 3].map(value =>
      <button key={value} aria-pressed={display === value} className={display === value ? "selected" : ""}
        disabled={!state.recording || !state.weight || source === "mock"}
        onClick={() => select(value)}>[{value}] Weight {value}</button>)}</div>
    <small>{waiting ? "서버 확인 중" : "다음 선택까지 유지 · 새 녹화는 1"}</small>
    {error && <p role="alert">{error}</p>}
  </section>;
}

export function TeleopModeButtons({ state, source, geomagic }: { state: WorkState; source: string; geomagic: boolean }) {
  const [error, setError] = useState("");
  const queue = useRef(Promise.resolve());
  const latest = useRef({ source, geomagic });
  latest.current = { source, geomagic };
  const modes = geomagic ? ["position", "rotation_z", "rotation", "6dof"] as const : ["position", "rotation"] as const;
  const labels = { position: "Position", rotation_z: "Rotation Z", rotation: geomagic ? "Rotation 3D" : "Rotation Z", "6dof": "6DoF" };
  const keys = { position: "A", rotation_z: "S", rotation: "D", "6dof": "F" };
  const select = (mode: WorkState["teleop_mode"]) => {
    if (latest.current.source === "mock") return;
    setError("");
    queue.current = queue.current.catch(() => {}).then(async () => {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 2000);
      try { await post("/api/teleop/mode", { mode }, controller.signal); }
      catch (e) { setError(String(e)); }
      finally { window.clearTimeout(timeout); }
    });
  };
  const selectRef = useRef(select);
  selectRef.current = select;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!latest.current.geomagic || latest.current.source === "mock") return;
      const target = event.target instanceof HTMLElement ? event.target : null;
      if (event.defaultPrevented || event.repeat || event.isComposing || event.ctrlKey || event.altKey || event.metaKey || event.shiftKey
          || target?.isContentEditable || target?.closest("input,textarea,select,[contenteditable]:not([contenteditable='false'])")) return;
      const mode = ({ KeyA: "position", KeyS: "rotation_z", KeyD: "rotation", KeyF: "6dof" } as const)[event.code as "KeyA" | "KeyS" | "KeyD" | "KeyF"];
      if (!mode) return;
      event.preventDefault();
      selectRef.current(mode);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return <><div className="weight-buttons">{modes.map(mode => <button key={mode}
    aria-pressed={state.teleop_mode === mode} disabled={source === "mock"}
    onClick={() => select(mode)}>
      {geomagic ? `[${keys[mode]}] ` : ""}{labels[mode]}</button>)}</div>{error && <p role="alert">{error}</p>}</>;
}
