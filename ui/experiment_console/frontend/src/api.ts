import type { Mode } from "./types";

export const API_ROOT = ((import.meta as any).env?.VITE_API_ROOT as string | undefined) ?? "http://127.0.0.1:8000";

export function apiUrl(path: string): string {
  return new URL(path, `${API_ROOT}/`).toString();
}

export async function getState(view: Mode): Promise<any> {
  const response = await fetch(`${API_ROOT}/api/state?mode=live&view=${view}`, { cache: "no-store" });
  if (!response.ok) throw new Error((await response.json()).error ?? `API ${response.status}`);
  return response.json();
}

export async function post(path: string, body: Record<string, unknown>, signal?: AbortSignal): Promise<any> {
  const response = await fetch(`${API_ROOT}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error ?? `API ${response.status}`);
  return result;
}

export async function getDatasets(): Promise<any[]> {
  const response = await fetch(`${API_ROOT}/api/datasets`, { cache: "no-store" });
  if (!response.ok) throw new Error((await response.json()).error ?? `API ${response.status}`);
  return (await response.json()).datasets ?? [];
}

export type FolderPickerPurpose =
  | "dataset_output"
  | "raw_input"
  | "processed_output"
  | "training_input"
  | "training_output";

export type FolderPickerResult =
  | { cancelled: true }
  | { cancelled: false; path: string; writable: boolean; free_bytes: number };

export async function selectNativeFolder(
  purpose: FolderPickerPurpose,
  initialPath: string,
): Promise<FolderPickerResult> {
  return post("/api/folder-picker", { purpose, initial_path: initialPath });
}

export async function validateExperimentPath(path: string): Promise<any> {
  const response = await fetch(`${API_ROOT}/api/experiment/validate?path=${encodeURIComponent(path)}`, { cache: "no-store" });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error ?? `API ${response.status}`);
  return result;
}
