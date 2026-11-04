// Shape of public/demo.json, written by `aerotrack demo-export` (src/aerotrack/demo.py).

// x1, y1, x2, y2 (pixels of the original frame), track id, class id
export type Track = [number, number, number, number, number, number];

export type Clip = {
  sequence: string;
  caption: string;
  dir: string;
  width: number;
  height: number;
  frames: number;
  first_frame: number;
  zoom: {id: number; w: number; h: number} | null;
  stages: string[];
  timing_ms: number[][];
  tracks: Track[][];
};

export type TrackingModel = {
  model: string;
  imgsz: number;
  HOTA: number;
  IDF1: number;
  fps: number;
  total_ms: number;
  backend: string;
  device: string;
  versions: string;
  run_id: string;
};

export type DemoData = {
  run_id: string;
  fps: number;
  model: string;
  train_run: string;
  split: string;
  classes: Record<string, string>;
  device: string;
  versions: string;
  median_ms: number;
  p95_ms: number;
  clips: Clip[];
  headline: {
    zero_shot: {
      report: string;
      split: string;
      models: {model: string; 'mAP50-95': number; run_id: string}[];
    };
    tracking: {report: string; split: string; budget_fps: number; models: TrackingModel[]};
  };
};
