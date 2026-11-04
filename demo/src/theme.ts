import {loadFont as loadInter} from '@remotion/google-fonts/Inter';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';

export const sans = loadInter('normal', {weights: ['400', '500', '700', '800'], subsets: ['latin']})
  .fontFamily;
export const mono = loadMono('normal', {weights: ['400', '600'], subsets: ['latin']}).fontFamily;

export const C = {
  bg: '#070B10',
  panel: 'rgba(9, 14, 20, 0.78)',
  line: 'rgba(255, 255, 255, 0.12)',
  text: '#EEF3F8',
  dim: '#8C99A8',
  accent: '#3DDCFF',
  warn: '#FFB547',
  ok: '#5CF29B',
  budget: '#FF5A6E',
};

export const CLASS_COLOR: Record<string, string> = {
  car: '#3DDCFF',
  van: '#8C9BFF',
  truck: '#FFB547',
  bus: '#FF7AB6',
  pedestrian: '#5CF29B',
};

export const STAGE_COLOR: Record<string, string> = {
  preprocess: '#8C9BFF',
  inference: '#3DDCFF',
  postprocess: '#FFB547',
  tracking: '#5CF29B',
};

// Timeline, in frames at data.fps (25: the project's budget).
export const T = {
  title: 100,
  situation: 125,
  clip: 150, // must match configs/demo/video.yaml (checked in Root)
  results: 200,
  outro: 100,
};
