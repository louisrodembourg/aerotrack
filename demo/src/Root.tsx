import {Composition, staticFile} from 'remotion';
import {Demo} from './Demo';
import {T} from './theme';
import type {DemoData} from './types';

export const RemotionRoot = () => (
  <Composition
    id="Demo"
    component={Demo}
    width={1920}
    height={1080}
    fps={25}
    durationInFrames={1}
    defaultProps={{data: null as DemoData | null}}
    calculateMetadata={async () => {
      const res = await fetch(staticFile('demo.json'));
      if (!res.ok) throw new Error('public/demo.json missing: run `uv run aerotrack demo-export`');
      const data = (await res.json()) as DemoData;
      for (const c of data.clips) {
        if (c.frames !== T.clip) throw new Error(`${c.sequence}: ${c.frames} frames, the timeline expects ${T.clip}`);
      }
      const total = T.title + T.situation + data.clips.length * T.clip + T.results + T.outro;
      return {durationInFrames: total, fps: data.fps, props: {data}};
    }}
  />
);
