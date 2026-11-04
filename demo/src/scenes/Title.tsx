import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, mono} from '../theme';
import type {DemoData} from '../types';

const Grid = ({frame}: {frame: number}) => (
  <svg width={1920} height={1080} style={{position: 'absolute', opacity: 0.35}}>
    <defs>
      <pattern id="g" width={80} height={80} patternUnits="userSpaceOnUse"
        patternTransform={`translate(0 ${(frame * 0.8) % 80})`}>
        <path d="M80 0H0V80" fill="none" stroke={C.line} strokeWidth={1} />
      </pattern>
      <radialGradient id="fade">
        <stop offset="0" stopColor="#fff" stopOpacity={1} />
        <stop offset="1" stopColor="#fff" stopOpacity={0} />
      </radialGradient>
      <mask id="m"><rect width={1920} height={1080} fill="url(#fade)" /></mask>
    </defs>
    <rect width={1920} height={1080} fill="url(#g)" mask="url(#m)" />
  </svg>
);

export const Title = ({data}: {data: DemoData}) => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();
  const word = 'aerotrack';
  const out = interpolate(frame, [durationInFrames - 12, durationInFrames], [1, 0], {
    extrapolateLeft: 'clamp',
  });
  const sub = spring({frame: frame - 22, fps, config: {damping: 200}});
  const chips = [
    `${data.headline.tracking.budget_fps} FPS budget, end to end`,
    'batch 1',
    'detection + tracking',
  ];
  // A scan line crossing the title once, like a sensor sweep.
  const scan = interpolate(frame, [8, 40], [-10, 110], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <AbsoluteFill style={{opacity: out, justifyContent: 'center', alignItems: 'center'}}>
      <Grid frame={frame} />
      <div style={{position: 'relative', display: 'flex', fontSize: 170, fontWeight: 800, letterSpacing: -6}}>
        {word.split('').map((ch, i) => {
          const s = spring({frame: frame - 4 - i * 2, fps, config: {damping: 14, mass: 0.6}});
          return (
            <span key={i} style={{
              display: 'inline-block', transform: `translateY(${(1 - s) * 60}px)`, opacity: s,
              color: i < 4 ? C.text : C.accent,
            }}>{ch}</span>
          );
        })}
        <div style={{
          position: 'absolute', top: -20, bottom: -20, left: `${scan}%`, width: 4,
          background: C.accent, boxShadow: `0 0 30px 8px ${C.accent}`, opacity: scan > 105 ? 0 : 0.8,
        }} />
      </div>
      <div style={{
        marginTop: 18, fontSize: 40, color: C.dim, opacity: sub, transform: `translateY(${(1 - sub) * 20}px)`,
      }}>
        Real-time detection and tracking of vehicles and pedestrians from a drone
      </div>
      <div style={{display: 'flex', gap: 16, marginTop: 44}}>
        {chips.map((c, i) => {
          const s = spring({frame: frame - 40 - i * 5, fps, config: {damping: 200}});
          return (
            <div key={c} style={{
              fontFamily: mono, fontSize: 26, padding: '10px 22px', borderRadius: 999,
              border: `1.5px solid ${i === 0 ? C.accent : C.line}`, color: i === 0 ? C.accent : C.text,
              opacity: s, transform: `scale(${0.8 + 0.2 * s})`,
            }}>{c}</div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
