import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, mono} from '../theme';
import type {DemoData} from '../types';

const STACK = ['PyTorch', 'YOLO26', 'TensorRT FP16', 'C++ / CUDA', 'ByteTrack', 'Kalman filter from scratch', 'TrackEval'];

export const Outro = ({data}: {data: DemoData}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame, fps, config: {damping: 200}});
  const fadeIn = interpolate(frame, [0, 10], [0, 1], {extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', opacity: fadeIn}}>
      <div style={{fontSize: 120, fontWeight: 800, letterSpacing: -4, transform: `scale(${0.9 + 0.1 * s})`}}>
        aero<span style={{color: C.accent}}>track</span>
      </div>
      <div style={{fontFamily: mono, fontSize: 34, marginTop: 10}}>github.com/louisrodembourg/aerotrack</div>
      <div style={{display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 14, marginTop: 44, maxWidth: 1300}}>
        {STACK.map((t, i) => {
          const c = spring({frame: frame - 8 - i * 3, fps, config: {damping: 200}});
          return (
            <div key={t} style={{
              fontSize: 24, padding: '8px 20px', borderRadius: 999, border: `1.5px solid ${C.line}`,
              opacity: c, transform: `translateY(${(1 - c) * 14}px)`,
            }}>{t}</div>
          );
        })}
      </div>
      <div style={{fontSize: 22, color: C.dim, marginTop: 44, opacity: spring({frame: frame - 30, fps, config: {damping: 200}})}}>
        Every number in this video comes from a tracked run
      </div>
    </AbsoluteFill>
  );
};
