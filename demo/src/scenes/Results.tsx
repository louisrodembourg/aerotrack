import {AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, mono} from '../theme';
import type {DemoData, TrackingModel} from '../types';

const BACKEND: Record<string, string> = {tensorrt: 'Python + TensorRT', cpp: 'C++ runtime', torch: 'PyTorch'};

const Bar = ({m, max, budget, delay}: {m: TrackingModel; max: number; budget: number; delay: number}) => {
  const frame = useCurrentFrame();
  const grow = interpolate(frame, [delay, delay + 30], [0, 1], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic),
  });
  const width = 900;
  const cpp = m.backend === 'cpp';
  return (
    <div style={{display: 'flex', alignItems: 'center', gap: 24, height: 76}}>
      <div style={{width: 300, textAlign: 'right'}}>
        <div style={{fontSize: 26, fontWeight: 700}}>{BACKEND[m.backend] ?? m.backend}</div>
        <div style={{fontFamily: mono, fontSize: 18, color: C.dim}}>YOLO26n @ {m.imgsz} · HOTA {(m.HOTA * 100).toFixed(1)}</div>
      </div>
      <div style={{position: 'relative', width, height: 44}}>
        <div style={{
          position: 'absolute', left: 0, top: 0, bottom: 0, width: (m.fps / max) * width * grow, borderRadius: 8,
          background: cpp ? C.accent : 'rgba(61, 220, 255, 0.35)',
        }} />
        <div style={{
          position: 'absolute', left: (m.fps / max) * width * grow + 14, top: 4, fontFamily: mono, fontSize: 28,
          fontWeight: 600, whiteSpace: 'nowrap', opacity: grow,
        }}>
          {(m.fps * grow).toFixed(0)} FPS <span style={{color: C.dim, fontSize: 20}}>{m.total_ms.toFixed(1)} ms</span>
        </div>
        <div style={{position: 'absolute', left: (budget / max) * width, top: -10, bottom: -10, width: 3, background: C.budget}} />
      </div>
    </div>
  );
};

export const Results = ({data}: {data: DemoData}) => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();
  const {zero_shot: zs, tracking} = data.headline;
  const models = [...tracking.models].sort((a, b) => b.imgsz - a.imgsz || a.fps - b.fps);
  const max = Math.max(...models.map((m) => m.fps)) * 1.25;
  const py = models.find((m) => m.imgsz === 1280 && m.backend !== 'cpp')!;
  const cpp = models.find((m) => m.imgsz === 1280 && m.backend === 'cpp')!;
  const [coco, tuned] = zs.models.map((m) => m['mAP50-95'] * 100);
  const count = interpolate(frame, [20, 55], [coco, tuned], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const fadeIn = interpolate(frame, [0, 10], [0, 1], {extrapolateRight: 'clamp'});
  const fadeOut = interpolate(frame, [durationInFrames - 10, durationInFrames], [1, 0], {extrapolateLeft: 'clamp'});
  const head = spring({frame, fps, config: {damping: 200}});
  const budget = tracking.budget_fps;

  return (
    <AbsoluteFill style={{padding: '80px 100px', opacity: fadeIn * fadeOut}}>
      <div style={{fontFamily: mono, fontSize: 22, color: C.accent, opacity: head}}>THE TRADE-OFF, MEASURED</div>
      <div style={{fontSize: 58, fontWeight: 800, marginTop: 8, opacity: head, transform: `translateY(${(1 - head) * 20}px)`}}>
        Same accuracy, {(cpp.fps / py.fps).toFixed(1)}× faster in C++
      </div>
      <div style={{display: 'flex', gap: 70, marginTop: 56}}>
        <div style={{
          width: 440, padding: 32, borderRadius: 20, background: C.panel, border: `1px solid ${C.line}`,
          opacity: spring({frame: frame - 8, fps, config: {damping: 200}}),
        }}>
          <div style={{fontSize: 26, fontWeight: 700}}>Fine-tuning on drone images</div>
          <div style={{fontSize: 18, color: C.dim, marginTop: 6}}>YOLO26n @ 640 · mAP50-95 · {zs.split}</div>
          <div style={{display: 'flex', alignItems: 'baseline', gap: 16, marginTop: 34}}>
            <span style={{fontFamily: mono, fontSize: 40, color: C.dim}}>{coco.toFixed(1)}</span>
            <span style={{fontSize: 34, color: C.dim}}>→</span>
            <span style={{fontFamily: mono, fontSize: 92, fontWeight: 600, color: C.accent}}>{count.toFixed(1)}</span>
          </div>
          <div style={{fontSize: 20, color: C.dim, marginTop: 8}}>COCO weights zero-shot → fine-tuned, on the classes COCO can name</div>
        </div>
        <div>
          <div style={{fontSize: 26, fontWeight: 700}}>Detection + tracking, end to end</div>
          <div style={{fontSize: 18, color: C.dim, marginTop: 6, marginBottom: 22}}>
            {cpp.device} · {cpp.versions} · TensorRT FP16 · batch 1 · HOTA on {tracking.split}, sequences unseen by the detector
          </div>
          {models.map((m, k) => <Bar key={m.run_id} m={m} max={max} budget={budget} delay={22 + k * 10} />)}
          <div style={{fontFamily: mono, fontSize: 18, color: C.budget, marginLeft: 324 + (budget / max) * 900 - 10, marginTop: 6}}>
            ↑ {budget} FPS budget
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
