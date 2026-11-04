import {useMemo} from 'react';
import {AbsoluteFill, Img, interpolate, Sequence, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, CLASS_COLOR, mono, STAGE_COLOR, T} from '../theme';
import type {Clip, DemoData, Track} from '../types';

const TRAIL = 18; // frames of trajectory behind each track
const JUMP = 40; // screen px between consecutive centres that breaks a trail
const MEAN = 8; // frames averaged in the big latency figure (raw values flicker unreadably)
const W = 1920;
const H = 1080;

const framePath = (clip: Clip, i: number) => staticFile(`${clip.dir}/${String(i).padStart(4, '0')}.jpg`);

type Index = {byId: Map<number, Track>[]; age: Map<number, number>[]; seen: number[]};

// Per frame: id -> box, id -> consecutive frames present so far, distinct ids so far.
const indexClip = (clip: Clip): Index => {
  const byId: Map<number, Track>[] = [];
  const age: Map<number, number>[] = [];
  const seen: number[] = [];
  const all = new Set<number>();
  clip.tracks.forEach((tracks, i) => {
    const m = new Map(tracks.map((t) => [t[4], t] as const));
    const a = new Map<number, number>();
    for (const id of m.keys()) {
      a.set(id, (i > 0 ? age[i - 1].get(id) ?? 0 : 0) + 1);
      all.add(id);
    }
    byId.push(m);
    age.push(a);
    seen.push(all.size);
  });
  return {byId, age, seen};
};

const Overlay = ({clip, idx, i, names}: {clip: Clip; idx: Index; i: number; names: Record<string, string>}) => {
  const s = W / clip.width; // screen px per frame px
  return (
    <svg width={W} height={H} viewBox={`0 0 ${clip.width} ${clip.height}`} preserveAspectRatio="none"
      style={{position: 'absolute'}}>
      {clip.tracks[i].map(([x1, y1, x2, y2, id, cls]) => {
        const color = CLASS_COLOR[names[cls]] ?? C.accent;
        // Trail segments; a jump (box entering through a frame edge) starts a new one.
        const segs: string[][] = [[]];
        let prev: [number, number] | null = null;
        for (let k = Math.max(0, i - TRAIL); k <= i; k++) {
          const b = idx.byId[k].get(id);
          if (!b) continue;
          const c: [number, number] = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
          if (prev && Math.hypot(c[0] - prev[0], c[1] - prev[1]) > JUMP / s) segs.push([]);
          segs[segs.length - 1].push(`${c[0]},${c[1]}`);
          prev = c;
        }
        const age = idx.age[i].get(id) ?? 1;
        const pop = interpolate(age, [1, 6], [1, 0], {extrapolateRight: 'clamp'});
        const pad = (pop * 14) / s;
        return (
          <g key={id}>
            {segs.map((pts, k) => (
              <polyline key={k} points={pts.join(' ')} fill="none" stroke={color} strokeOpacity={0.55}
                strokeWidth={2} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
            ))}
            <rect x={x1 - pad} y={y1 - pad} width={x2 - x1 + 2 * pad} height={y2 - y1 + 2 * pad}
              fill={color} fillOpacity={0.08 + pop * 0.2} stroke={color} strokeWidth={2}
              vectorEffect="non-scaling-stroke" />
            {(y2 - y1) * s > 16 && (
              <text x={x1} y={y1 - 4 / s} fill={color} fontSize={13 / s} fontFamily={mono} fontWeight={600}>
                {id}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
};

const Hud = ({data, clip, idx, i}: {data: DemoData; clip: Clip; idx: Index; i: number}) => {
  const {fps} = useVideoConfig();
  const budget = data.headline.tracking.budget_fps;
  const budgetMs = 1000 / budget;
  const totals = clip.timing_ms.map((t) => t.reduce((a, b) => a + b, 0));
  const recent = totals.slice(Math.max(0, i - MEAN + 1), i + 1);
  const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
  const ok = 1000 / mean >= budget;
  const scale = budgetMs * 2.5; // bar spans 0..2.5x the budget
  const enter = spring({frame: i, fps, config: {damping: 200}});

  const counts: Record<string, number> = {};
  for (const t of clip.tracks[i]) counts[data.classes[t[5]]] = (counts[data.classes[t[5]]] ?? 0) + 1;

  const spark = totals.slice(Math.max(0, i - 49), i + 1);
  const sx = (k: number) => (k / 49) * 440;
  const sy = (v: number) => 70 - Math.min(v / scale, 1) * 70;

  return (
    <div style={{
      position: 'absolute', top: 36, right: 36, width: 500, padding: 26, borderRadius: 18,
      background: C.panel, border: `1px solid ${C.line}`, backdropFilter: 'blur(8px)',
      transform: `translateX(${(1 - enter) * 40}px)`, opacity: enter,
    }}>
      <div style={{display: 'flex', justifyContent: 'space-between', fontFamily: mono, fontSize: 18, color: C.dim}}>
        <span>
          <span style={{color: C.budget, opacity: Math.floor(i / 12) % 2 ? 0.35 : 1}}>●</span> LIVE PIPELINE
        </span>
        <span>frame {String(clip.first_frame + i).padStart(4, '0')}</span>
      </div>
      <div style={{display: 'flex', alignItems: 'baseline', gap: 14, marginTop: 14}}>
        <span style={{fontFamily: mono, fontSize: 64, fontWeight: 600}}>{mean.toFixed(1)}</span>
        <span style={{fontSize: 22, color: C.dim, whiteSpace: 'nowrap'}}>ms / frame</span>
        <span style={{marginLeft: 'auto', fontFamily: mono, fontSize: 28, fontWeight: 600, whiteSpace: 'nowrap', color: ok ? C.ok : C.warn}}>
          {(1000 / mean).toFixed(0)} FPS
        </span>
      </div>
      <div style={{fontSize: 15, color: C.dim, marginTop: 2}}>
        mean of the last {MEAN} frames · detection + tracking, measured in this run
      </div>
      {/* stage breakdown of the current frame against the budget */}
      <div style={{position: 'relative', height: 22, marginTop: 20, borderRadius: 6, background: 'rgba(255,255,255,0.06)', overflow: 'hidden'}}>
        {(() => {
          let x = 0;
          return clip.timing_ms[i].map((v, k) => {
            const w = Math.max(0, Math.min(v / scale, 1 - x));
            const el = <div key={k} style={{position: 'absolute', left: `${x * 100}%`, width: `${w * 100}%`, top: 0, bottom: 0, background: STAGE_COLOR[clip.stages[k]]}} />;
            x += w;
            return el;
          });
        })()}
        <div style={{position: 'absolute', left: `${(budgetMs / scale) * 100}%`, top: 0, bottom: 0, width: 3, background: C.budget}} />
      </div>
      <div style={{position: 'relative', height: 20, fontFamily: mono, fontSize: 14, color: C.budget}}>
        <span style={{position: 'absolute', left: `${(budgetMs / scale) * 100}%`, transform: 'translateX(-50%)', top: 4}}>
          {budget} FPS budget
        </span>
      </div>
      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 18px', marginTop: 12, fontSize: 18}}>
        {clip.stages.map((st, k) => (
          <div key={st} style={{display: 'flex', alignItems: 'center', gap: 8}}>
            <span style={{width: 12, height: 12, borderRadius: 3, background: STAGE_COLOR[st]}} />
            <span style={{color: C.dim}}>{st === 'postprocess' ? 'NMS' : st}</span>
            <span style={{marginLeft: 'auto', fontFamily: mono}}>{clip.timing_ms[i][k].toFixed(1)}</span>
          </div>
        ))}
      </div>
      <svg width={440} height={74} style={{marginTop: 16, display: 'block'}}>
        <line x1={0} x2={440} y1={sy(budgetMs)} y2={sy(budgetMs)} stroke={C.budget} strokeDasharray="5 5" strokeWidth={1.5} />
        <polyline fill="none" stroke={C.accent} strokeWidth={2}
          points={spark.map((v, k) => `${sx(k + 49 - (spark.length - 1))},${sy(v)}`).join(' ')} />
      </svg>
      <div style={{height: 1, background: C.line, margin: '18px 0'}} />
      <div style={{display: 'flex', alignItems: 'baseline', gap: 12}}>
        <span style={{fontFamily: mono, fontSize: 44, fontWeight: 600}}>{clip.tracks[i].length}</span>
        <span style={{fontSize: 22, color: C.dim}}>objects tracked</span>
        <span style={{marginLeft: 'auto', fontFamily: mono, fontSize: 18, color: C.dim}}>{idx.seen[i]} ids so far</span>
      </div>
      <div style={{display: 'flex', flexWrap: 'wrap', gap: '6px 16px', marginTop: 10, fontSize: 18}}>
        {Object.values(data.classes).filter((n) => counts[n]).map((n) => (
          <span key={n} style={{display: 'flex', alignItems: 'center', gap: 6}}>
            <span style={{width: 10, height: 10, borderRadius: 5, background: CLASS_COLOR[n]}} />
            {n} <span style={{fontFamily: mono, color: C.dim}}>{counts[n]}</span>
          </span>
        ))}
      </div>
    </div>
  );
};

const Zoom = ({clip, idx, i, names}: {clip: Clip; idx: Index; i: number; names: Record<string, string>}) => {
  const {fps} = useVideoConfig();
  const z = clip.zoom!;
  const start = 25;
  const end = clip.frames - 10;
  if (i < start) return null;
  // Follow the track; hold its last known box when it is missing for a frame.
  let box: Track | undefined;
  for (let k = i; k >= 0 && !box; k--) box = idx.byId[k].get(z.id);
  if (!box) return null;
  const s = W / clip.width;
  const size = 360;
  const mag = 14; // frame px -> inset px
  const cx = (box[0] + box[2]) / 2;
  const cy = (box[1] + box[3]) / 2;
  const open = spring({frame: i - start, fps, config: {damping: 16}}) *
    interpolate(i, [end, clip.frames - 1], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const left = 60;
  const top = 330;
  const color = CLASS_COLOR[names[box[5]]] ?? C.accent;
  const held = idx.age[i].get(z.id) ?? 0;
  return (
    <>
      <svg width={W} height={H} style={{position: 'absolute', opacity: open}}>
        <circle cx={cx * s} cy={cy * s} r={26} fill="none" stroke={C.text} strokeWidth={2.5} />
        <line x1={cx * s - 26} y1={cy * s} x2={left + size} y2={top + size / 2} stroke={C.text} strokeWidth={2} strokeDasharray="6 6" />
      </svg>
      <div style={{
        position: 'absolute', left, top, width: size, height: size, borderRadius: 18, overflow: 'hidden',
        border: `3px solid ${C.text}`, transform: `scale(${open})`, transformOrigin: 'right center',
        boxShadow: '0 20px 60px rgba(0,0,0,0.6)',
      }}>
        <Img src={framePath(clip, i)} style={{
          position: 'absolute', width: clip.width * mag, height: clip.height * mag,
          left: size / 2 - cx * mag, top: size / 2 - cy * mag, maxWidth: 'none',
        }} />
        <div style={{
          position: 'absolute', left: size / 2 - ((box[2] - box[0]) * mag) / 2 - 4,
          top: size / 2 - ((box[3] - box[1]) * mag) / 2 - 4,
          width: (box[2] - box[0]) * mag + 8, height: (box[3] - box[1]) * mag + 8,
          border: `3px solid ${color}`,
        }} />
        <div style={{
          position: 'absolute', left: 0, right: 0, bottom: 0, padding: '10px 14px',
          background: 'rgba(0,0,0,0.7)', fontFamily: mono, fontSize: 17, lineHeight: 1.4,
        }}>
          <div style={{color}}>{names[box[5]]} #{z.id} · {z.w.toFixed(0)} × {z.h.toFixed(0)} px</div>
          <div style={{color: C.dim}}>tracked for {held} frames · zoom ×{mag}</div>
        </div>
      </div>
    </>
  );
};

const ClipView = ({data, clip, n}: {data: DemoData; clip: Clip; n: number}) => {
  const i = useCurrentFrame();
  const {fps} = useVideoConfig();
  const idx = useMemo(() => indexClip(clip), [clip]);
  const cap = spring({frame: i - 4, fps, config: {damping: 200}});
  const flash = n > 0 ? interpolate(i, [0, 7], [0.55, 0], {extrapolateRight: 'clamp'}) : 0;
  const capOut = interpolate(i, [70, 85], [1, 0.0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill>
      <Img src={framePath(clip, i)} style={{width: W, height: H}} />
      <AbsoluteFill style={{background: 'linear-gradient(90deg, rgba(0,0,0,0.45), rgba(0,0,0,0) 35%, rgba(0,0,0,0) 65%, rgba(0,0,0,0.45))'}} />
      <Overlay clip={clip} idx={idx} i={i} names={data.classes} />
      {clip.zoom && <Zoom clip={clip} idx={idx} i={i} names={data.classes} />}
      <Hud data={data} clip={clip} idx={idx} i={i} />
      <div style={{
        position: 'absolute', left: 60, top: 48, opacity: cap * (0.35 + 0.65 * capOut),
        transform: `translateY(${(1 - cap) * -20}px)`,
      }}>
        <div style={{fontFamily: mono, fontSize: 20, color: C.accent}}>
          {n + 1} / {data.clips.length} · {data.split} · never seen by the detector
        </div>
        <div style={{fontSize: 56, fontWeight: 800, marginTop: 4, textShadow: '0 4px 24px rgba(0,0,0,0.7)'}}>
          {clip.caption}
        </div>
      </div>
      <div style={{
        position: 'absolute', left: 60, bottom: 40, fontFamily: mono, fontSize: 18, color: C.text,
        padding: '10px 16px', borderRadius: 10, background: C.panel, border: `1px solid ${C.line}`,
      }}>
        {data.model} · {data.device} · {data.versions} · batch 1 · played at {data.fps} FPS
      </div>
      <AbsoluteFill style={{background: '#fff', opacity: flash}} />
    </AbsoluteFill>
  );
};

export const Footage = ({data}: {data: DemoData}) => (
  <AbsoluteFill>
    {data.clips.map((clip, n) => (
      <Sequence key={clip.sequence} from={n * T.clip} durationInFrames={T.clip}>
        <ClipView data={data} clip={clip} n={n} />
      </Sequence>
    ))}
  </AbsoluteFill>
);
