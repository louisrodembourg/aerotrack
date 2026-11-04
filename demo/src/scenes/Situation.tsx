import {AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, CLASS_COLOR, mono} from '../theme';
import type {DemoData} from '../types';

// A schematic top-down street map: what a drone sees, before the real footage.
const ROADS_H = [330, 760];
const ROADS_V = [560, 1180, 1640];
const ROAD = 84;

type Mover = {id: number; cls: string; axis: 'h' | 'v'; lane: number; speed: number; offset: number};

// Deterministic traffic: lane = road centre +/- half a lane, speed in px per frame.
const MOVERS: Mover[] = [
  ...[0, 1, 2, 3, 4, 5].map((i) => ({
    id: 3 + i * 7, cls: i % 4 === 3 ? 'truck' : i % 3 === 2 ? 'van' : 'car', axis: 'h' as const,
    lane: ROADS_H[i % 2] + (i % 2 ? 20 : -20), speed: (i % 2 ? -1 : 1) * (5 + (i % 3)), offset: i * 410,
  })),
  ...[0, 1, 2, 3, 4].map((i) => ({
    id: 51 + i * 5, cls: i === 2 ? 'bus' : 'car', axis: 'v' as const,
    lane: ROADS_V[i % 3] + (i % 2 ? 20 : -20), speed: (i % 2 ? -1 : 1) * (4 + (i % 2)), offset: i * 290,
  })),
  ...Array.from({length: 14}, (_, i) => ({
    id: 80 + i, cls: 'pedestrian', axis: (i % 2 ? 'h' : 'v') as 'h' | 'v',
    lane: i % 2 ? ROADS_H[i % 2] + (i % 4 < 2 ? -56 : 56) : ROADS_V[i % 3] + (i % 4 < 2 ? -56 : 56),
    speed: (i % 3 ? 1 : -1) * 1.1, offset: i * 157,
  })),
];

const position = (m: Mover, f: number) => {
  const span = m.axis === 'h' ? 2200 : 1400;
  const s = (((m.offset + m.speed * f) % span) + span) % span - 140;
  return m.axis === 'h' ? {x: s, y: m.lane} : {x: m.lane, y: s};
};

const size = (cls: string) =>
  cls === 'pedestrian' ? [9, 9] : cls === 'bus' ? [30, 74] : cls === 'truck' ? [28, 62] : [22, 40];

const Drone = ({x, y, frame}: {x: number; y: number; frame: number}) => (
  <g transform={`translate(${x} ${y})`}>
    {[[-46, -46], [46, -46], [-46, 46], [46, 46]].map(([dx, dy], i) => (
      <g key={i} transform={`translate(${dx} ${dy})`}>
        <circle r={30} fill="none" stroke={C.text} strokeOpacity={0.25} strokeWidth={2} />
        <line x1={-26} x2={26} stroke={C.text} strokeWidth={3}
          transform={`rotate(${(frame * 47 + i * 45) % 360})`} />
      </g>
    ))}
    <line x1={-46} y1={-46} x2={46} y2={46} stroke={C.text} strokeWidth={8} />
    <line x1={46} y1={-46} x2={-46} y2={46} stroke={C.text} strokeWidth={8} />
    <rect x={-22} y={-22} width={44} height={44} rx={10} fill={C.bg} stroke={C.accent} strokeWidth={4} />
    <circle r={6} fill={C.accent} />
  </g>
);

export const Situation = ({data}: {data: DemoData}) => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();
  const budgetMs = 1000 / data.headline.tracking.budget_fps;

  const fly = interpolate(frame, [0, 45], [0, 1], {extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
  const dx = interpolate(fly, [0, 1], [-200, 1180]);
  const dy = 700; // over the crossing of the lower road and the middle avenue
  const fp = {w: 720, h: 405}; // camera footprint, 16:9 like the footage
  const fpIn = spring({frame: frame - 38, fps, config: {damping: 200}});
  // Last second: dive into the footprint, which becomes the real footage.
  const dive = interpolate(frame, [durationInFrames - 26, durationInFrames], [0, 1], {
    extrapolateLeft: 'clamp', easing: Easing.in(Easing.cubic),
  });
  const scale = 1 + dive * (1920 / fp.w - 1);
  const fade = interpolate(frame, [0, 8], [0, 1], {extrapolateRight: 'clamp'});

  const lines = [
    'A drone films the street from above.',
    'Objects are a few pixels wide, dense, and moving.',
    `Each frame: detect, then track, in under ${budgetMs.toFixed(0)} ms.`,
  ];

  return (
    <AbsoluteFill style={{opacity: fade}}>
      <svg width={1920} height={1080} style={{
        transform: `scale(${scale})`, transformOrigin: `${dx}px ${dy}px`,
      }}>
        <rect width={1920} height={1080} fill="#0C1219" />
        {/* city blocks */}
        {[0, 1, 2].flatMap((r) =>
          [0, 1, 2, 3].map((c) => {
            const xs = [-200, ...ROADS_V, 2100];
            const ys = [-200, ...ROADS_H, 1300];
            const x = xs[c] + ROAD / 2 + 36;
            const y = ys[r] + ROAD / 2 + 36;
            return (
              <rect key={`${r}-${c}`} x={x} y={y} rx={14}
                width={xs[c + 1] - xs[c] - ROAD - 72} height={ys[r + 1] - ys[r] - ROAD - 72}
                fill={(r + c) % 3 === 0 ? '#16212C' : '#121B24'} stroke="#1D2A37" strokeWidth={2} />
            );
          }),
        )}
        {ROADS_H.map((y) => (
          <g key={`h${y}`}>
            <rect x={0} y={y - ROAD / 2} width={1920} height={ROAD} fill="#1A232D" />
            <line x1={0} x2={1920} y1={y} y2={y} stroke="#3A4654" strokeWidth={2} strokeDasharray="24 20" />
          </g>
        ))}
        {ROADS_V.map((x) => (
          <g key={`v${x}`}>
            <rect x={x - ROAD / 2} y={0} width={ROAD} height={1080} fill="#1A232D" />
            <line y1={0} y2={1080} x1={x} x2={x} stroke="#3A4654" strokeWidth={2} strokeDasharray="24 20" />
          </g>
        ))}
        {MOVERS.map((m) => {
          const {x, y} = position(m, frame);
          const [w, h] = m.axis === 'h' ? [...size(m.cls)].reverse() : size(m.cls);
          const inside = Math.abs(x - dx) < fp.w / 2 && Math.abs(y - dy) < fp.h / 2;
          const color = CLASS_COLOR[m.cls];
          return (
            <g key={m.id}>
              {m.cls === 'pedestrian'
                ? <circle cx={x} cy={y} r={4.5} fill="#C9D3DD" />
                : <rect x={x - w / 2} y={y - h / 2} width={w} height={h} rx={5} fill="#C9D3DD" />}
              {inside && fpIn > 0.5 && (
                <g opacity={fpIn}>
                  <rect x={x - w / 2 - 6} y={y - h / 2 - 6} width={w + 12} height={h + 12}
                    fill="none" stroke={color} strokeWidth={2.5} />
                  <text x={x - w / 2 - 6} y={y - h / 2 - 11} fill={color} fontSize={15}
                    fontFamily={mono} fontWeight={600}>{m.id}</text>
                </g>
              )}
            </g>
          );
        })}
        {/* camera footprint */}
        <g opacity={fpIn}>
          <rect x={dx - fp.w / 2} y={dy - fp.h / 2} width={fp.w} height={fp.h}
            fill={C.accent} fillOpacity={0.06} stroke={C.accent} strokeWidth={3} strokeDasharray="14 10" />
          <text x={dx - fp.w / 2 + 12} y={dy - fp.h / 2 + 30} fill={C.accent} fontSize={20}
            fontFamily={mono} opacity={1 - dive}>CAMERA FOOTPRINT</text>
        </g>
        <g opacity={1 - dive}>
          <Drone x={dx} y={dy} frame={frame} />
        </g>
      </svg>
      <div style={{
        position: 'absolute', left: 80, bottom: 80, padding: '28px 36px', borderRadius: 18,
        background: C.panel, border: `1px solid ${C.line}`, opacity: 1 - dive * 1.5,
      }}>
        {lines.map((l, i) => {
          const s = spring({frame: frame - 10 - i * 22, fps, config: {damping: 200}});
          return (
            <div key={l} style={{
              fontSize: 36, fontWeight: i === 2 ? 700 : 500, color: i === 2 ? C.accent : C.text,
              opacity: s, transform: `translateX(${(1 - s) * -30}px)`, marginTop: i ? 12 : 0,
            }}>{l}</div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
