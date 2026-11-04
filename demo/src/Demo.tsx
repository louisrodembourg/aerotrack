import {AbsoluteFill, Sequence} from 'remotion';
import {Footage} from './scenes/Footage';
import {Outro} from './scenes/Outro';
import {Results} from './scenes/Results';
import {Situation} from './scenes/Situation';
import {Title} from './scenes/Title';
import {C, sans, T} from './theme';
import type {DemoData} from './types';

export const Demo = ({data}: {data: DemoData | null}) => {
  if (!data) return null;
  const footage = data.clips.length * T.clip;
  let t = 0;
  const at = (d: number) => {
    const from = t;
    t += d;
    return from;
  };
  return (
    <AbsoluteFill style={{background: C.bg, fontFamily: sans, color: C.text}}>
      <Sequence from={at(T.title)} durationInFrames={T.title}>
        <Title data={data} />
      </Sequence>
      <Sequence from={at(T.situation)} durationInFrames={T.situation}>
        <Situation data={data} />
      </Sequence>
      <Sequence from={at(footage)} durationInFrames={footage}>
        <Footage data={data} />
      </Sequence>
      <Sequence from={at(T.results)} durationInFrames={T.results}>
        <Results data={data} />
      </Sequence>
      <Sequence from={at(T.outro)} durationInFrames={T.outro}>
        <Outro data={data} />
      </Sequence>
    </AbsoluteFill>
  );
};
