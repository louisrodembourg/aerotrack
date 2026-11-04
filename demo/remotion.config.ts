import {existsSync} from 'node:fs';
import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');
Config.setJpegQuality(92);

// The chrome-headless-shell Remotion downloads aborts at start-up on macOS 26
// (icu_util.cc: Check failed); the installed Google Chrome renders the same pages.
const SYSTEM_CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
if (existsSync(SYSTEM_CHROME)) {
  Config.setBrowserExecutable(SYSTEM_CHROME);
}
