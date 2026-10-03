// Render index.html frame-by-frame with headless Chromium and encode to MP4.
//   node src/render.js                      -> out/startale-pieverse-motion.mp4
//   node src/render.js --preview 1,3.5,7    -> out/preview/t*.jpg
// Env: FPS (render rate, default 60; blended down to 30 for motion blur), WORKERS (default 4).
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const { spawnSync } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
const pi = args.indexOf('--preview');
const preview = pi >= 0 ? args[pi + 1].split(',').map(Number) : null;
const FPS = Number(process.env.FPS) || 60;
const WORKERS = Number(process.env.WORKERS) || 4;
const OUT = path.join(ROOT, 'out/startale-pieverse-motion.mp4');
const FRAMES = path.join(ROOT, 'out/frames');

const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg' };
const server = http.createServer((req, res) => {
  const p = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
  fs.readFile(p, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': types[path.extname(p)] || 'application/octet-stream' });
    res.end(data);
  });
}).listen(0);

async function openPage(browser) {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(`http://localhost:${server.address().port}/index.html`);
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 60000 });
  return page;
}

async function shoot(page, t, file) {
  await page.evaluate((t) => window.renderAt(t), t);
  await (await page.$('#stage')).screenshot({ path: file, type: 'jpeg', quality: 95 });
}

(async () => {
  if (preview) {
    const browser = await chromium.launch();
    const page = await openPage(browser);
    fs.mkdirSync(path.join(ROOT, 'out/preview'), { recursive: true });
    for (const t of preview) await shoot(page, t, path.join(ROOT, `out/preview/t${t.toFixed(2)}.jpg`));
    await browser.close();
    return server.close();
  }

  fs.rmSync(FRAMES, { recursive: true, force: true });
  fs.mkdirSync(FRAMES, { recursive: true });
  const probe = await chromium.launch();
  const duration = await (await openPage(probe)).evaluate(() => DURATION);
  await probe.close();
  const n = Math.round(duration * FPS);
  let next = 0, done = 0;
  const t0 = Date.now();

  await Promise.all(Array.from({ length: WORKERS }, async () => {
    const browser = await chromium.launch();
    const page = await openPage(browser);
    while (next < n) {
      const i = next++;
      await shoot(page, i / FPS, path.join(FRAMES, `f${String(i).padStart(5, '0')}.jpg`));
      if (++done % 50 === 0) console.log(`${done}/${n} frames  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    }
    await browser.close();
  }));
  server.close();

  // blend pairs of frames for motion blur, then fade out to white
  const blend = FPS >= 60 ? `tmix=frames=${Math.round(FPS / 30)},fps=30,` : '';
  const r = spawnSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(FPS), '-i', path.join(FRAMES, 'f%05d.jpg'),
    '-vf', `${blend}format=yuv420p`,
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-movflags', '+faststart', OUT], { stdio: 'inherit' });
  if (r.status !== 0) process.exit(r.status);
  console.log('wrote', path.relative(ROOT, OUT), `in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
})();
