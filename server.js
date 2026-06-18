// ── LOCAL DEV SERVER ──────────────────────────────────────────────────────────
// This file is ONLY for running on your own computer.
// On Vercel, the api/vision.js serverless function handles everything instead.
// Run with:  npm run dev
// ─────────────────────────────────────────────────────────────────────────────
import 'dotenv/config';
import express from 'express';
import cors from 'cors';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '10mb' }));

// Serve the whole project as static files so you can open it in browser
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
const __dirname = dirname(fileURLToPath(import.meta.url));
app.use(express.static(__dirname));

app.get('/', (_req, res) => res.sendFile(join(__dirname, 'index.html')));

// The same proxy logic as api/vision.js
app.post('/api/vision', async (req, res) => {
  const token = process.env.GITHUB_TOKEN;
  if (!token) return res.status(500).json({ error: 'GITHUB_TOKEN not set. Check your .env file.' });

  const { mode, imageBase64, mediaType = 'image/jpeg' } = req.body;
  if (!mode || !imageBase64) return res.status(400).json({ error: 'Need mode and imageBase64.' });
  if (!['scene', 'ocr'].includes(mode)) return res.status(400).json({ error: 'mode must be scene or ocr.' });

  const textPrompt = mode === 'scene'
    ? 'You are an assistive AI. Look at this camera snapshot and in 2-3 natural sentences, describe the overall scene and note any prominent objects, obstacles, or vehicles.'
    : 'You are a highly accurate OCR scanner. Read all text in the image exactly as it appears, preserving line breaks. No filler or explanations. If there is no readable text output EXACTLY: NO_TEXT_FOUND';

  try {
    const upstream = await fetch('https://models.inference.ai.azure.com/chat/completions', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        ...(mode === 'ocr' && { temperature: 0.1 }),
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: textPrompt },
            { type: 'image_url', image_url: { url: `data:${mediaType};base64,${imageBase64}` } }
          ]
        }]
      })
    });

    const data = await upstream.json();
    if (!upstream.ok) return res.status(upstream.status).json({ error: data?.error?.message ?? 'Upstream error' });
    return res.json({ result: data?.choices?.[0]?.message?.content ?? '' });

  } catch (err) {
    console.error('Proxy error:', err);
    return res.status(500).json({ error: 'Proxy failed: ' + err.message });
  }
});

app.listen(PORT, () => {
  console.log(`✅ Virtual Eyes AI running at http://localhost:${PORT}`);
  console.log(`   Homepage  → http://localhost:${PORT}/index.html`);
  console.log(`   Detection → http://localhost:${PORT}/Main_detection/html.html`);
});
