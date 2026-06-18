import 'dotenv/config';

export default async function handler(req, res) {
  // ── CORS ──────────────────────────────────────────────────────────────────
  res.setHeader('Access-Control-Allow-Origin', process.env.ALLOWED_ORIGIN || '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  // ── Token lives here on the server — never sent to the browser ────────────
  const token = process.env.GITHUB_TOKEN;
  if (!token) return res.status(500).json({ error: 'GITHUB_TOKEN not set on server.' });

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
}
