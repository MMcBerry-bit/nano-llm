import { Router, type IRouter } from "express";

const router: IRouter = Router();
const KV_URL = process.env.REPLIT_DB_URL;

function makeCode(): string {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  return Array.from({ length: 6 }, () => chars[Math.floor(Math.random() * chars.length)]).join("");
}

async function kvSet(key: string, value: string): Promise<void> {
  if (!KV_URL) throw new Error("REPLIT_DB_URL not set");
  await fetch(KV_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `${encodeURIComponent(key)}=${encodeURIComponent(value)}`,
  });
}

async function kvGet(key: string): Promise<string | null> {
  if (!KV_URL) throw new Error("REPLIT_DB_URL not set");
  const resp = await fetch(`${KV_URL}/${encodeURIComponent(key)}`);
  if (resp.status === 404) return null;
  return resp.text();
}

async function kvList(prefix: string): Promise<string[]> {
  if (!KV_URL) throw new Error("REPLIT_DB_URL not set");
  const resp = await fetch(`${KV_URL}?prefix=${encodeURIComponent(prefix)}&encode=true`);
  const text = await resp.text();
  if (!text.trim()) return [];
  try {
    return (JSON.parse(text) as string[]).map(decodeURIComponent);
  } catch {
    return text.split("\n").map((k) => k.trim()).filter(Boolean).map(decodeURIComponent);
  }
}

async function kvDelete(key: string): Promise<void> {
  if (!KV_URL) throw new Error("REPLIT_DB_URL not set");
  await fetch(`${KV_URL}/${encodeURIComponent(key)}`, { method: "DELETE" });
}

// GET /api/models — list all model metadata
router.get("/models", async (_req, res) => {
  try {
    const keys = await kvList("nano-llm:meta:");
    const entries = await Promise.all(
      keys.map(async (key) => {
        const raw = await kvGet(key);
        return raw ? JSON.parse(raw) : null;
      })
    );
    const sorted = entries
      .filter(Boolean)
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
    res.json(sorted);
  } catch (err) {
    res.status(500).json({ error: "Failed to list models" });
  }
});

// POST /api/models — upload a model, store weights + metadata separately
router.post("/models", async (req, res) => {
  try {
    const code = makeCode();
    const { losses = [], config = {}, vocab = [] } = req.body as {
      losses: { step: number; loss: number }[];
      config: Record<string, number>;
      vocab: string[];
    };

    const lossValues = losses.map((l) => l.loss).filter((v) => isFinite(v));
    const meta = {
      code,
      createdAt: new Date().toISOString(),
      steps: losses.length,
      loss: lossValues.at(-1) ?? 0,
      bestLoss: lossValues.length ? Math.min(...lossValues) : 0,
      vocabSize: vocab.length,
      config: {
        contextLength: config.contextLength ?? 0,
        embeddingDim: config.embeddingDim ?? 0,
        numLayers: config.numLayers ?? 0,
        numHeads: config.numHeads ?? 0,
        learningRate: config.learningRate ?? 0,
      },
    };

    await kvSet(`nano-llm:${code}`, JSON.stringify(req.body));
    await kvSet(`nano-llm:meta:${code}`, JSON.stringify(meta));
    res.json({ code });
  } catch (err) {
    res.status(500).json({ error: "Failed to save model" });
  }
});

// GET /api/models/:code — download a specific model
router.get("/models/:code", async (req, res) => {
  try {
    const code = req.params.code.toUpperCase();
    const raw = await kvGet(`nano-llm:${code}`);
    if (!raw) return res.status(404).json({ error: "Code not found" });
    res.json(JSON.parse(raw));
  } catch (err) {
    res.status(500).json({ error: "Failed to load model" });
  }
});

// DELETE /api/models/:code — remove a model
router.delete("/models/:code", async (req, res) => {
  try {
    const code = req.params.code.toUpperCase();
    await kvDelete(`nano-llm:${code}`);
    await kvDelete(`nano-llm:meta:${code}`);
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: "Failed to delete model" });
  }
});

export default router;
