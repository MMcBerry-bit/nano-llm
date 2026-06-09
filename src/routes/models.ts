import { Router, type IRouter } from "express";

const router: IRouter = Router();

const KV_URL = process.env.REPLIT_DB_URL;

function makeCode(): string {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  return Array.from({ length: 6 }, () => chars[Math.floor(Math.random() * chars.length)]).join("");
}

async function kvSet(key: string, value: string): Promise<void> {
  if (!KV_URL) throw new Error("REPLIT_DB_URL not set");
  const body = `${encodeURIComponent(key)}=${encodeURIComponent(value)}`;
  await fetch(KV_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
}

async function kvGet(key: string): Promise<string | null> {
  if (!KV_URL) throw new Error("REPLIT_DB_URL not set");
  const resp = await fetch(`${KV_URL}/${encodeURIComponent(key)}`);
  if (resp.status === 404) return null;
  return resp.text();
}

router.post("/models", async (req, res) => {
  try {
    const code = makeCode();
    await kvSet(`nano-llm:${code}`, JSON.stringify(req.body));
    res.json({ code });
  } catch (err) {
    res.status(500).json({ error: "Failed to save model" });
  }
});

router.get("/models/:code", async (req, res) => {
  try {
    const raw = await kvGet(`nano-llm:${req.params.code.toUpperCase()}`);
    if (!raw) return res.status(404).json({ error: "Code not found" });
    res.json(JSON.parse(raw));
  } catch (err) {
    res.status(500).json({ error: "Failed to load model" });
  }
});

export default router;
