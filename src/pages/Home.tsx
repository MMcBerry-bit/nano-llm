import React, { useState, useMemo, useRef } from "react";
import { useTrainer } from "../hooks/useTrainer";
import { SAMPLE_PYTHON_CODE, SAMPLE_JS_CODE } from "../lib/sampleCode";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Badge } from "@/components/ui/badge";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import {
  Play, Square, Settings2, Database, Zap, Loader2, RotateCcw,
  CheckCircle2, Download, Upload, History, Share2, Hash,
  ChevronDown, ChevronUp, X, Copy, Check,
} from "lucide-react";

export default function Home() {
  const {
    isReady, isTraining, isPrepared, trainingState,
    vocabSize, tokenCount, generatedText, hasSavedState,
    prepare, startTraining, stopTraining, generate, reset,
    restoreLastSession, saveToFile, loadFromFile,
    shareOnline, loadByCode,
  } = useTrainer();

  const [trainingText, setTrainingText] = useState("");
  const [config, setConfig] = useState({
    contextLength: 128, embeddingDim: 64, numHeads: 4,
    numLayers: 3, batchSize: 8, learningRate: 0.001,
  });
  const [isPreparing, setIsPreparing] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(100);
  const [temperature, setTemperature] = useState(0.8);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [shareCode, setShareCode] = useState("");
  const [isSharing, setIsSharing] = useState(false);
  const [shareError, setShareError] = useState("");
  const [codeCopied, setCodeCopied] = useState(false);

  const [loadCode, setLoadCode] = useState("");
  const [isLoadingCode, setIsLoadingCode] = useState(false);
  const [loadError, setLoadError] = useState("");

  const [inspectorOpen, setInspectorOpen] = useState(false);

  const handleConfigChange = (key: keyof typeof config, value: number) => {
    setConfig((prev) => {
      const next = { ...prev, [key]: value };
      if (key === "numHeads" || key === "embeddingDim") {
        if (next.embeddingDim % next.numHeads !== 0)
          next.embeddingDim = Math.ceil(next.embeddingDim / next.numHeads) * next.numHeads;
      }
      return next;
    });
  };

  const handlePrepare = async () => {
    if (!trainingText) return;
    setIsPreparing(true);
    try { await prepare(trainingText, config); } finally { setIsPreparing(false); }
  };

  const handleGenerate = () => {
    if (!isReady || !prompt) return;
    generate(prompt, maxNewTokens, temperature);
  };

  const handleRestore = async () => {
    setIsRestoring(true);
    try { await restoreLastSession(); } finally { setIsRestoring(false); }
  };

  const handleFileLoad = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";
    await loadFromFile(file);
  };

  const handleShare = async () => {
    setIsSharing(true);
    setShareCode("");
    setShareError("");
    try {
      const code = await shareOnline();
      setShareCode(code);
    } catch {
      setShareError("Upload failed — check connection");
    } finally {
      setIsSharing(false);
    }
  };

  const handleLoadCode = async () => {
    if (!loadCode.trim()) return;
    setIsLoadingCode(true);
    setLoadError("");
    try {
      await loadByCode(loadCode);
      setLoadCode("");
    } catch {
      setLoadError("Code not found");
    } finally {
      setIsLoadingCode(false);
    }
  };

  const copyCode = () => {
    navigator.clipboard.writeText(shareCode);
    setCodeCopied(true);
    setTimeout(() => setCodeCopied(false), 2000);
  };

  const chartData = useMemo(() => {
    const losses = trainingState.losses;
    if (losses.length <= 500) return losses;
    const step = Math.ceil(losses.length / 500);
    return losses.filter((_, i) => i % step === 0);
  }, [trainingState.losses]);

  const bestLoss = useMemo(() =>
    trainingState.losses.length
      ? Math.min(...trainingState.losses.map((l) => l.loss))
      : null,
    [trainingState.losses]
  );

  return (
    <div className="min-h-screen w-full bg-background text-foreground p-6 font-sans">
      <div className="max-w-7xl mx-auto space-y-6">

        {/* HEADER */}
        <header className="flex items-center justify-between border-b border-border pb-6">
          <div className="flex items-center gap-3">
            <div className="bg-primary/10 text-primary p-2 rounded-xl">
              <Zap className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">Nano LLM</h1>
              <p className="text-sm text-muted-foreground">In-browser transformer trainer</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {hasSavedState && !isPrepared && (
              <Button variant="outline" size="sm" onClick={handleRestore} disabled={isRestoring}
                className="rounded-lg text-primary border-primary/30 hover:bg-primary/5">
                {isRestoring ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <History className="w-4 h-4 mr-2" />}
                Restore last session
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={saveToFile}
              disabled={!isReady || trainingState.step === 0 || isTraining}
              className="rounded-lg text-muted-foreground hover:text-foreground" title="Download model as file">
              <Download className="w-4 h-4 mr-2" /> Save file
            </Button>
            <Button variant="ghost" size="sm" onClick={() => fileInputRef.current?.click()}
              disabled={isTraining} className="rounded-lg text-muted-foreground hover:text-foreground">
              <Upload className="w-4 h-4 mr-2" /> Load file
            </Button>
            <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={handleFileLoad} />
            <Button variant="ghost" size="sm" onClick={reset} disabled={isTraining}
              className="rounded-lg text-muted-foreground hover:text-foreground">
              <RotateCcw className="w-4 h-4 mr-2" /> Reset
            </Button>
          </div>
        </header>

        {/* SHARE / LOAD ONLINE BAR */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Share online */}
          <Card className="bg-card border-border rounded-2xl shadow-sm">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <Share2 className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium">Share online</span>
                <span className="text-xs text-muted-foreground ml-auto">Upload model → get a code</span>
              </div>
              <div className="flex gap-2">
                <Button size="sm" onClick={handleShare}
                  disabled={!isReady || trainingState.step === 0 || isTraining || isSharing}
                  className="rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 h-9 px-4 text-sm">
                  {isSharing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Share2 className="w-3.5 h-3.5 mr-1.5" />}
                  {isSharing ? "Uploading…" : "Share"}
                </Button>
                {shareCode && (
                  <div className="flex items-center gap-2 flex-1 bg-primary/5 border border-primary/20 rounded-lg px-3">
                    <span className="font-mono font-bold text-primary tracking-widest text-sm">{shareCode}</span>
                    <Button variant="ghost" size="sm" onClick={copyCode} className="h-7 w-7 p-0 ml-auto">
                      {codeCopied ? <Check className="w-3.5 h-3.5 text-green-600" /> : <Copy className="w-3.5 h-3.5" />}
                    </Button>
                  </div>
                )}
                {shareError && <span className="text-xs text-red-500 flex items-center">{shareError}</span>}
              </div>
            </CardContent>
          </Card>

          {/* Load by code */}
          <Card className="bg-card border-border rounded-2xl shadow-sm">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <Hash className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium">Load by code</span>
                <span className="text-xs text-muted-foreground ml-auto">Enter a share code to load remotely</span>
              </div>
              <div className="flex gap-2">
                <Input
                  value={loadCode}
                  onChange={(e) => { setLoadCode(e.target.value.toUpperCase()); setLoadError(""); }}
                  onKeyDown={(e) => e.key === "Enter" && handleLoadCode()}
                  placeholder="ABC123"
                  maxLength={6}
                  className="font-mono tracking-widest text-center h-9 rounded-lg border-border bg-transparent shadow-sm text-sm uppercase"
                  disabled={isLoadingCode || isTraining}
                />
                <Button size="sm" onClick={handleLoadCode}
                  disabled={!loadCode.trim() || isLoadingCode || isTraining}
                  className="rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 h-9 px-4 text-sm">
                  {isLoadingCode ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : "Load"}
                </Button>
                {loadError && <span className="text-xs text-red-500 flex items-center">{loadError}</span>}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* MODEL INSPECTOR */}
        {isPrepared && (
          <Card className="bg-card border-border rounded-2xl shadow-sm">
            <button
              onClick={() => setInspectorOpen((o) => !o)}
              className="w-full flex items-center justify-between px-5 py-4 text-sm font-medium hover:bg-muted/30 rounded-2xl transition-colors"
            >
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-primary" />
                <span>Model Inspector</span>
                {trainingState.step > 0 && (
                  <Badge variant="secondary" className="bg-primary/10 text-primary rounded-full text-xs font-normal">
                    {trainingState.step.toLocaleString()} steps · loss {trainingState.loss.toFixed(4)}
                  </Badge>
                )}
              </div>
              {inspectorOpen ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
            </button>

            {inspectorOpen && (
              <CardContent className="pt-0 pb-5 px-5 space-y-5">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {[
                    { label: "Context Length", value: config.contextLength },
                    { label: "Embedding Dim", value: config.embeddingDim },
                    { label: "Attention Heads", value: config.numHeads },
                    { label: "Layers", value: config.numLayers },
                    { label: "Vocab Size", value: vocabSize },
                    { label: "Token Count", value: tokenCount.toLocaleString() },
                    { label: "Total Steps", value: trainingState.step.toLocaleString() },
                    { label: "Best Loss", value: bestLoss != null ? bestLoss.toFixed(4) : "---" },
                  ].map(({ label, value }) => (
                    <div key={label} className="bg-muted/30 border border-border/50 rounded-xl p-3">
                      <div className="text-xs text-muted-foreground mb-1">{label}</div>
                      <div className="text-sm font-semibold">{value}</div>
                    </div>
                  ))}
                </div>

                <div>
                  <div className="text-xs text-muted-foreground mb-2">Vocabulary ({vocabSize} characters)</div>
                  <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                    {/* Show vocab from trainingText since we have it */}
                    {Array.from(new Set(trainingText.split(""))).sort().map((ch) => (
                      <span key={ch} className="font-mono text-xs bg-muted/50 border border-border/50 rounded px-1.5 py-0.5">
                        {ch === " " ? "·" : ch === "\n" ? "↵" : ch === "\t" ? "→" : ch}
                      </span>
                    ))}
                  </div>
                </div>

                {trainingState.losses.length > 0 && (
                  <div>
                    <div className="text-xs text-muted-foreground mb-2">Loss over last 20 steps</div>
                    <div className="overflow-x-auto">
                      <table className="text-xs w-full">
                        <tbody>
                          <tr>
                            {trainingState.losses.slice(-20).map(({ step, loss }) => (
                              <td key={step} className="text-center px-1 py-1 border-r border-border/30 last:border-0">
                                <div className="text-muted-foreground">{step.toLocaleString()}</div>
                                <div className="font-mono text-primary">{loss.toFixed(3)}</div>
                              </td>
                            ))}
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </CardContent>
            )}
          </Card>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* LEFT COLUMN */}
          <div className="lg:col-span-4 space-y-6">
            <Card className="bg-card border-border rounded-2xl shadow-sm">
              <CardHeader className="pb-4">
                <CardTitle className="text-base font-medium flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary" /> Training Data
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" className="w-full rounded-full text-sm font-medium"
                    onClick={() => setTrainingText(SAMPLE_PYTHON_CODE)} disabled={isTraining || isPrepared}>Python</Button>
                  <Button variant="outline" size="sm" className="w-full rounded-full text-sm font-medium"
                    onClick={() => setTrainingText(SAMPLE_JS_CODE)} disabled={isTraining || isPrepared}>JS</Button>
                </div>
                <Textarea value={trainingText} onChange={(e) => setTrainingText(e.target.value)}
                  placeholder="Paste raw text for training..."
                  className="h-40 bg-transparent resize-none rounded-xl border-border focus-visible:ring-1 focus-visible:ring-primary shadow-sm text-sm"
                  disabled={isTraining || isPrepared} />
                <div className="flex justify-between items-center text-xs text-muted-foreground">
                  <Badge variant="secondary" className="font-normal text-muted-foreground bg-muted/50 hover:bg-muted/50 rounded-md">
                    {trainingText.length} chars
                  </Badge>
                  {isPrepared && (
                    <div className="flex gap-2">
                      <span className="bg-muted/50 px-2 py-1 rounded-md">{tokenCount} tokens</span>
                      <span className="bg-muted/50 px-2 py-1 rounded-md">{vocabSize} vocab</span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card border-border rounded-2xl shadow-sm">
              <CardHeader className="pb-4">
                <CardTitle className="text-base font-medium flex items-center gap-2">
                  <Settings2 className="w-4 h-4 text-primary" /> Model Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <Label className="text-sm font-normal text-muted-foreground">Context Length</Label>
                    <Badge variant="secondary" className="bg-primary/10 text-primary hover:bg-primary/20 rounded-full font-medium">{config.contextLength}</Badge>
                  </div>
                  <Slider disabled={isTraining || isPrepared} min={32} max={256} step={8}
                    value={[config.contextLength]} onValueChange={([v]) => handleConfigChange("contextLength", v)}
                    className="[&_[role=slider]]:border-primary [&_[role=slider]]:bg-primary [&_.bg-primary]:bg-primary" />
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <Label className="text-sm font-normal text-muted-foreground">Embedding Dim</Label>
                    <Badge variant="secondary" className="bg-primary/10 text-primary hover:bg-primary/20 rounded-full font-medium">{config.embeddingDim}</Badge>
                  </div>
                  <Slider disabled={isTraining || isPrepared} min={32} max={256} step={8}
                    value={[config.embeddingDim]} onValueChange={([v]) => handleConfigChange("embeddingDim", v)}
                    className="[&_[role=slider]]:border-primary [&_[role=slider]]:bg-primary [&_.bg-primary]:bg-primary" />
                </div>
                <div className="grid grid-cols-2 gap-4 pt-2">
                  <div className="space-y-2">
                    <Label className="text-sm font-normal text-muted-foreground">Heads</Label>
                    <ToggleGroup type="single" value={config.numHeads.toString()}
                      onValueChange={(v) => v && handleConfigChange("numHeads", parseInt(v))}
                      disabled={isTraining || isPrepared} className="justify-start gap-1">
                      {["1", "2", "4", "8"].map(val => (
                        <ToggleGroupItem key={val} value={val}
                          className="flex-1 rounded-lg h-9 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground border border-transparent data-[state=off]:border-border data-[state=off]:hover:bg-muted">
                          {val}
                        </ToggleGroupItem>
                      ))}
                    </ToggleGroup>
                  </div>
                  <div className="space-y-2">
                    <Label className="text-sm font-normal text-muted-foreground">Layers</Label>
                    <Input type="number" disabled={isTraining || isPrepared} min={1} max={6}
                      value={config.numLayers} onChange={(e) => handleConfigChange("numLayers", parseInt(e.target.value) || 1)}
                      className="h-9 rounded-lg border-border bg-transparent shadow-sm" />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-sm font-normal text-muted-foreground">Batch Size</Label>
                  <Input type="number" disabled={isTraining || isPrepared} min={4} max={64}
                    value={config.batchSize} onChange={(e) => handleConfigChange("batchSize", parseInt(e.target.value) || 4)}
                    className="h-9 rounded-lg border-border bg-transparent shadow-sm" />
                </div>
                <div className="space-y-2">
                  <Label className="text-sm font-normal text-muted-foreground">Learning Rate</Label>
                  <Input type="number" disabled={isTraining || isPrepared} min={0.0001} max={0.01} step={0.0001}
                    value={config.learningRate} onChange={(e) => handleConfigChange("learningRate", parseFloat(e.target.value) || 0.001)}
                    className="h-9 rounded-lg border-border bg-transparent shadow-sm" />
                </div>
                <Button className="w-full rounded-xl h-11 font-medium bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm"
                  onClick={handlePrepare} disabled={isTraining || isPrepared || !trainingText || isPreparing}>
                  {isPreparing ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Preparing...</>
                    : isPrepared ? <><CheckCircle2 className="w-4 h-4 mr-2" />Model Ready</>
                    : "Build Model"}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* RIGHT COLUMN */}
          <div className="lg:col-span-8 space-y-6">
            <Card className="bg-card border-border rounded-2xl shadow-sm flex flex-col min-h-[420px]">
              <CardHeader className="pb-4 flex-row items-center justify-between space-y-0 border-b border-border/50">
                <CardTitle className="text-base font-medium flex items-center gap-2">
                  <Play className="w-4 h-4 text-primary" /> Training
                </CardTitle>
                <div className="flex items-center gap-3">
                  {isTraining ? (
                    <Badge className="bg-primary/10 text-primary hover:bg-primary/10 font-normal rounded-full gap-1.5 pl-1.5">
                      <span className="w-2 h-2 rounded-full bg-primary animate-pulse" /> Training
                    </Badge>
                  ) : trainingState.step > 0 ? (
                    <Badge variant="outline" className="border-green-200 text-green-700 bg-green-50/50 font-normal rounded-full">Stopped</Badge>
                  ) : (
                    <Badge variant="secondary" className="font-normal rounded-full bg-muted text-muted-foreground">Idle</Badge>
                  )}
                  {!isTraining ? (
                    <Button size="sm" onClick={startTraining} disabled={!isReady}
                      className="h-9 rounded-full bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm px-4">
                      <Play className="w-4 h-4 mr-1.5" /> {trainingState.step > 0 ? "Resume" : "Start"}
                    </Button>
                  ) : (
                    <Button size="sm" onClick={stopTraining} variant="outline"
                      className="h-9 rounded-full border-red-200 text-red-600 hover:bg-red-50 hover:text-red-700 shadow-sm px-4">
                      <Square className="w-4 h-4 mr-1.5 fill-current" /> Stop
                    </Button>
                  )}
                </div>
              </CardHeader>
              <CardContent className="pt-6 flex-1 flex flex-col">
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="bg-muted/30 border border-border/50 rounded-xl p-4 flex flex-col">
                    <span className="text-muted-foreground text-sm font-medium mb-1">Step</span>
                    <span className="text-2xl font-semibold tracking-tight">{trainingState.step.toLocaleString()}</span>
                  </div>
                  <div className="bg-muted/30 border border-border/50 rounded-xl p-4 flex flex-col">
                    <span className="text-muted-foreground text-sm font-medium mb-1">Loss</span>
                    <span className="text-2xl font-semibold text-primary tracking-tight">
                      {trainingState.loss > 0 ? trainingState.loss.toFixed(4) : "---"}
                    </span>
                  </div>
                </div>

                {isTraining && (
                  <div className="mb-6 space-y-2">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Running</span>
                      <span className="animate-pulse">∞ infinite steps</span>
                    </div>
                    <div className="h-2 rounded-full bg-muted/50 overflow-hidden">
                      <div className="h-full bg-primary rounded-full"
                        style={{ width: "40%", animation: "progressSlide 1.5s ease-in-out infinite" }} />
                    </div>
                  </div>
                )}

                <div className="flex-1 bg-background border border-border/50 rounded-xl min-h-[200px] p-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" opacity={0.5} />
                      <XAxis dataKey="step" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickMargin={10} />
                      <YAxis domain={["auto", "auto"]} stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickMargin={10} />
                      <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontSize: "12px" }}
                        itemStyle={{ color: "hsl(var(--primary))", fontWeight: 500 }}
                        labelStyle={{ color: "hsl(var(--muted-foreground))", marginBottom: "4px" }} />
                      <Line type="monotone" dataKey="loss" stroke="hsl(var(--primary))" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card border-border rounded-2xl shadow-sm">
              <CardHeader className="pb-4">
                <CardTitle className="text-base font-medium flex items-center gap-2">
                  <Zap className="w-4 h-4 text-primary" /> Generate
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex gap-4 items-end">
                  <div className="flex-1 space-y-2">
                    <Label className="text-sm font-normal text-muted-foreground">Prompt</Label>
                    <Input value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="def "
                      className="bg-transparent border-border rounded-xl focus-visible:ring-primary h-11 shadow-sm font-mono text-sm"
                      disabled={!isReady || trainingState.step === 0} />
                  </div>
                  <Button onClick={handleGenerate} disabled={!isReady || trainingState.step === 0 || !prompt}
                    className="h-11 rounded-xl font-medium px-6 bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm">
                    Generate
                  </Button>
                </div>
                <div className="grid grid-cols-2 gap-8 px-1">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <Label className="text-sm font-normal text-muted-foreground">Max Tokens</Label>
                      <Badge variant="secondary" className="bg-primary/10 text-primary hover:bg-primary/20 rounded-full font-medium">{maxNewTokens}</Badge>
                    </div>
                    <Slider disabled={!isReady || trainingState.step === 0} min={10} max={500} step={10}
                      value={[maxNewTokens]} onValueChange={([v]) => setMaxNewTokens(v)}
                      className="[&_[role=slider]]:border-primary [&_[role=slider]]:bg-primary [&_.bg-primary]:bg-primary" />
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <Label className="text-sm font-normal text-muted-foreground">Temperature</Label>
                      <Badge variant="secondary" className="bg-primary/10 text-primary hover:bg-primary/20 rounded-full font-medium">{temperature.toFixed(2)}</Badge>
                    </div>
                    <Slider disabled={!isReady || trainingState.step === 0} min={0.1} max={2.0} step={0.1}
                      value={[temperature]} onValueChange={([v]) => setTemperature(v)}
                      className="[&_[role=slider]]:border-primary [&_[role=slider]]:bg-primary [&_.bg-primary]:bg-primary" />
                  </div>
                </div>
                <div className="rounded-xl overflow-hidden border border-border shadow-sm">
                  <div className="bg-muted/50 px-4 py-2 border-b border-border">
                    <span className="text-xs font-medium text-muted-foreground">Output</span>
                  </div>
                  <div className="w-full h-48 bg-background p-4 overflow-y-auto font-mono text-sm whitespace-pre-wrap">
                    {generatedText ? (
                      <span>
                        <span className="text-muted-foreground">{prompt}</span>
                        <span>{generatedText.slice(prompt.length)}</span>
                      </span>
                    ) : (
                      <span className="text-muted-foreground/40 italic">Generated code will appear here...</span>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
