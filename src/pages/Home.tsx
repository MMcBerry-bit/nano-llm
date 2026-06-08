import React, { useState, useEffect, useMemo, useRef } from "react";
import { useTrainer } from "../hooks/useTrainer";
import { SAMPLE_PYTHON_CODE, SAMPLE_JS_CODE } from "../lib/sampleCode";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Play, Square, Settings2, Database, Zap, Loader2, RotateCcw, CheckCircle2 } from "lucide-react";

export default function Home() {
  const {
    isReady,
    isTraining,
    isPrepared,
    trainingState,
    vocabSize,
    tokenCount,
    generatedText,
    prepare,
    startTraining,
    stopTraining,
    generate,
    reset,
  } = useTrainer();

  // Data Section State
  const [trainingText, setTrainingText] = useState("");

  // Model Config State
  const [config, setConfig] = useState({
    contextLength: 128,
    embeddingDim: 64,
    numHeads: 4,
    numLayers: 3,
    batchSize: 8,
    learningRate: 0.001,
    maxSteps: 3000,
  });
  
  const [isPreparing, setIsPreparing] = useState(false);

  // Generation Section State
  const [prompt, setPrompt] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(100);
  const [temperature, setTemperature] = useState(0.8);

  // Enforce embeddingDim divisible by numHeads
  const handleConfigChange = (key: keyof typeof config, value: number) => {
    setConfig((prev) => {
      const next = { ...prev, [key]: value };
      if (key === "numHeads" || key === "embeddingDim") {
        if (next.embeddingDim % next.numHeads !== 0) {
          next.embeddingDim = Math.ceil(next.embeddingDim / next.numHeads) * next.numHeads;
        }
      }
      return next;
    });
  };

  const handlePrepare = async () => {
    if (!trainingText) return;
    setIsPreparing(true);
    try {
      await prepare(trainingText, config);
    } finally {
      setIsPreparing(false);
    }
  };

  const handleGenerate = () => {
    if (!isReady || !prompt) return;
    generate(prompt, maxNewTokens, temperature);
  };

  return (
    <div className="min-h-screen w-full bg-background text-foreground p-6 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">
        <header className="flex items-center justify-between border-b border-border pb-6">
          <div className="flex items-center gap-3">
            <div className="bg-primary/10 text-primary p-2 rounded-xl">
              <Zap className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-foreground">Nano LLM</h1>
              <p className="text-sm text-muted-foreground">In-browser transformer trainer</p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={reset} disabled={isTraining} className="rounded-lg text-muted-foreground hover:text-foreground">
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* LEFT COLUMN: Data & Config */}
          <div className="lg:col-span-4 space-y-6">
            {/* DATA SECTION */}
            <Card className="bg-card border-border rounded-2xl shadow-sm">
              <CardHeader className="pb-4">
                <CardTitle className="text-base font-medium flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary" /> Training Data
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" className="w-full rounded-full text-sm font-medium" onClick={() => setTrainingText(SAMPLE_PYTHON_CODE)} disabled={isTraining || isPrepared}>Python</Button>
                  <Button variant="outline" size="sm" className="w-full rounded-full text-sm font-medium" onClick={() => setTrainingText(SAMPLE_JS_CODE)} disabled={isTraining || isPrepared}>JS</Button>
                </div>
                <Textarea
                  value={trainingText}
                  onChange={(e) => setTrainingText(e.target.value)}
                  placeholder="Paste raw text for training..."
                  className="h-40 bg-transparent resize-none rounded-xl border-border focus-visible:ring-1 focus-visible:ring-primary shadow-sm text-sm"
                  disabled={isTraining || isPrepared}
                />
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

            {/* MODEL CONFIG SECTION */}
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
                  <Slider
                    disabled={isTraining || isPrepared}
                    min={32} max={256} step={8}
                    value={[config.contextLength]}
                    onValueChange={([v]) => handleConfigChange("contextLength", v)}
                    className="[&_[role=slider]]:border-primary [&_[role=slider]]:bg-primary [&_[role=slider]]:shadow-sm [&_.bg-primary]:bg-primary"
                  />
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <Label className="text-sm font-normal text-muted-foreground">Embedding Dim</Label>
                    <Badge variant="secondary" className="bg-primary/10 text-primary hover:bg-primary/20 rounded-full font-medium">{config.embeddingDim}</Badge>
                  </div>
                  <Slider
                    disabled={isTraining || isPrepared}
                    min={32} max={256} step={8}
                    value={[config.embeddingDim]}
                    onValueChange={([v]) => handleConfigChange("embeddingDim", v)}
                    className="[&_[role=slider]]:border-primary [&_[role=slider]]:bg-primary [&_[role=slider]]:shadow-sm [&_.bg-primary]:bg-primary"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4 pt-2">
                  <div className="space-y-2">
                    <Label className="text-sm font-normal text-muted-foreground">Heads</Label>
                    <ToggleGroup 
                      type="single" 
                      value={config.numHeads.toString()} 
                      onValueChange={(v) => v && handleConfigChange("numHeads", parseInt(v))}
                      disabled={isTraining || isPrepared}
                      className="justify-start gap-1"
                    >
                      {["1", "2", "4", "8"].map(val => (
                        <ToggleGroupItem 
                          key={val} 
                          value={val} 
                          className="flex-1 rounded-lg h-9 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground border border-transparent data-[state=off]:border-border data-[state=off]:hover:bg-muted"
                        >
                          {val}
                        </ToggleGroupItem>
                      ))}
                    </ToggleGroup>
                  </div>
                  <div className="space-y-2">
                    <Label className="text-sm font-normal text-muted-foreground">Layers</Label>
                    <Input
                      type="number"
                      disabled={isTraining || isPrepared}
                      min={1} max={6}
                      value={config.numLayers}
                      onChange={(e) => handleConfigChange("numLayers", parseInt(e.target.value) || 1)}
                      className="h-9 rounded-lg border-border bg-transparent shadow-sm"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label className="text-sm font-normal text-muted-foreground">Batch Size</Label>
                    <Input
                      type="number"
                      disabled={isTraining || isPrepared}
                      min={4} max={64}
                      value={config.batchSize}
                      onChange={(e) => handleConfigChange("batchSize", parseInt(e.target.value) || 4)}
                      className="h-9 rounded-lg border-border bg-transparent shadow-sm"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-sm font-normal text-muted-foreground">Max Steps</Label>
                    <Input
                      type="number"
                      disabled={isTraining || isPrepared}
                      min={50} max={2000} step={50}
                      value={config.maxSteps}
                      onChange={(e) => handleConfigChange("maxSteps", parseInt(e.target.value) || 50)}
                      className="h-9 rounded-lg border-border bg-transparent shadow-sm"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-normal text-muted-foreground">Learning Rate</Label>
                  <Input
                    type="number"
                    disabled={isTraining || isPrepared}
                    min={0.0001} max={0.01} step={0.0001}
                    value={config.learningRate}
                    onChange={(e) => handleConfigChange("learningRate", parseFloat(e.target.value) || 0.001)}
                    className="h-9 rounded-lg border-border bg-transparent shadow-sm"
                  />
                </div>

                <Button 
                  className="w-full rounded-xl h-11 font-medium bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm" 
                  onClick={handlePrepare}
                  disabled={isTraining || isPrepared || !trainingText || isPreparing}
                >
                  {isPreparing ? (
                    <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Preparing...</>
                  ) : isPrepared ? (
                    <><CheckCircle2 className="w-4 h-4 mr-2" /> Model Ready</>
                  ) : (
                    "Build Model"
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* RIGHT COLUMN: Training & Generation */}
          <div className="lg:col-span-8 space-y-6">
            
            {/* TRAINING SECTION */}
            <Card className="bg-card border-border rounded-2xl shadow-sm flex flex-col min-h-[420px]">
              <CardHeader className="pb-4 flex-row items-center justify-between space-y-0 border-b border-border/50">
                <CardTitle className="text-base font-medium flex items-center gap-2">
                  <Play className="w-4 h-4 text-primary" /> Training
                </CardTitle>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    {isTraining ? (
                      <Badge className="bg-primary/10 text-primary hover:bg-primary/10 font-normal rounded-full gap-1.5 pl-1.5">
                        <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                        Training
                      </Badge>
                    ) : trainingState.step >= config.maxSteps ? (
                      <Badge variant="outline" className="border-green-200 text-green-700 bg-green-50/50 font-normal rounded-full">
                        Complete
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="font-normal rounded-full bg-muted text-muted-foreground">
                        Idle
                      </Badge>
                    )}
                  </div>
                  {!isTraining ? (
                    <Button 
                      size="sm" 
                      onClick={startTraining} 
                      disabled={!isReady || trainingState.step >= config.maxSteps}
                      className="h-9 rounded-full bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm px-4"
                    >
                      <Play className="w-4 h-4 mr-1.5" /> Start
                    </Button>
                  ) : (
                    <Button 
                      size="sm" 
                      onClick={stopTraining} 
                      variant="outline"
                      className="h-9 rounded-full border-red-200 text-red-600 hover:bg-red-50 hover:text-red-700 shadow-sm px-4"
                    >
                      <Square className="w-4 h-4 mr-1.5 fill-current" /> Stop
                    </Button>
                  )}
                </div>
              </CardHeader>
              <CardContent className="pt-6 flex-1 flex flex-col">
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="bg-muted/30 border border-border/50 rounded-xl p-4 flex flex-col">
                    <span className="text-muted-foreground text-sm font-medium mb-1">Step</span>
                    <span className="text-2xl font-semibold text-foreground tracking-tight">{trainingState.step} <span className="text-muted-foreground text-lg font-normal">/ {config.maxSteps}</span></span>
                  </div>
                  <div className="bg-muted/30 border border-border/50 rounded-xl p-4 flex flex-col">
                    <span className="text-muted-foreground text-sm font-medium mb-1">Loss</span>
                    <span className="text-2xl font-semibold text-primary tracking-tight">{trainingState.loss > 0 ? trainingState.loss.toFixed(4) : "---"}</span>
                  </div>
                </div>

                <div className="mb-6 space-y-2">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Progress</span>
                    <span>{Math.round((trainingState.step / config.maxSteps) * 100)}%</span>
                  </div>
                  <Progress value={(trainingState.step / config.maxSteps) * 100} className="h-2 rounded-full bg-muted/50 [&>div]:bg-primary" />
                </div>

                <div className="flex-1 bg-background border border-border/50 rounded-xl min-h-[200px] p-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingState.losses} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" opacity={0.5} />
                      <XAxis 
                        dataKey="step" 
                        stroke="hsl(var(--muted-foreground))" 
                        fontSize={11} 
                        tickLine={false}
                        axisLine={false}
                        tickMargin={10}
                      />
                      <YAxis 
                        domain={['auto', 'auto']} 
                        stroke="hsl(var(--muted-foreground))" 
                        fontSize={11}
                        tickLine={false}
                        axisLine={false}
                        tickMargin={10}
                      />
                      <Tooltip 
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '12px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)' }}
                        itemStyle={{ color: 'hsl(var(--primary))', fontWeight: 500 }}
                        labelStyle={{ color: 'hsl(var(--muted-foreground))', marginBottom: '4px' }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="loss" 
                        stroke="hsl(var(--primary))" 
                        strokeWidth={2.5}
                        dot={false}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* GENERATION SECTION */}
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
                    <Input 
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder="def "
                      className="bg-transparent border-border rounded-xl focus-visible:ring-primary h-11 shadow-sm font-mono text-sm"
                      disabled={!isReady || trainingState.step === 0}
                    />
                  </div>
                  <Button 
                    onClick={handleGenerate}
                    disabled={!isReady || trainingState.step === 0 || !prompt}
                    className="h-11 rounded-xl font-medium px-6 bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm"
                  >
                    Generate
                  </Button>
                </div>

                <div className="grid grid-cols-2 gap-8 px-1">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <Label className="text-sm font-normal text-muted-foreground">Max Tokens</Label>
                      <Badge variant="secondary" className="bg-primary/10 text-primary hover:bg-primary/20 rounded-full font-medium">{maxNewTokens}</Badge>
                    </div>
                    <Slider
                      disabled={!isReady || trainingState.step === 0}
                      min={10} max={500} step={10}
                      value={[maxNewTokens]}
                      onValueChange={([v]) => setMaxNewTokens(v)}
                      className="[&_[role=slider]]:border-primary [&_[role=slider]]:bg-primary [&_[role=slider]]:shadow-sm [&_.bg-primary]:bg-primary"
                    />
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <Label className="text-sm font-normal text-muted-foreground">Temperature</Label>
                      <Badge variant="secondary" className="bg-primary/10 text-primary hover:bg-primary/20 rounded-full font-medium">{temperature.toFixed(2)}</Badge>
                    </div>
                    <Slider
                      disabled={!isReady || trainingState.step === 0}
                      min={0.1} max={2.0} step={0.1}
                      value={[temperature]}
                      onValueChange={([v]) => setTemperature(v)}
                      className="[&_[role=slider]]:border-primary [&_[role=slider]]:bg-primary [&_[role=slider]]:shadow-sm [&_.bg-primary]:bg-primary"
                    />
                  </div>
                </div>

                <div className="rounded-xl overflow-hidden border border-border shadow-sm">
                  <div className="bg-muted/50 px-4 py-2 border-b border-border flex items-center justify-between">
                    <span className="text-xs font-medium text-muted-foreground">Output</span>
                  </div>
                  <div className="w-full h-48 bg-background p-4 overflow-y-auto font-mono text-sm text-foreground whitespace-pre-wrap">
                    {generatedText ? (
                      <span>
                        <span className="text-muted-foreground">{prompt}</span>
                        <span className="text-foreground">{generatedText.slice(prompt.length)}</span>
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
