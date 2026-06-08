import * as tf from "@tensorflow/tfjs";
import { CharTokenizer } from "./tokenizer";

export interface TrainingConfig {
  contextLength: number;
  embeddingDim: number;
  numHeads: number;
  numLayers: number;
  batchSize: number;
  learningRate: number;
  maxSteps: number;
}

export interface TrainingState {
  step: number;
  loss: number;
  isTraining: boolean;
  losses: { step: number; loss: number }[];
}

export type TrainingCallback = (state: TrainingState) => void;

export class NanoLLMTrainer {
  private tokenizer: CharTokenizer;
  private model: tf.LayersModel | null = null;
  private optimizer: tf.Optimizer | null = null;
  private tokens: number[] = [];
  private config: TrainingConfig | null = null;
  private shouldStop = false;
  private losses: { step: number; loss: number }[] = [];

  constructor() {
    this.tokenizer = new CharTokenizer();
  }

  prepare(text: string, config: TrainingConfig): { vocabSize: number; tokenCount: number } {
    this.config = config;
    this.tokenizer.fit(text);
    this.tokens = this.tokenizer.encode(text);
    return {
      vocabSize: this.tokenizer.vocabSize,
      tokenCount: this.tokens.length,
    };
  }

  async build(): Promise<tf.LayersModel> {
    if (!this.config) throw new Error("Call prepare() first");
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.losses = [];

    const { vocabSize, contextLength, embeddingDim, numHeads, numLayers } =
      this.config as TrainingConfig & { vocabSize: number };

    const vSize = this.tokenizer.vocabSize;
    const { buildModel } = await import("./model");
    this.model = buildModel({
      vocabSize: vSize,
      contextLength,
      embeddingDim,
      numHeads,
      numLayers,
      dropoutRate: 0.1,
    });

    this.optimizer = tf.train.adam(this.config.learningRate);
    return this.model;
  }

  private getBatch(): { xTokens: tf.Tensor; xPos: tf.Tensor; yTokens: tf.Tensor } {
    const { contextLength, batchSize } = this.config!;
    const maxStart = this.tokens.length - contextLength - 1;

    const starts = Array.from({ length: batchSize }, () =>
      Math.floor(Math.random() * maxStart)
    );

    const xData: number[][] = starts.map((s) =>
      this.tokens.slice(s, s + contextLength)
    );
    const yData: number[][] = starts.map((s) =>
      this.tokens.slice(s + 1, s + contextLength + 1)
    );

    const posData = Array.from({ length: contextLength }, (_, i) => i);
    const xPos = tf.tile(
      tf.tensor2d([posData], [1, contextLength], "int32"),
      [batchSize, 1]
    );

    return {
      xTokens: tf.tensor2d(xData, [batchSize, contextLength], "int32"),
      xPos,
      yTokens: tf.tensor2d(yData, [batchSize, contextLength], "int32"),
    };
  }

  async train(onUpdate: TrainingCallback): Promise<void> {
    if (!this.model || !this.optimizer || !this.config) {
      throw new Error("Call build() before train()");
    }
    this.shouldStop = false;

    const { maxSteps, contextLength, batchSize } = this.config;

    for (let step = 0; step < maxSteps; step++) {
      if (this.shouldStop) break;

      const { xTokens, xPos, yTokens } = this.getBatch();

      const lossValue = await tf.tidy(() => {
        const lossFn = () => {
          const logits = this.model!.apply([xTokens, xPos]) as tf.Tensor;
          const logitsFlat = tf.reshape(logits, [-1, this.tokenizer.vocabSize]);
          const labelsFlat = tf.reshape(yTokens, [-1]);
          return tf.losses.softmaxCrossEntropy(
            tf.oneHot(labelsFlat, this.tokenizer.vocabSize),
            logitsFlat
          ) as tf.Scalar;
        };
        const { value, grads } = tf.variableGrads(lossFn as () => tf.Scalar);
        (this.optimizer as tf.Optimizer).applyGradients(grads);
        return value;
      });

      const loss = (await lossValue.data())[0];
      lossValue.dispose();
      xTokens.dispose();
      xPos.dispose();
      yTokens.dispose();

      const entry = { step: step + 1, loss };
      this.losses.push(entry);

      onUpdate({
        step: step + 1,
        loss,
        isTraining: !this.shouldStop,
        losses: [...this.losses],
      });

      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }

    onUpdate({
      step: this.losses.length,
      loss: this.losses[this.losses.length - 1]?.loss ?? 0,
      isTraining: false,
      losses: [...this.losses],
    });
  }

  stop(): void {
    this.shouldStop = true;
  }

  generate(prompt: string, maxNewTokens: number, temperature: number): string {
    if (!this.model || !this.config) return "";
    const { contextLength } = this.config;

    let context = this.tokenizer.encode(prompt);
    const generated: number[] = [];

    for (let i = 0; i < maxNewTokens; i++) {
      const ctx = context.slice(-contextLength);
      while (ctx.length < contextLength) ctx.unshift(0);

      const xTokens = tf.tensor2d([ctx], [1, contextLength], "int32");
      const xPos = tf.tensor2d(
        [Array.from({ length: contextLength }, (_, j) => j)],
        [1, contextLength],
        "int32"
      );

      const logits = this.model.predict([xTokens, xPos]) as tf.Tensor;
      const lastLogits = tf.squeeze(logits.slice([0, contextLength - 1, 0], [1, 1, -1]), [0, 1]);
      const scaled = lastLogits.div(temperature);
      const probs = tf.softmax(scaled);
      const probsData = Array.from(probs.dataSync());

      xTokens.dispose();
      xPos.dispose();
      logits.dispose();
      lastLogits.dispose();
      scaled.dispose();
      probs.dispose();

      const sampled = sampleFromDistribution(probsData);
      context.push(sampled);
      generated.push(sampled);
    }

    return this.tokenizer.decode(generated);
  }

  getVocabSize(): number {
    return this.tokenizer.vocabSize;
  }

  getLosses(): { step: number; loss: number }[] {
    return [...this.losses];
  }
}

function sampleFromDistribution(probs: number[]): number {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) return i;
  }
  return probs.length - 1;
}
