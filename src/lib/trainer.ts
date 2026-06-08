import * as tf from "@tensorflow/tfjs";
import { CharTokenizer } from "./tokenizer";
import { NanoTransformer } from "./model";

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
  private transformer: NanoTransformer | null = null;
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
    return { vocabSize: this.tokenizer.vocabSize, tokenCount: this.tokens.length };
  }

  async build(): Promise<void> {
    if (!this.config) throw new Error("Call prepare() first");

    if (this.transformer) {
      this.transformer.dispose();
      this.transformer = null;
    }
    if (this.optimizer) {
      this.optimizer.dispose?.();
      this.optimizer = null;
    }
    this.losses = [];

    const { contextLength, embeddingDim, numHeads, numLayers, learningRate } = this.config;

    this.transformer = new NanoTransformer({
      vocabSize: this.tokenizer.vocabSize,
      contextLength,
      embeddingDim,
      numHeads,
      numLayers,
    });

    this.optimizer = tf.train.adam(learningRate);

    await new Promise<void>((r) => setTimeout(r, 0));
  }

  private getBatch(): { xTokens: tf.Tensor2D; yTokens: tf.Tensor2D } {
    const { contextLength, batchSize } = this.config!;
    const maxStart = this.tokens.length - contextLength - 1;
    const starts = Array.from({ length: batchSize }, () =>
      Math.floor(Math.random() * maxStart)
    );
    return {
      xTokens: tf.tensor2d(
        starts.map((s) => this.tokens.slice(s, s + contextLength)),
        [batchSize, contextLength],
        "int32"
      ),
      yTokens: tf.tensor2d(
        starts.map((s) => this.tokens.slice(s + 1, s + contextLength + 1)),
        [batchSize, contextLength],
        "int32"
      ),
    };
  }

  async train(onUpdate: TrainingCallback): Promise<void> {
    if (!this.transformer || !this.optimizer || !this.config) {
      throw new Error("Call build() before train()");
    }
    this.shouldStop = false;
    const { maxSteps } = this.config;
    const variables = this.transformer.trainableVariables;

    for (let step = 0; step < maxSteps; step++) {
      if (this.shouldStop) break;

      const { xTokens, yTokens } = this.getBatch();
      const vocabSize = this.tokenizer.vocabSize;

      const lossFn = (): tf.Scalar => {
        const logits = this.transformer!.forward(xTokens);
        const logitsFlat = logits.reshape([-1, vocabSize]) as tf.Tensor2D;
        const labelsFlat = yTokens.reshape([-1]) as tf.Tensor1D;
        const labelsOneHot = tf.oneHot(labelsFlat, vocabSize);
        return tf.losses.softmaxCrossEntropy(labelsOneHot, logitsFlat) as tf.Scalar;
      };

      const { value, grads } = tf.variableGrads(lossFn, variables);

      const gradValues = Object.values(grads) as tf.Tensor[];
      const globalNorm = tf.sqrt(
        gradValues.map((g) => g.square().sum()).reduce((a, b) => a.add(b))
      );
      const clipNorm = 1.0;
      const scale = tf.minimum(tf.scalar(1.0), tf.scalar(clipNorm).div(globalNorm.add(1e-6)));
      const clippedGrads: typeof grads = {};
      for (const key of Object.keys(grads)) {
        clippedGrads[key] = (grads[key] as tf.Tensor).mul(scale);
      }
      globalNorm.dispose();
      scale.dispose();

      this.optimizer.applyGradients(clippedGrads);
      Object.values(clippedGrads).forEach((g) => (g as tf.Tensor).dispose());

      const loss = (await value.data())[0];
      value.dispose();
      Object.values(grads).forEach((g) => (g as tf.Tensor).dispose());
      xTokens.dispose();
      yTokens.dispose();

      const entry = { step: step + 1, loss };
      this.losses.push(entry);

      onUpdate({
        step: step + 1,
        loss,
        isTraining: !this.shouldStop,
        losses: [...this.losses],
      });

      await new Promise<void>((r) => setTimeout(r, 0));
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
    if (!this.transformer || !this.config) return "";
    const { contextLength } = this.config;
    const { vocabSize } = this.tokenizer;

    let context = this.tokenizer.encode(prompt);
    const generated: number[] = [];

    for (let i = 0; i < maxNewTokens; i++) {
      const ctx = context.slice(-contextLength);
      while (ctx.length < contextLength) ctx.unshift(0);

      const xTokens = tf.tensor2d([ctx], [1, contextLength], "int32");
      const logits = this.transformer.forward(xTokens);
      const lastLogits = logits
        .slice([0, contextLength - 1, 0], [1, 1, -1])
        .reshape([vocabSize]) as tf.Tensor1D;
      const probs = tf.softmax(lastLogits.div(temperature));
      const probsData = Array.from(probs.dataSync() as Float32Array);

      xTokens.dispose();
      logits.dispose();
      lastLogits.dispose();
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
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}
