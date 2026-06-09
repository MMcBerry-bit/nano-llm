import * as tf from "@tensorflow/tfjs";

export interface ModelConfig {
  vocabSize: number;
  contextLength: number;
  embeddingDim: number;
  numHeads: number;
  numLayers: number;
  dropoutRate?: number;
}

function randn(shape: number[], scale = 0.02): tf.Variable {
  return tf.variable(tf.randomNormal(shape, 0, scale));
}

function zeros(shape: number[]): tf.Variable {
  return tf.variable(tf.zeros(shape));
}

function ones(shape: number[]): tf.Variable {
  return tf.variable(tf.ones(shape));
}

function float32ToBase64(arr: Float32Array): string {
  // TF.js dataSync() can return a Float32Array that is a view into a larger
  // backing buffer (byteOffset > 0). Using arr.buffer directly would encode
  // the whole underlying ArrayBuffer — extra bytes included — causing shape
  // mismatches on restore. Slice to exactly the bytes owned by this view.
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function base64ToFloat32(b64: string): Float32Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

interface LNWeights { gamma: tf.Variable; beta: tf.Variable }
interface AttnWeights { qkv: tf.Variable; proj: tf.Variable; projBias: tf.Variable }
interface FFNWeights { w1: tf.Variable; b1: tf.Variable; w2: tf.Variable; b2: tf.Variable }
interface BlockWeights { ln1: LNWeights; attn: AttnWeights; ln2: LNWeights; ffn: FFNWeights }

export class NanoTransformer {
  private tokEmb: tf.Variable;
  private posEmb: tf.Variable;
  private blocks: BlockWeights[];
  private lnFinal: LNWeights;
  private lmHead: tf.Variable;
  public readonly config: ModelConfig;

  constructor(config: ModelConfig) {
    this.config = config;
    const { vocabSize, contextLength, embeddingDim, numLayers } = config;

    this.tokEmb = randn([vocabSize, embeddingDim]);
    this.posEmb = randn([contextLength, embeddingDim]);

    this.blocks = Array.from({ length: numLayers }, () => ({
      ln1: { gamma: ones([embeddingDim]), beta: zeros([embeddingDim]) },
      attn: {
        qkv: randn([embeddingDim, 3 * embeddingDim]),
        proj: randn([embeddingDim, embeddingDim]),
        projBias: zeros([embeddingDim]),
      },
      ln2: { gamma: ones([embeddingDim]), beta: zeros([embeddingDim]) },
      ffn: {
        w1: randn([embeddingDim, 4 * embeddingDim]),
        b1: zeros([4 * embeddingDim]),
        w2: randn([4 * embeddingDim, embeddingDim]),
        b2: zeros([embeddingDim]),
      },
    }));

    this.lnFinal = { gamma: ones([embeddingDim]), beta: zeros([embeddingDim]) };
    this.lmHead = randn([embeddingDim, vocabSize]);
  }

  private layerNorm(x: tf.Tensor, ln: LNWeights): tf.Tensor {
    const mean = x.mean(-1, true);
    const std = x.sub(mean).square().mean(-1, true).add(1e-5).sqrt();
    return x.sub(mean).div(std).mul(ln.gamma).add(ln.beta);
  }

  private gelu(x: tf.Tensor): tf.Tensor {
    return x.mul(tf.sigmoid(x.mul(1.702)));
  }

  private mat2d(x: tf.Tensor, w: tf.Variable): tf.Tensor {
    const batchSeq = x.shape.slice(0, -1).reduce((a, b) => a * (b as number), 1);
    const inDim = x.shape[x.shape.length - 1] as number;
    const outDim = w.shape[1] as number;
    const origShape = x.shape.slice(0, -1) as number[];
    return x.reshape([batchSeq, inDim]).matMul(w).reshape([...origShape, outDim]);
  }

  private selfAttention(x: tf.Tensor, attn: AttnWeights): tf.Tensor {
    const { numHeads, embeddingDim } = this.config;
    const headDim = Math.floor(embeddingDim / numHeads);
    const scale = Math.sqrt(headDim);
    const seqLen = x.shape[1] as number;
    const batchSize = x.shape[0] as number;

    const qkv = this.mat2d(x, attn.qkv);
    const q = qkv.slice([0, 0, 0], [-1, -1, embeddingDim]);
    const k = qkv.slice([0, 0, embeddingDim], [-1, -1, embeddingDim]);
    const v = qkv.slice([0, 0, 2 * embeddingDim], [-1, -1, embeddingDim]);

    const toHeads = (t: tf.Tensor) =>
      t.reshape([batchSize, seqLen, numHeads, headDim]).transpose([0, 2, 1, 3]);

    const qH = toHeads(q);
    const kH = toHeads(k);
    const vH = toHeads(v);

    let scores = qH.matMul(kH, false, true).div(scale);

    const lowerTri = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0);
    const causalMask = tf.where(
      lowerTri.equal(tf.scalar(0)),
      tf.fill([seqLen, seqLen], -1e4),
      tf.zeros([seqLen, seqLen])
    );
    scores = scores.add(causalMask);
    const attnW = tf.softmax(scores, -1);

    const out = attnW.matMul(vH);
    const outFlat = out.transpose([0, 2, 1, 3]).reshape([batchSize, seqLen, embeddingDim]);
    return this.mat2d(outFlat, attn.proj).add(attn.projBias);
  }

  forward(tokens: tf.Tensor2D): tf.Tensor3D {
    const { embeddingDim } = this.config;
    const batchSize = tokens.shape[0] as number;
    const seqLen = tokens.shape[1] as number;

    const flatTokens = tokens.reshape([-1]) as tf.Tensor1D;
    const tokEmbFlat = tf.gather(this.tokEmb, flatTokens);
    const tokEmbOut = tokEmbFlat.reshape([batchSize, seqLen, embeddingDim]);

    const posIdx = tf.range(0, seqLen, 1, "int32");
    const posEmbOut = tf.gather(this.posEmb, posIdx);

    let x = tokEmbOut.add(posEmbOut);

    for (const block of this.blocks) {
      const n1 = this.layerNorm(x, block.ln1);
      x = x.add(this.selfAttention(n1, block.attn));

      const n2 = this.layerNorm(x, block.ln2);
      const ff = this.gelu(this.mat2d(n2, block.ffn.w1).add(block.ffn.b1));
      x = x.add(this.mat2d(ff, block.ffn.w2).add(block.ffn.b2));
    }

    x = this.layerNorm(x, this.lnFinal);
    return this.mat2d(x, this.lmHead) as tf.Tensor3D;
  }

  get trainableVariables(): tf.Variable[] {
    const vars: tf.Variable[] = [this.tokEmb, this.posEmb];
    for (const b of this.blocks) {
      vars.push(b.ln1.gamma, b.ln1.beta);
      vars.push(b.attn.qkv, b.attn.proj, b.attn.projBias);
      vars.push(b.ln2.gamma, b.ln2.beta);
      vars.push(b.ffn.w1, b.ffn.b1, b.ffn.w2, b.ffn.b2);
    }
    vars.push(this.lnFinal.gamma, this.lnFinal.beta);
    vars.push(this.lmHead);
    return vars;
  }

  getWeights(): Array<{ b64: string; shape: number[] }> {
    return this.trainableVariables.map((v) => ({
      b64: float32ToBase64(v.dataSync() as Float32Array),
      shape: v.shape as number[],
    }));
  }

  setWeights(weights: Array<{ b64: string; shape: number[] }>): void {
    const vars = this.trainableVariables;
    if (vars.length !== weights.length) throw new Error("Weight count mismatch");
    vars.forEach((v, i) => {
      const data = base64ToFloat32(weights[i].b64);
      const tensor = tf.tensor(data, weights[i].shape as tf.Shape);
      v.assign(tensor);
      tensor.dispose();
    });
  }

  dispose(): void {
    this.trainableVariables.forEach((v) => v.dispose());
  }
}
