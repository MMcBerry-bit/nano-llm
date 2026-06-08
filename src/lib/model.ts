import * as tf from "@tensorflow/tfjs";

export interface ModelConfig {
  vocabSize: number;
  contextLength: number;
  embeddingDim: number;
  numHeads: number;
  numLayers: number;
  dropoutRate: number;
}

function causalSelfAttention(
  x: tf.Tensor,
  embeddingDim: number,
  numHeads: number,
  training: boolean
): tf.Tensor {
  const headDim = Math.floor(embeddingDim / numHeads);
  const seqLen = x.shape[1] as number;

  const qkvDense = tf.layers.dense({ units: embeddingDim * 3, useBias: false });
  const qkv = qkvDense.apply(x) as tf.Tensor;

  const [q, k, v] = tf.split(qkv, 3, -1);

  function reshapeHeads(t: tf.Tensor): tf.Tensor {
    return tf.transpose(
      tf.reshape(t, [-1, seqLen, numHeads, headDim]),
      [0, 2, 1, 3]
    );
  }

  const qh = reshapeHeads(q);
  const kh = reshapeHeads(k);
  const vh = reshapeHeads(v);

  const scale = Math.sqrt(headDim);
  let attn = tf.matMul(qh, kh, false, true).div(scale);

  const mask = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0);
  const negInf = tf.fill([seqLen, seqLen], -1e9);
  const causalMask = tf.where(mask.equal(tf.scalar(1)), tf.zeros([seqLen, seqLen]), negInf);
  attn = attn.add(causalMask.expandDims(0).expandDims(0));

  attn = tf.softmax(attn, -1);

  const out = tf.matMul(attn, vh);
  const outT = tf.transpose(out, [0, 2, 1, 3]);
  const outR = tf.reshape(outT, [-1, seqLen, embeddingDim]);

  const projDense = tf.layers.dense({ units: embeddingDim });
  return projDense.apply(outR) as tf.Tensor;
}

export function buildModel(config: ModelConfig): tf.LayersModel {
  const {
    vocabSize,
    contextLength,
    embeddingDim,
    numHeads,
    numLayers,
  } = config;

  const inputTokens = tf.input({ shape: [contextLength], dtype: "int32" });
  const inputPos = tf.input({ shape: [contextLength], dtype: "int32" });

  const tokEmb = tf.layers
    .embedding({ inputDim: vocabSize, outputDim: embeddingDim, name: "tok_emb" })
    .apply(inputTokens) as tf.Tensor;

  const posEmb = tf.layers
    .embedding({ inputDim: contextLength, outputDim: embeddingDim, name: "pos_emb" })
    .apply(inputPos) as tf.Tensor;

  let x = tf.layers.add().apply([tokEmb, posEmb]) as tf.Tensor;

  for (let i = 0; i < numLayers; i++) {
    const ln1 = tf.layers.layerNormalization({ name: `ln1_${i}` }).apply(x) as tf.Tensor;
    const attnOut = tf.layers
      .multiHeadAttention({ numHeads, keyDim: Math.floor(embeddingDim / numHeads), name: `mha_${i}` })
      .apply([ln1, ln1]) as tf.Tensor;
    x = tf.layers.add().apply([x, attnOut]) as tf.Tensor;

    const ln2 = tf.layers.layerNormalization({ name: `ln2_${i}` }).apply(x) as tf.Tensor;
    let ff = tf.layers.dense({ units: embeddingDim * 4, activation: "gelu", name: `ff1_${i}` }).apply(ln2) as tf.Tensor;
    ff = tf.layers.dense({ units: embeddingDim, name: `ff2_${i}` }).apply(ff) as tf.Tensor;
    x = tf.layers.add().apply([x, ff]) as tf.Tensor;
  }

  x = tf.layers.layerNormalization({ name: "ln_final" }).apply(x) as tf.Tensor;
  const logits = tf.layers
    .dense({ units: vocabSize, name: "lm_head" })
    .apply(x) as tf.Tensor;

  const model = tf.model({ inputs: [inputTokens, inputPos], outputs: logits });
  return model;
}
