export class CharTokenizer {
  private charToIdx: Map<string, number> = new Map();
  private idxToChar: string[] = [];
  public vocabSize: number = 0;

  fit(text: string): void {
    const chars = Array.from(new Set(text.split(""))).sort();
    this.idxToChar = chars;
    this.charToIdx = new Map(chars.map((c, i) => [c, i]));
    this.vocabSize = chars.length;
  }

  encode(text: string): number[] {
    return text.split("").map((c) => this.charToIdx.get(c) ?? 0);
  }

  decode(indices: number[]): string {
    return indices.map((i) => this.idxToChar[i] ?? "").join("");
  }

  getVocab(): string[] {
    return [...this.idxToChar];
  }

  setVocab(vocab: string[]): void {
    this.idxToChar = [...vocab];
    this.charToIdx = new Map(vocab.map((c, i) => [c, i]));
    this.vocabSize = vocab.length;
  }
}
