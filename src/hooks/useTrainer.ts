import { useState, useRef, useCallback } from "react";
import { NanoLLMTrainer, TrainingConfig, TrainingState } from "../lib/trainer";

export interface UseTrainerReturn {
  isReady: boolean;
  isTraining: boolean;
  isPrepared: boolean;
  trainingState: TrainingState;
  vocabSize: number;
  tokenCount: number;
  generatedText: string;
  prepare: (text: string, config: TrainingConfig) => Promise<void>;
  startTraining: () => Promise<void>;
  stopTraining: () => void;
  generate: (prompt: string, maxNewTokens: number, temperature: number) => string;
  reset: () => void;
}

const DEFAULT_STATE: TrainingState = {
  step: 0,
  loss: 0,
  isTraining: false,
  losses: [],
};

export function useTrainer(): UseTrainerReturn {
  const trainerRef = useRef<NanoLLMTrainer>(new NanoLLMTrainer());
  const [isReady, setIsReady] = useState(false);
  const [isPrepared, setIsPrepared] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingState, setTrainingState] = useState<TrainingState>(DEFAULT_STATE);
  const [vocabSize, setVocabSize] = useState(0);
  const [tokenCount, setTokenCount] = useState(0);
  const [generatedText, setGeneratedText] = useState("");

  const prepare = useCallback(async (text: string, config: TrainingConfig) => {
    const trainer = trainerRef.current;
    const result = trainer.prepare(text, config);
    setVocabSize(result.vocabSize);
    setTokenCount(result.tokenCount);
    setTrainingState(DEFAULT_STATE);
    setIsReady(false);
    setIsPrepared(false);
    setGeneratedText("");

    await trainer.build();
    setIsReady(true);
    setIsPrepared(true);
  }, []);

  const startTraining = useCallback(async () => {
    if (!isReady) return;
    setIsTraining(true);
    setTrainingState((s) => ({ ...s, isTraining: true }));

    await trainerRef.current.train((state) => {
      setTrainingState(state);
      if (!state.isTraining) setIsTraining(false);
    });

    setIsTraining(false);
  }, [isReady]);

  const stopTraining = useCallback(() => {
    trainerRef.current.stop();
    setIsTraining(false);
  }, []);

  const generate = useCallback(
    (prompt: string, maxNewTokens: number, temperature: number): string => {
      const text = trainerRef.current.generate(prompt, maxNewTokens, temperature);
      setGeneratedText(text);
      return text;
    },
    []
  );

  const reset = useCallback(() => {
    trainerRef.current.stop();
    trainerRef.current = new NanoLLMTrainer();
    setIsReady(false);
    setIsPrepared(false);
    setIsTraining(false);
    setTrainingState(DEFAULT_STATE);
    setVocabSize(0);
    setTokenCount(0);
    setGeneratedText("");
  }, []);

  return {
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
  };
}
