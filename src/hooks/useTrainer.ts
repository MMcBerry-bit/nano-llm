import { useState, useRef, useCallback } from "react";
import { NanoLLMTrainer, TrainingConfig, TrainingState, SavedState } from "../lib/trainer";

const STORAGE_KEY = "nano-llm-model-v1";

export interface UseTrainerReturn {
  isReady: boolean;
  isTraining: boolean;
  isPrepared: boolean;
  trainingState: TrainingState;
  vocabSize: number;
  tokenCount: number;
  generatedText: string;
  hasSavedState: boolean;
  prepare: (text: string, config: TrainingConfig) => Promise<void>;
  startTraining: () => Promise<void>;
  stopTraining: () => void;
  generate: (prompt: string, maxNewTokens: number, temperature: number) => string;
  reset: () => void;
  restoreLastSession: () => Promise<void>;
  saveToFile: () => void;
  loadFromFile: (file: File) => Promise<void>;
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
  const [hasSavedState, setHasSavedState] = useState(() => !!localStorage.getItem(STORAGE_KEY));

  const applyLoadedState = useCallback((result: {
    vocabSize: number;
    tokenCount: number;
    step: number;
    losses: { step: number; loss: number }[];
  }) => {
    setVocabSize(result.vocabSize);
    setTokenCount(result.tokenCount);
    setTrainingState({
      step: result.step,
      loss: result.losses[result.losses.length - 1]?.loss ?? 0,
      isTraining: false,
      losses: result.losses,
    });
    setIsReady(true);
    setIsPrepared(true);
    setGeneratedText("");
  }, []);

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
    try {
      const state = trainerRef.current.getState();
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      setHasSavedState(true);
    } catch (_) {}
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

  const restoreLastSession = useCallback(async () => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const state = JSON.parse(raw) as SavedState;
    const result = await trainerRef.current.loadState(state);
    applyLoadedState(result);
  }, [applyLoadedState]);

  const saveToFile = useCallback(() => {
    try {
      const state = trainerRef.current.getState();
      const blob = new Blob([JSON.stringify(state)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "nano-llm-model.json";
      a.click();
      URL.revokeObjectURL(url);
    } catch (_) {}
  }, []);

  const loadFromFile = useCallback(async (file: File) => {
    const state = JSON.parse(await file.text()) as SavedState;
    const result = await trainerRef.current.loadState(state);
    applyLoadedState(result);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      setHasSavedState(true);
    } catch (_) {}
  }, [applyLoadedState]);

  return {
    isReady,
    isTraining,
    isPrepared,
    trainingState,
    vocabSize,
    tokenCount,
    generatedText,
    hasSavedState,
    prepare,
    startTraining,
    stopTraining,
    generate,
    reset,
    restoreLastSession,
    saveToFile,
    loadFromFile,
  };
}
