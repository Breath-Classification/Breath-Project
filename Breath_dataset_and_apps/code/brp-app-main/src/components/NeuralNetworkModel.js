import { NativeModules } from "react-native";
const { TFLiteModule, PytorchModule } = NativeModules;

/**
 * Funkcja uniwersalna do predykcji dla Tensometru
 * @param {Array} points - tablica punktów {y, x}
 * @param {"tf" | "torch"} engine - wybór silnika
 * @returns {Promise<Array>} - wynik predykcji w formie [klasa]
 */
export const predictTens = async (points, engine = "tf") => {
  const flatInput = points.map((p) => p.y);

  try {
    if (engine === "tf") {
      const result = await TFLiteModule.predict(flatInput);
      return [result]; // zachowujemy ten sam interface
    } else if (engine === "torch") {
      if (flatInput.length !== 30 * 6) {
        console.warn("Input length should be 180 floats (30x6)");
        return null;
      }
      const result = await PytorchModule.predict(flatInput);
      return [result];
    } else {
      console.warn("Unknown engine, defaulting to tf");
      const result = await TFLiteModule.predict(flatInput);
      return [result];
    }
  } catch (error) {
    console.error(`${engine} prediction error:`, error);
    return null;
  }
};

/**
 * Funkcja uniwersalna do predykcji dla Accelerometru
 * @param {Array} points - tablica punktów {y, x}
 * @param {"tf" | "torch"} engine - wybór silnika
 * @returns {Promise<Array>} - wynik predykcji w formie [klasa]
 */
export const predictAcc = async (points, engine = "tf") => {
  const flatInput = points.map((p) => p.y);

  try {
    if (engine === "tf") {
      const result = await TFLiteModule.predictAcc(flatInput);
      return [result];
    } else if (engine === "torch") {
      if (flatInput.length !== 30 * 6) {
        console.warn("Input length should be 180 floats (30x6)");
        return null;
      }
      const result = await PytorchModule.predict(flatInput);
      return [result];
    } else {
      console.warn("Unknown engine, defaulting to tf");
      const result = await TFLiteModule.predictAcc(flatInput);
      return [result];
    }
  } catch (error) {
    console.error(`${engine} prediction error:`, error);
    return null;
  }
};