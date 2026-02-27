import { NativeModules } from "react-native";
const { TFLiteModule, PytorchModule } = NativeModules;

/**
 * Funkcja uniwersalna do predykcji dla Tensometru
 * @param {Array} points - tablica punktów {y, x}
 * @param {"tf" | "torch"} engine - wybór silnika
 * @returns {Promise<Array>} - wynik predykcji w formie [klasa]
 */
let torchBuffer = [];
export const predictTens = async (points, engine = "tf") => {
  const flatInput = points.map((p) => p.y);

  try {
    if (engine === "tf") {
      const result = await TFLiteModule.predict(flatInput);
      return [result]; // zachowujemy ten sam interface
    } else if (engine === "torch") {
      if(flatInput.length==12)
        return;
      torchBuffer.push(...flatInput);

      // Jeśli mamy mniej niż 180 → jeszcze nie predykujemy
      if (torchBuffer.length < 180) {
        console.warn("Zbieram dane:", torchBuffer.length);
        return null;
      }

      // Jeśli mamy więcej niż 180 → przytnij do ostatnich 180
      if (torchBuffer.length > 180) {
        torchBuffer = torchBuffer.slice(-180);
      }
      const result = await PytorchModule.predict(torchBuffer);

      torchBuffer = torchBuffer.slice(6);

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
        console.warn("Input  adjhashgjdajh length should be 180 floats (30x6)");
        console.warn("a wynosi");
        
        console.warn(flatInput?.length);
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