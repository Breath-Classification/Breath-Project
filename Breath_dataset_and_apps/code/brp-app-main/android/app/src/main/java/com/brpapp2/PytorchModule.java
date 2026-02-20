package com.brpapp2;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import com.facebook.react.bridge.Callback;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableType;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import org.tensorflow.lite.DataType;
import org.pytorch.Module;
import org.pytorch.IValue;
import org.pytorch.Tensor;
import android.content.Context;
import java.io.File;
import java.io.InputStream;
import java.io.FileOutputStream;

public class PytorchModule extends ReactContextBaseJavaModule {

private Module model;

  public PytorchModule(ReactApplicationContext reactContext) {
    super(reactContext);
  }

  @Override
  public String getName() {
    return "PytorchModule";
  }

  @ReactMethod
  public synchronized void loadModel(
    int sizeOfInput,
    String nameOfTheModel,
    Promise promise
  ) { // synchronized for safe model loading
    try {
      model = Module.load(assetFilePath(getReactApplicationContext(), nameOfTheModel + ".pt"));
      // Initialize buffers after the model is loaded
      promise.resolve("Model loaded successfully");
    } catch (Exception e) {
      promise.reject("ERROR_LOADING_MODEL", e);
    }
  }
  
  private float[][] createSlidingWindows(float[] input, int windowSize, int stride) {
    int numWindows = (input.length - windowSize) / stride + 1;
    float[][] windows = new float[numWindows][windowSize];

    for (int i = 0; i < numWindows; i++) {
        for (int j = 0; j < windowSize; j++) {
            windows[i][j] = input[i * stride + j];
        }
    }
    return windows;
}

  @ReactMethod
  public void predict(ReadableArray variables,int windowSize, int stride, Promise promise) {
    if (model == null) {
      promise.reject(
        "MODEL_NOT_LOADED",
        "Model not loaded. Make sure to call loadModel() first."
      );
      return;
    }

    // Set values for the inputArray
    float[] inputArray = new float[variables.size()];
    for (int i = 0; i < variables.size(); i++) {
      // Make sure the value is a number and cast it to float
      if (!variables.isNull(i) && variables.getType(i) == ReadableType.Number) {
        inputArray[i] = (float) variables.getDouble(i);
      } else {
        promise.reject(
          "INVALID_INPUT_TYPE",
          "Input must be an array of numbers."
        );
        return;
      }
    }

     // Tworzymy sliding windows
    if (inputArray.length < windowSize) {
        promise.reject("INPUT_TOO_SHORT", "Input length is smaller than window size.");
        return;
    }
    float[][] windows = createSlidingWindows(inputArray, windowSize, stride);

    // Lista wyników dla każdego okna
    int[] results = new int[windows.length];

    for (int i = 0; i < windows.length; i++) {
        Tensor inputTensor = Tensor.fromBlob(windows[i], new long[]{1, windowSize});
        Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
        float[] outputArray = outputTensor.getDataAsFloatArray();

        // Znajdujemy indeks max dla tego okna
        int maxIndex = 0;
        float maxValue = outputArray[0];
        for (int j = 1; j < outputArray.length; j++) {
            if (outputArray[j] > maxValue) {
                maxValue = outputArray[j];
                maxIndex = j;
            }
        }

        results[i] = maxIndex;
    }

    // Zwracamy tablicę wyników do JS
    promise.resolve(Arrays.asList(Arrays.stream(results).boxed().toArray(Integer[]::new)));
}

  private String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
        return file.getAbsolutePath();
    }
    try (InputStream is = context.getAssets().open(assetName);
         FileOutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
            os.write(buffer, 0, read);
        }
        os.flush();
        return file.getAbsolutePath();
    }
}
}
