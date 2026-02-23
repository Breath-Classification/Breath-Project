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
   @ReactMethod
  public synchronized void loadAccModel(
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
  
  @ReactMethod
public void predict(ReadableArray variables, Promise promise) {

    if (model == null) {
        promise.reject("MODEL_NOT_LOADED", "Model not loaded.");
        return;
    }

    if (variables.size() != 180) {
        promise.reject("INVALID_INPUT_SIZE",
            "Expected 180 values (30x6) but got " + variables.size());
        return;
    }

    float[] inputArray = new float[variables.size()];

    for (int i = 0; i < variables.size(); i++) {
        if (!variables.isNull(i) && variables.getType(i) == ReadableType.Number) {
            inputArray[i] = (float) variables.getDouble(i);
        } else {
            promise.reject("INVALID_INPUT_TYPE", "Input must be numeric.");
            return;
        }
    }

    // LSTM input shape: [1, 30, 6]
    Tensor inputTensor = Tensor.fromBlob(
        inputArray,
        new long[]{1, 30, 6}
    );

    Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
    float[] outputArray = outputTensor.getDataAsFloatArray();

    // argmax
    int maxIndex = 0;
    float maxValue = outputArray[0];

    for (int i = 1; i < outputArray.length; i++) {
        if (outputArray[i] > maxValue) {
            maxValue = outputArray[i];
            maxIndex = i;
        }
    }
    
    promise.resolve(maxIndex+1);
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
