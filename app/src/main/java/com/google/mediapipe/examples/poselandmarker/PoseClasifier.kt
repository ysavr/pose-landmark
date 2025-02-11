package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.json.JSONObject

class PoseClassifier(context: Context) {
    private val interpreter: Interpreter
    private val labels: Map<Int, String>

    init {
        // Load TFLite model
        val model = FileUtil.loadMappedFile(context, "pose_classifier_2.tflite")
        interpreter = Interpreter(model)

        // Load Labels from JSON
        val jsonString = context.assets.open("pose_labels.json").bufferedReader().use { it.readText() }

        // Convert JSON to JSONObject
        val jsonObject = JSONObject(jsonString)

        // Convert JSONObject to Map<Int, String>
        labels = jsonObject.keys().asSequence()
            .associate { it.toInt() to jsonObject.getString(it) }
    }

    fun classifyPose(landmarks: FloatArray): String {
        if (landmarks.isEmpty()) return "No landmarks detected"

        Log.d("PoseClassifier", "classifyPose: landmark size ${landmarks.size}")

        val input = arrayOf(landmarks) // Convert landmarks to model input format
        val output = Array(1) { FloatArray(labels.size) } // Expected output shape [1, num_classes]

        try {
            // Ensure interpreter is initialized
            if (interpreter == null) return "Model not initialized"

            val inputTensor = interpreter.getInputTensor(0)
            val inputShape = inputTensor.shape() // Example: [1, 99]

            Log.d("PoseClassifier", "Expected input shape: ${inputShape.contentToString()}")
            Log.d("PoseClassifier", "Actual input size: ${landmarks.size}")

            // Check if input size matches model requirements
            if (landmarks.size != inputShape[1]) {
                return "Invalid input size: Expected ${inputShape[1]}, got ${landmarks.size}"
            }

            // Get actual output tensor shape
            val outputTensor = interpreter.getOutputTensor(0)
            val outputShape = outputTensor.shape() // Example: [1, 5]

            // Check if the first dimension is 0 (no valid output)
            if (outputShape[0] == 0) return "No pose detected"

            // Run inference
            interpreter.run(input, output)

            // Check if output contains valid probabilities
            if (output[0].all { it == 0f }) return "Pose not recognized"

            // Get index of highest probability
            val maxIndex = output[0].indices.maxByOrNull { output[0][it] } ?: return "Unknown"

            // Return label name
            return labels[maxIndex] ?: "Unknown"
        } catch (e: IllegalArgumentException) {
            e.printStackTrace()
            return "Invalid pose input"
        } catch (e: Exception) {
            e.printStackTrace()
            return "Error running pose classification"
        }
    }

}
