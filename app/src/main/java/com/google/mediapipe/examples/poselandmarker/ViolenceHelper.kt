/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import kotlin.collections.HashMap

class ViolenceHelper(
    var minPoseDetectionConfidence: Float = DEFAULT_POSE_DETECTION_CONFIDENCE,
    var minPoseTrackingConfidence: Float = DEFAULT_POSE_TRACKING_CONFIDENCE,
    var minPosePresenceConfidence: Float = DEFAULT_POSE_PRESENCE_CONFIDENCE,
    var maxPoses: Int = MAX_POSE,
    var currentDelegate: Int = DELEGATE_CPU,
    var currentModel: Int = MODEL_POSE_LANDMARKER_FULL,
    var runningMode: RunningMode = RunningMode.LIVE_STREAM,
    val context: Context,
    val violenceListener: ViolenceListener?
) {

    companion object {
        private const val TAG = "ViolenceHelper"
        private const val DEFAULT_POSE_DETECTION_CONFIDENCE = 0.7F
        private const val DEFAULT_POSE_TRACKING_CONFIDENCE = 0.7F
        private const val DEFAULT_POSE_PRESENCE_CONFIDENCE = 0.7F
        private const val MAX_POSE = 5
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val GPU_ERROR = 1
        private const val SEQUENCE_LENGTH = 10
        private const val NUM_KEYPOINTS = 33
        private const val FEATURE_SIZE = 99 // 33 * 3 (x, y, visibility) for TFLite
        private const val TFLITE_INPUT_SIZE = 1 * SEQUENCE_LENGTH * FEATURE_SIZE * 4 // Bytes
        private const val TFLITE_OUTPUT_SIZE = 1 * 1 * 4 // Bytes
        const val MODEL_POSE_LANDMARKER_FULL = 0
        const val MODEL_POSE_LANDMARKER_LITE = 1
        const val MODEL_POSE_LANDMARKER_HEAVY = 2
    }

    private var poseLandmarker: PoseLandmarker? = null
    private var tfliteInterpreter: Interpreter? = null
    private val poseSequences = HashMap<Int, LinkedList<FloatArray>>()
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(TFLITE_INPUT_SIZE).apply { order(ByteOrder.nativeOrder()) }
    private val outputBuffer: ByteBuffer = ByteBuffer.allocateDirect(TFLITE_OUTPUT_SIZE).apply { order(ByteOrder.nativeOrder()) }

    init {
        setupPoseLandmarker()
        setupTfliteModel()
    }

    fun clearViolenceHelper() {
        poseLandmarker?.close()
        poseLandmarker = null
        poseSequences.clear()
        inputBuffer.clear()
        outputBuffer.clear()
    }

    fun isClose(): Boolean = poseLandmarker == null

    fun setupPoseLandmarker() {
        val baseOptionBuilder = BaseOptions.builder()

        when (currentDelegate) {
            DELEGATE_CPU -> baseOptionBuilder.setDelegate(Delegate.CPU)
            DELEGATE_GPU -> baseOptionBuilder.setDelegate(Delegate.GPU)
        }

        val modelName = when (currentModel) {
            MODEL_POSE_LANDMARKER_FULL -> "pose_landmarker_full.task"
            MODEL_POSE_LANDMARKER_LITE -> "pose_landmarker_lite.task"
            MODEL_POSE_LANDMARKER_HEAVY -> "pose_landmarker_heavy.task"
            else -> "pose_landmarker_full.task"
        }

        baseOptionBuilder.setModelAssetPath(modelName)

        if (runningMode == RunningMode.LIVE_STREAM && violenceListener == null) {
            throw IllegalStateException("violenceListener must be set when runningMode is LIVE_STREAM.")
        }

        try {
            val baseOptions = baseOptionBuilder.build()
            val optionsBuilder = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setNumPoses(maxPoses)
                .setMinPoseDetectionConfidence(minPoseDetectionConfidence)
                .setMinTrackingConfidence(minPoseTrackingConfidence)
                .setMinPosePresenceConfidence(minPosePresenceConfidence)
                .setRunningMode(runningMode)

            if (runningMode == RunningMode.LIVE_STREAM) {
                optionsBuilder
                    .setResultListener(this::returnLivestreamResult)
                    .setErrorListener(this::returnLivestreamError)
            }

            poseLandmarker = PoseLandmarker.createFromOptions(context, optionsBuilder.build())
            Log.i(TAG, "Pose Landmarker initialized successfully with model: $modelName")
        } catch (e: Exception) {
            violenceListener?.onError("Pose Landmarker failed to initialize: ${e.message}", if (e is RuntimeException) GPU_ERROR else 0)
            Log.e(TAG, "Pose Landmarker setup failed: ${e.message}")
        }
    }

    private fun setupTfliteModel() {
        try {
            val tfliteModel = context.assets.open("violence_mediapipe_multiperson.tflite").readBytes()
            val modelBuffer = ByteBuffer.allocateDirect(tfliteModel.size).apply {
                order(ByteOrder.nativeOrder())
                put(tfliteModel)
                rewind()
            }
            tfliteInterpreter = Interpreter(modelBuffer)
            Log.i(TAG, "TFLite model loaded successfully")
        } catch (e: Exception) {
            violenceListener?.onError("TFLite model failed to load: ${e.message}")
            Log.e(TAG, "TFLite model setup failed: ${e.message}")
            tfliteInterpreter = null
        }
    }

    fun detectLiveStream(imageProxy: ImageProxy, isFrontCamera: Boolean) {
        val frameTime = SystemClock.uptimeMillis()
        val bitmapBuffer = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
        Log.d(TAG, "ImageProxy: width=${imageProxy.width}, height=${imageProxy.height}, rotation=${imageProxy.imageInfo.rotationDegrees}")
        imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
        imageProxy.close()
        val mpImage = BitmapImageBuilder(bitmapBuffer).build()
        poseLandmarker?.detectAsync(mpImage, frameTime)
    }

    private fun preprocessForRendering(result: PoseLandmarkerResult): List<FloatArray> {
        val allLandmarks = result.landmarks()
        val renderingLandmarks = MutableList(allLandmarks.size) { FloatArray(NUM_KEYPOINTS * 3) }

        for (i in allLandmarks.indices) {
            val landmarks = allLandmarks[i]
            for (j in 0 until NUM_KEYPOINTS) {
                val landmark = landmarks[j]
                renderingLandmarks[i][j * 3] = landmark.x()
                renderingLandmarks[i][j * 3 + 1] = landmark.y()
                renderingLandmarks[i][j * 3 + 2] = landmark.presence().orElse(0.0f)
            }
        }
        return renderingLandmarks
    }

    private fun preprocessForTflite(result: PoseLandmarkerResult): List<FloatArray> {
        val allLandmarks = result.landmarks()
        val tfliteLandmarks = MutableList(allLandmarks.size) { FloatArray(NUM_KEYPOINTS * 3) } // 33 * 3 = 99

        for (i in allLandmarks.indices) {
            val landmarks = allLandmarks[i]
            for (j in 0 until NUM_KEYPOINTS) {
                val landmark = landmarks[j]
                tfliteLandmarks[i][j * 3] = landmark.x()
                tfliteLandmarks[i][j * 3 + 1] = landmark.y()
                tfliteLandmarks[i][j * 3 + 2] = landmark.presence().orElse(0.0f)
            }
        }
        return tfliteLandmarks
    }

    private fun detectViolenceWithTflite(landmarksList: List<FloatArray>): List<Boolean> {
        if (tfliteInterpreter == null) return List(landmarksList.size) { false }

        val violencePredictions = MutableList(landmarksList.size) { false }
        val sequenceReady = mutableListOf<Int>()

        // Update sequences for each pose
        for (poseIndex in landmarksList.indices) {
            val sequence = poseSequences.getOrPut(poseIndex) { LinkedList() }
            sequence.add(landmarksList[poseIndex])
            if (sequence.size > SEQUENCE_LENGTH) sequence.removeFirst()
            if (sequence.size == SEQUENCE_LENGTH) sequenceReady.add(poseIndex)
        }

        if (sequenceReady.isEmpty()) return violencePredictions

        // Process each pose with a full sequence
        for (poseIndex in sequenceReady) {
            val sequence = poseSequences[poseIndex]!!

            // Fill reusable input buffer
            inputBuffer.rewind()
            for (t in 0 until SEQUENCE_LENGTH) {
                val frame = sequence[t]
                for (k in frame.indices) {
                    inputBuffer.putFloat(frame[k])
                }
            }

            // Run inference
            outputBuffer.rewind()
            try {
                tfliteInterpreter?.run(inputBuffer, outputBuffer)
                outputBuffer.rewind()
                val prediction = outputBuffer.float
                Log.d(TAG, "Pose $poseIndex: Prediction = $prediction")
                violencePredictions[poseIndex] = prediction > 0.3f
            } catch (e: Exception) {
                Log.e(TAG, "TFLite inference failed for pose $poseIndex: ${e.message}")
            }
        }

        // Clean up sequences for poses no longer detected
        poseSequences.keys.retainAll(landmarksList.indices.toSet())

        return violencePredictions
    }

    private fun returnLivestreamResult(result: PoseLandmarkerResult, input: MPImage) {
        val finishTimeMs = SystemClock.uptimeMillis()
        val inferenceTimeMs = finishTimeMs - result.timestampMs()

        val renderingLandmarks = preprocessForRendering(result)
        val tfliteLandmarks = preprocessForTflite(result)

        val isViolent = detectViolenceWithTflite(tfliteLandmarks)
        violenceListener?.onResults(
            ResultBundle(
                results = result,
                inferenceTime = inferenceTimeMs,
                inputImageHeight = input.height,
                inputImageWidth = input.width,
                isViolent = isViolent,
                renderingLandmarks = renderingLandmarks
            )
        )
    }

    private fun returnLivestreamError(error: RuntimeException) {
        violenceListener?.onError(error.message ?: "Unknown error", GPU_ERROR)
    }

    data class ResultBundle(
        val results: PoseLandmarkerResult,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
        val isViolent: List<Boolean>,
        val renderingLandmarks: List<FloatArray>
    )

    interface ViolenceListener {
        fun onError(error: String, errorCode: Int = 0)
        fun onResults(resultBundle: ResultBundle)
    }
}