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
package com.google.mediapipe.examples.poselandmarker.fragment

import android.annotation.SuppressLint
import android.content.Context
import android.content.res.Configuration
import android.graphics.Point
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Camera
import androidx.camera.core.AspectRatio
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.Navigation
import com.google.mediapipe.examples.poselandmarker.PoseLandmarkerHelper
import com.google.mediapipe.examples.poselandmarker.MainViewModel
import com.google.mediapipe.examples.poselandmarker.PoseClassifier
import com.google.mediapipe.examples.poselandmarker.R
import com.google.mediapipe.examples.poselandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.lang.StrictMath.atan2
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.absoluteValue
import kotlin.math.atan2

class CameraFragment : Fragment(), PoseLandmarkerHelper.LandmarkerListener {

    companion object {
        private const val TAG = "Pose Landmarker"
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var poseLandmarkerHelper: PoseLandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraFacing = CameraSelector.LENS_FACING_BACK

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ExecutorService

    private fun detectPose(result: PoseLandmarkerResult): String {
        val landmarks = preprocessPoseLandmarks(result)

        // Ensure we have 33 landmarks, each with x, y, z (total 99 values)
        if (landmarks.size != 99) {
            Log.e(TAG, "Incorrect landmark size: ${landmarks.size}, expected 99")
            return "Invalid pose input"
        }

        // Initialize Pose Classifier
        val poseClassifier = PoseClassifier(requireContext())

        // Run classification
        val poseLabel = poseClassifier.classifyPose(landmarks)

        return poseLabel
    }

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(), R.id.fragment_container
            ).navigate(R.id.action_camera_to_permissions)
        }

        // Start the PoseLandmarkerHelper again when users come back
        // to the foreground.
        backgroundExecutor.execute {
            if(this::poseLandmarkerHelper.isInitialized) {
                if (poseLandmarkerHelper.isClose()) {
                    poseLandmarkerHelper.setupPoseLandmarker()
                }
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if(this::poseLandmarkerHelper.isInitialized) {
            viewModel.setMinPoseDetectionConfidence(poseLandmarkerHelper.minPoseDetectionConfidence)
            viewModel.setMinPoseTrackingConfidence(poseLandmarkerHelper.minPoseTrackingConfidence)
            viewModel.setMinPosePresenceConfidence(poseLandmarkerHelper.minPosePresenceConfidence)
            viewModel.setDelegate(poseLandmarkerHelper.currentDelegate)

            // Close the PoseLandmarkerHelper and release resources
            backgroundExecutor.execute { poseLandmarkerHelper.clearPoseLandmarker() }
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()

        // Shut down our background executor
        backgroundExecutor.shutdown()
        backgroundExecutor.awaitTermination(
            Long.MAX_VALUE, TimeUnit.NANOSECONDS
        )
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Initialize our background executor
        backgroundExecutor = Executors.newSingleThreadExecutor()

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }

        poseLandmarkerHelper = PoseLandmarkerHelper(
            context = requireContext(),
            runningMode = RunningMode.LIVE_STREAM,
            minPoseDetectionConfidence = viewModel.currentMinPoseDetectionConfidence,
            minPoseTrackingConfidence = viewModel.currentMinPoseTrackingConfidence,
            minPosePresenceConfidence = viewModel.currentMinPosePresenceConfidence,
            currentDelegate = viewModel.currentDelegate,
            poseLandmarkerHelperListener = this
        )
        // Create the PoseLandmarkerHelper that will handle the inference
        backgroundExecutor.execute {
            poseLandmarkerHelper = PoseLandmarkerHelper(
                context = requireContext(),
                runningMode = RunningMode.LIVE_STREAM,
                minPoseDetectionConfidence = viewModel.currentMinPoseDetectionConfidence,
                minPoseTrackingConfidence = viewModel.currentMinPoseTrackingConfidence,
                minPosePresenceConfidence = viewModel.currentMinPosePresenceConfidence,
                currentDelegate = viewModel.currentDelegate,
                poseLandmarkerHelperListener = this
            )
        }

        // Attach listeners to UI control widgets
        initBottomSheetControls()
    }

    private fun initBottomSheetControls() {
        // init bottom sheet settings

        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinPoseDetectionConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinPoseTrackingConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinPosePresenceConfidence
            )

        // When clicked, lower pose detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdMinus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseDetectionConfidence >= 0.2) {
                poseLandmarkerHelper.minPoseDetectionConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise pose detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdPlus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseDetectionConfidence <= 0.8) {
                poseLandmarkerHelper.minPoseDetectionConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower pose tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdMinus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseTrackingConfidence >= 0.2) {
                poseLandmarkerHelper.minPoseTrackingConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise pose tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdPlus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseTrackingConfidence <= 0.8) {
                poseLandmarkerHelper.minPoseTrackingConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower pose presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdMinus.setOnClickListener {
            if (poseLandmarkerHelper.minPosePresenceConfidence >= 0.2) {
                poseLandmarkerHelper.minPosePresenceConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise pose presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdPlus.setOnClickListener {
            if (poseLandmarkerHelper.minPosePresenceConfidence <= 0.8) {
                poseLandmarkerHelper.minPosePresenceConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, change the underlying hardware used for inference.
        // Current options are CPU and GPU
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
            viewModel.currentDelegate, false
        )
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {
                    try {
                        poseLandmarkerHelper.currentDelegate = p2
                        updateControlsUi()
                    } catch(e: UninitializedPropertyAccessException) {
                        Log.e(TAG, "PoseLandmarkerHelper has not been initialized yet.")
                    }
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }

        // When clicked, change the underlying model used for object detection
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.setSelection(
            viewModel.currentModel,
            false
        )
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?,
                    p1: View?,
                    p2: Int,
                    p3: Long
                ) {
                    poseLandmarkerHelper.currentModel = p2
                    updateControlsUi()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }
    }

    // Update the values displayed in the bottom sheet. Reset Poselandmarker
    // helper.
    private fun updateControlsUi() {
        if(this::poseLandmarkerHelper.isInitialized) {
            fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
                String.format(
                    Locale.US,
                    "%.2f",
                    poseLandmarkerHelper.minPoseDetectionConfidence
                )
            fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
                String.format(
                    Locale.US,
                    "%.2f",
                    poseLandmarkerHelper.minPoseTrackingConfidence
                )
            fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
                String.format(
                    Locale.US,
                    "%.2f",
                    poseLandmarkerHelper.minPosePresenceConfidence
                )

            // Needs to be cleared instead of reinitialized because the GPU
            // delegate needs to be initialized on the thread using it when applicable
            backgroundExecutor.execute {
                poseLandmarkerHelper.clearPoseLandmarker()
                poseLandmarkerHelper.setupPoseLandmarker()
            }
            fragmentCameraBinding.overlay.clear()
        }
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            }, ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {

        // CameraProvider
        val cameraProvider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(cameraFacing).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer =
            ImageAnalysis.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(backgroundExecutor) { image ->
                        detectPose(image)
                    }
                }

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun detectPose(imageProxy: ImageProxy) {
        if(this::poseLandmarkerHelper.isInitialized) {
            poseLandmarkerHelper.detectLiveStream(
                imageProxy = imageProxy,
                isFrontCamera = cameraFacing == CameraSelector.LENS_FACING_FRONT
            )
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation =
            fragmentCameraBinding.viewFinder.display.rotation
    }

    private val poseLandmarks = listOf(
        "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer", "Right Eye Inner", "Right Eye",
        "Right Eye Outer", "Left Ear", "Right Ear", "Mouth Left", "Mouth Right", "Left Shoulder",
        "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Pinky",
        "Right Pinky", "Left Index", "Right Index", "Left Thumb", "Right Thumb", "Left Hip",
        "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Left Heel",
        "Right Heel", "Left Foot Index", "Right Foot Index"
    )

    private fun processPoseResult(result: PoseLandmarkerResult) {
        result.landmarks().firstOrNull()?.forEachIndexed { index, landmark ->
            val bodyPart = poseLandmarks.getOrNull(index) ?: "Unknown"
            Log.d(TAG, "PoseMapping $bodyPart -> x=${landmark.x()}, y=${landmark.y()}, z=${landmark.z()}")
        }
    }

    private fun isRightHandRaised(result: PoseLandmarkerResult): Boolean {
        val landmarks = result.landmarks().firstOrNull() ?: return false
        val rightWrist = landmarks[14]
        val rightShoulder = landmarks[10]
        return rightWrist.y() < rightShoulder.y()
    }

    private fun isPlankPose(result: PoseLandmarkerResult): Boolean {
        val landmarks = result.landmarks().firstOrNull() ?: return false

        val leftShoulder = landmarks[11]
        val rightShoulder = landmarks[12]
        val leftHip = landmarks[23]
        val rightHip = landmarks[24]
        val leftAnkle = landmarks[27]
        val rightAnkle = landmarks[28]

        // Check alignment of shoulders, hips, and ankles
        val shoulderHipSlope = (leftShoulder.y() - leftHip.y()).absoluteValue +
                (rightShoulder.y() - rightHip.y()).absoluteValue
        val hipAnkleSlope = (leftHip.y() - leftAnkle.y()).absoluteValue +
                (rightHip.y() - rightAnkle.y()).absoluteValue

        // Thresholds to determine alignment (adjust as necessary)
        val alignmentThreshold = 0.1f

        // Ensure the body is straight (shoulders, hips, and ankles aligned)
        return shoulderHipSlope < alignmentThreshold && hipAnkleSlope < alignmentThreshold
    }

    private fun isTreePose(result: PoseLandmarkerResult): Boolean {
        val landmarks = result.landmarks().firstOrNull() ?: return false

        // Landmark indices for key body parts
        val leftWrist = landmarks[15]
        val rightWrist = landmarks[16]
        val leftShoulder = landmarks[11]
        val rightShoulder = landmarks[12]
        val leftKnee = landmarks[25]
        val rightKnee = landmarks[26]
        val leftAnkle = landmarks[27]
        val rightAnkle = landmarks[28]

        // 1. Check if both hands are raised above the shoulders
        val handsRaised = leftWrist.y() < leftShoulder.y() && rightWrist.y() < rightShoulder.y()

        // 2. Check if one foot is on the opposite thigh (e.g., left ankle near right thigh or vice versa)
        val leftFootOnRightThigh = (leftAnkle.x() - rightKnee.x()).absoluteValue < 0.1f &&
                (leftAnkle.y() - rightKnee.y()).absoluteValue < 0.1f
        val rightFootOnLeftThigh = (rightAnkle.x() - leftKnee.x()).absoluteValue < 0.1f &&
                (rightAnkle.y() - leftKnee.y()).absoluteValue < 0.1f

        // Combine conditions to determine if the pose is a Tree Pose
        return handsRaised && (leftFootOnRightThigh || rightFootOnLeftThigh)
    }

    fun calculateAngle(a: Point, b: Point, c: Point): Double {
        val radians = atan2(c.y - b.y, c.x - b.x) - atan2(a.y - b.y, a.x - b.x)
        var angle = Math.toDegrees(radians.toDouble()).absoluteValue
        if (angle > 180) angle = 360 - angle
        return angle
    }

    data class Point(val x: Float, val y: Float)

    private fun isSquatPose(result: PoseLandmarkerResult): Boolean {
        val landmarks = result.landmarks().firstOrNull() ?: return false

        // Landmark indices for key body parts
        val leftHip = landmarks[23]
        val rightHip = landmarks[24]
        val leftKnee = landmarks[25]
        val rightKnee = landmarks[26]
        val leftAnkle = landmarks[27]
        val rightAnkle = landmarks[28]

        // Convert NormalizedLandmark to Point
        val leftHipPoint = Point(leftHip.x(), leftHip.y())
        val leftKneePoint = Point(leftKnee.x(), leftKnee.y())
        val leftAnklePoint = Point(leftAnkle.x(), leftAnkle.y())
        val rightHipPoint = Point(rightHip.x(), rightHip.y())
        val rightKneePoint = Point(rightKnee.x(), rightKnee.y())
        val rightAnklePoint = Point(rightAnkle.x(), rightAnkle.y())

        // Calculate the angles of the left and right legs
        val leftLegAngle = calculateAngle(leftHipPoint, leftKneePoint, leftAnklePoint)
        val rightLegAngle = calculateAngle(rightHipPoint, rightKneePoint, rightAnklePoint)

        // Check if both knees are bent (angle < 100 degrees, adjustable threshold)
        val kneesBent = leftLegAngle < 100 && rightLegAngle < 100

        // Check if hips are lower than knees
        val hipsLowered = leftHip.y() > leftKnee.y() && rightHip.y() > rightKnee.y()

        // Check if both feet are on the ground (ankles are at a similar height)
        val feetOnGround = (leftAnkle.y() - rightAnkle.y()).absoluteValue < 0.1f

        // Combine conditions to determine if the pose is a squat
        return kneesBent && hipsLowered && feetOnGround
    }

    private fun workOutActivity(result: PoseLandmarkerResult): String {
        val rightHandRaised = isRightHandRaised(result)
        val plankStatus = isPlankPose(result)
        val treePoseStatus = isTreePose(result)
        val squatStatus = isSquatPose(result)

        return when {
            plankStatus -> "Plank Pose"
            treePoseStatus -> "Tree Pose"
            rightHandRaised -> "Right Hand Raised"
            squatStatus -> "Squat Pose"
            else -> "I don't know"
        }
    }

    private fun preprocessPoseLandmarks(result: PoseLandmarkerResult): FloatArray {
        val landmarks = mutableListOf<Float>()

        // Ensure pose landmarks exist
        val poseLandmarks = result.landmarks().firstOrNull() ?: return FloatArray(0)

        for (landmark in poseLandmarks) {
            landmarks.add(landmark.x()) // X coordinate
            landmarks.add(landmark.y()) // Y coordinate
            landmarks.add(landmark.z()) // Z coordinate
        }

        if (landmarks.size != 99) {
            Log.e(TAG, "Extracted ${landmarks.size} landmarks, expected 99")
        }

        return landmarks.toFloatArray()
    }
    // Update UI after pose have been detected. Extracts original
    // image height/width to scale and place the landmarks properly through
    // OverlayView
    override fun onResults(
        resultBundle: PoseLandmarkerHelper.ResultBundle
    ) {
        var personActivity = ""
        var detectedPose = ""
        activity?.runOnUiThread {
            val results = resultBundle.results
            for (landmark in results) {
                detectedPose = detectPose(landmark)
                if (detectedPose != "Error classification") {
                    Log.d(TAG, "onResults: detectPose $detectedPose")
                }
                processPoseResult(landmark)
                personActivity = workOutActivity(landmark)
            }
            if (_fragmentCameraBinding != null) {
                fragmentCameraBinding.bottomSheetLayout.poseClassificationName.text = detectedPose
                fragmentCameraBinding.bottomSheetLayout.plankPoseStatus.text = personActivity
                fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                    String.format("%d ms", resultBundle.inferenceTime)

                // Pass necessary information to OverlayView for drawing on the canvas
                fragmentCameraBinding.overlay.setResults(
                    resultBundle.results.first(),
                    resultBundle.inputImageHeight,
                    resultBundle.inputImageWidth,
                    RunningMode.LIVE_STREAM
                )

                // Force a redraw
                fragmentCameraBinding.overlay.invalidate()
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
            if (errorCode == PoseLandmarkerHelper.GPU_ERROR) {
                fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
                    PoseLandmarkerHelper.DELEGATE_CPU, false
                )
            }
        }
    }
}
