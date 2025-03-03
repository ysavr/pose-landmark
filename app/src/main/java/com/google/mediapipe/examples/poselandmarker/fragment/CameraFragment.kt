package com.google.mediapipe.examples.poselandmarker.fragment

import android.annotation.SuppressLint
import android.content.Context
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.Navigation
import com.google.mediapipe.examples.poselandmarker.MainViewModel
import com.google.mediapipe.examples.poselandmarker.ViolenceHelper
import com.google.mediapipe.examples.poselandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFragment : Fragment(), ViolenceHelper.ViolenceListener {

    companion object {
        private const val TAG = "Violence Detection"
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null
    private val fragmentCameraBinding get() = _fragmentCameraBinding!!

    private lateinit var violenceHelper: ViolenceHelper
    private val viewModel: MainViewModel by activityViewModels()
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraFacing = CameraSelector.LENS_FACING_BACK

    private lateinit var backgroundExecutor: ExecutorService

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)
        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Initialize background executor
        backgroundExecutor = Executors.newSingleThreadExecutor()

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            setUpCamera()
        }

        // Initialize ViolenceHelper
        violenceHelper = ViolenceHelper(
            context = requireContext(),
            runningMode = RunningMode.LIVE_STREAM,
            minPoseDetectionConfidence = viewModel.currentMinPoseDetectionConfidence,
            minPoseTrackingConfidence = viewModel.currentMinPoseTrackingConfidence,
            minPosePresenceConfidence = viewModel.currentMinPosePresenceConfidence,
            currentDelegate = viewModel.currentDelegate,
            violenceListener = this
        )

        // Attach listeners to UI controls
        initBottomSheetControls()
    }

    private fun initBottomSheetControls() {
        // Update thresholds and model selection
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(Locale.US, "%.2f", viewModel.currentMinPoseDetectionConfidence)
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(Locale.US, "%.2f", viewModel.currentMinPoseTrackingConfidence)
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(Locale.US, "%.2f", viewModel.currentMinPosePresenceConfidence)

        // Threshold adjustment buttons
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdMinus.setOnClickListener {
            if (violenceHelper.minPoseDetectionConfidence >= 0.2) {
                violenceHelper.minPoseDetectionConfidence -= 0.1f
                updateControlsUi()
            }
        }
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdPlus.setOnClickListener {
            if (violenceHelper.minPoseDetectionConfidence <= 0.9) {
                violenceHelper.minPoseDetectionConfidence += 0.1f
                updateControlsUi()
            }
        }
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdMinus.setOnClickListener {
            if (violenceHelper.minPoseTrackingConfidence >= 0.2) {
                violenceHelper.minPoseTrackingConfidence -= 0.1f
                updateControlsUi()
            }
        }
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdPlus.setOnClickListener {
            if (violenceHelper.minPoseTrackingConfidence <= 0.9) {
                violenceHelper.minPoseTrackingConfidence += 0.1f
                updateControlsUi()
            }
        }
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdMinus.setOnClickListener {
            if (violenceHelper.minPosePresenceConfidence >= 0.2) {
                violenceHelper.minPosePresenceConfidence -= 0.1f
                updateControlsUi()
            }
        }
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdPlus.setOnClickListener {
            if (violenceHelper.minPosePresenceConfidence <= 0.9) {
                violenceHelper.minPosePresenceConfidence += 0.1f
                updateControlsUi()
            }
        }

        // Model selection spinner
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    violenceHelper.currentModel = position
                    updateControlsUi()
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
    }

    private fun updateControlsUi() {
        if (::violenceHelper.isInitialized) {
            fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
                String.format(Locale.US, "%.2f", violenceHelper.minPoseDetectionConfidence)
            fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
                String.format(Locale.US, "%.2f", violenceHelper.minPoseTrackingConfidence)
            fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
                String.format(Locale.US, "%.2f", violenceHelper.minPosePresenceConfidence)

            // Reinitialize ViolenceHelper with updated settings
            backgroundExecutor.execute {
                violenceHelper.clearViolenceHelper()
                violenceHelper.setupPoseLandmarker()
            }
            fragmentCameraBinding.overlay.clear()
        }
    }

    private fun setUpCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(cameraFacing)
            .build()

        // Preview use case
        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3) // Matches 640x480
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()
            .also {
                it.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
            }

        // Image analysis use case
        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also {
                it.setAnalyzer(backgroundExecutor) { imageProxy ->
                    detectPose(imageProxy)
                }
            }

        // Unbind all use cases before rebinding
        cameraProvider.unbindAll()

        try {
            cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun detectPose(imageProxy: ImageProxy) {
        if (::violenceHelper.isInitialized) {
            // Always pass isFrontCamera as false since we're using rear camera only
            violenceHelper.detectLiveStream(
                imageProxy = imageProxy,
                isFrontCamera = false // Force no flipping for rear camera
            )
        }
    }

    override fun onResults(resultBundle: ViolenceHelper.ResultBundle) {
        activity?.runOnUiThread {
            val results = resultBundle.results
            val inferenceTime = resultBundle.inferenceTime
            val inputImageHeight = resultBundle.inputImageHeight
            val inputImageWidth = resultBundle.inputImageWidth
            val isViolent = resultBundle.isViolent

            Log.d("CameraFragment", "Input dims: width=$inputImageWidth, height=$inputImageHeight")
            Log.d("CameraFragment", "Is Violent: $isViolent")

            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                String.format("%d ms", inferenceTime)

            if (isViolent.any { it }) {
                    fragmentCameraBinding.bottomSheetLayout.poseClassificationName.text = "Violent Activity Detected"
            } else {
                fragmentCameraBinding.bottomSheetLayout.poseClassificationName.text = "No Violent Activity"
            }

            fragmentCameraBinding.overlay.setResults(
                results,
                inputImageHeight,
                inputImageWidth,
                RunningMode.LIVE_STREAM
            )
            fragmentCameraBinding.overlay.invalidate()
        }
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Log.e(TAG, "Error: $error")
            if (errorCode == ViolenceHelper.GPU_ERROR) {
                fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
                    ViolenceHelper.DELEGATE_CPU, false
                )
            }
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()
        backgroundExecutor.shutdown()
        backgroundExecutor.awaitTermination(Long.MAX_VALUE, java.util.concurrent.TimeUnit.NANOSECONDS)
    }
}