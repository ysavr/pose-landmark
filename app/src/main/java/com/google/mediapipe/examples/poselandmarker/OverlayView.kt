package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: PoseLandmarkerResult? = null
    private var pointPaint = Paint()
    private var linePaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var offsetX: Float = 0f
    private var offsetY: Float = 0f

    private val poseColors = listOf(
        Color.YELLOW, Color.CYAN, Color.MAGENTA, Color.GREEN, Color.RED
    )

    init {
        initPaints()
    }

    fun clear() {
        results = null
        invalidate()
    }

    private fun initPaints() {
        linePaint.color = ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        results?.let { poseLandmarkerResult ->
            val allLandmarks = poseLandmarkerResult.landmarks()

            for (poseIndex in allLandmarks.indices) {
                val landmarks = allLandmarks[poseIndex]
                val pointColor = poseColors[poseIndex % poseColors.size]
                pointPaint.color = pointColor

                for (normalizedLandmark in landmarks) {
                    val originalX = normalizedLandmark.x() * imageWidth
                    val originalY = normalizedLandmark.y() * imageHeight

                    // Rotate 90 degrees clockwise
                    val rotatedX = (imageHeight - originalY) * scaleFactor + offsetX
                    val rotatedY = (originalX) * scaleFactor + offsetY

                    canvas.drawPoint(rotatedX, rotatedY, pointPaint)
                }

                POSE_CONNECTIONS.forEach { connection ->
                    val startLandmark = landmarks[connection.first]
                    val endLandmark = landmarks[connection.second]

                    val startX = startLandmark.x() * imageWidth
                    val startY = startLandmark.y() * imageHeight
                    val endX = endLandmark.x() * imageWidth
                    val endY = endLandmark.y() * imageHeight

                    // Rotate connections by 90 degrees clockwise
                    val rotatedStartX = (imageHeight - startY) * scaleFactor + offsetX
                    val rotatedStartY = (startX) * scaleFactor + offsetY
                    val rotatedEndX = (imageHeight - endY) * scaleFactor + offsetX
                    val rotatedEndY = (endX) * scaleFactor + offsetY

                    canvas.drawLine(rotatedStartX, rotatedStartY, rotatedEndX, rotatedEndY, linePaint)
                }
            }
        }
    }

    fun setResults(
        poseLandmarkerResults: PoseLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        results = poseLandmarkerResults
        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        // Maintain aspect ratio
        val scaleX = width.toFloat() / imageWidth
        val scaleY = height.toFloat() / imageHeight
        scaleFactor = min(scaleX, scaleY)

        // Calculate offsets for centering
        offsetX = (width - (imageWidth * scaleFactor)) / 2f
        offsetY = (height - (imageHeight * scaleFactor)) / 2f

        Log.d("OverlayView", "ScaleFactor: $scaleFactor, OffsetX: $offsetX, OffsetY: $offsetY")

        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 12F

        private val POSE_CONNECTIONS = listOf(
            Pair(0, 1), Pair(1, 2), Pair(2, 3), Pair(3, 7), Pair(0, 4),
            Pair(4, 5), Pair(5, 6), Pair(6, 8), Pair(9, 10), Pair(11, 12),
            Pair(12, 14), Pair(14, 16), Pair(16, 22), Pair(16, 20), Pair(16, 18),
            Pair(18, 20), Pair(11, 13), Pair(13, 15), Pair(15, 21), Pair(15, 19),
            Pair(15, 17), Pair(17, 19), Pair(12, 24), Pair(24, 26), Pair(26, 28),
            Pair(28, 32), Pair(32, 30), Pair(28, 30), Pair(11, 23), Pair(23, 25),
            Pair(25, 27), Pair(27, 29), Pair(29, 31), Pair(27, 31)
        )
    }
}
