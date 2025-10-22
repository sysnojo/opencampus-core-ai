package com.example.opencampus.ui.scanner

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.common.util.concurrent.ListenableFuture
import org.pytorch.*
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.util.concurrent.Executors

class ScannerActivity : ComponentActivity() {
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) openCamera()
            else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        checkCameraPermission()
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> openCamera()
            else -> requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun openCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        setContent {
            MaterialTheme {
                CameraScreen(cameraProviderFuture)
            }
        }
    }
}

@Composable
fun CameraScreen(cameraProviderFuture: ListenableFuture<ProcessCameraProvider>) {
    val context = LocalContext.current
    var labelText by remember { mutableStateOf("Loading model...") }
    var focusType by remember { mutableStateOf("ruang_dosen_1") } // Bisa diganti jadi "ruang_dosen_2"

    val executor = remember { Executors.newSingleThreadExecutor() }
    val previewView = remember { PreviewView(context) }

    // === LOAD LABELS ===
    val labels = context.assets.open("class_names.json").bufferedReader().use {
        org.json.JSONArray(it.readText()).let { jsonArr ->
            List(jsonArr.length()) { i -> jsonArr.getString(i) }
        }
    }

    // === FILTER LABEL SESUAI FOKUS (ruang_dosen_1 atau ruang_dosen_2) ===
    fun getTargetLabels(type: String): Pair<List<Int>, List<String>> {
        val idx = labels.indices.filter { labels[it].startsWith(type + "_") }
        return Pair(idx, idx.map { labels[it] })
    }

    LaunchedEffect(Unit) {
        try {
            // === LOAD MODEL ===
            val modelName = "oc-mnet-v0.ptl"
            val internalFile = File(context.filesDir, modelName)
            if (!internalFile.exists()) {
                context.assets.open(modelName).use { input ->
                    internalFile.outputStream().use { output -> input.copyTo(output) }
                }
            }
            val module = LiteModuleLoader.load(internalFile.absolutePath)
            labelText = "Model loaded ✅"

            // === OPEN CAMERA ===
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                imageAnalyzer.setAnalyzer(executor, ImageAnalysis.Analyzer { imageProxy ->
                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                    val bitmap = imageProxy.toBitmap(rotationDegrees)

                    if (bitmap != null) {
                        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
                        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                            resized,
                            floatArrayOf(0.485f, 0.456f, 0.406f),
                            floatArrayOf(0.229f, 0.224f, 0.225f)
                        )

                        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
                        val scores = outputTensor.dataAsFloatArray

                        val (targetIdx, targetLbl) = getTargetLabels(focusType)
                        val filteredScores = targetIdx.map { scores[it] }

                        // Ambil skor tertinggi
                        val maxIdx = filteredScores.indices.maxByOrNull { filteredScores[it] } ?: -1
                        val predicted = if (maxIdx >= 0) targetLbl[maxIdx] else "Unknown"

                        // Log debug top-3
                        filteredScores.withIndex()
                            .sortedByDescending { it.value }
                            .take(3)
                            .forEach {
                                Log.i(
                                    "TorchRealtime",
                                    "${targetLbl[it.index]}: ${"%.2f".format(it.value)}"
                                )
                            }

                        Log.i("TorchRealtime", "✅ Predicted: $predicted")
                        labelText = predicted
                    }

                    imageProxy.close()
                })

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    context as ComponentActivity,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
            }, ContextCompat.getMainExecutor(context))

        } catch (e: Exception) {
            labelText = "Model load failed: ${e.message}"
            e.printStackTrace()
        }
    }

    // === UI ===
    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .background(Color.Black.copy(alpha = 0.6f))
                .padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = labelText,
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp
            )
            Spacer(modifier = Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { focusType = "ruang_dosen_1" }) {
                    Text("Fokus: Dosen 1")
                }
                Button(onClick = { focusType = "ruang_dosen_2" }) {
                    Text("Fokus: Dosen 2")
                }
            }
        }
    }
}

// === EXTENSION UNTUK CONVERT IMAGEPROXY KE BITMAP ===
fun ImageProxy.toBitmap(rotationDegrees: Int): Bitmap? {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = android.graphics.YuvImage(
        nv21,
        android.graphics.ImageFormat.NV21,
        width,
        height,
        null
    )

    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 90, out)
    val imageBytes = out.toByteArray()
    val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

    val matrix = android.graphics.Matrix()
    matrix.postRotate(rotationDegrees.toFloat())

    return android.graphics.Bitmap.createBitmap(
        bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
    )
}
