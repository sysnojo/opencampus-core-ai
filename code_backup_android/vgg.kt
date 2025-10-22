package com.example.opencampus.ui.scanner

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.Handler
import android.os.Looper
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
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.sqrt

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
    var lastBitmap by remember { mutableStateOf<Bitmap?>(null) }

    val executor = remember { Executors.newSingleThreadExecutor() }
    val previewView = remember { PreviewView(context) }

    val labels = listOf(
        "ruang_dosen_1_10m", "ruang_dosen_1_15m", "ruang_dosen_1_20m",
        "ruang_dosen_1_25m", "ruang_dosen_1_30m", "ruang_dosen_1_35m",
        "ruang_dosen_1_40m", "ruang_dosen_1_45m", "ruang_dosen_1_5m",
        "ruang_dosen_2_10m", "ruang_dosen_2_15m", "ruang_dosen_2_20m",
        "ruang_dosen_2_25m", "ruang_dosen_2_30m", "ruang_dosen_2_35m",
        "ruang_dosen_2_40m", "ruang_dosen_2_5m"
    )

    LaunchedEffect(Unit) {
        try {
            // === Load model ===
            val modelName = "openibl_lite_vgg16.pt"
            val internalFile = File(context.filesDir, modelName)
            if (!internalFile.exists()) {
                context.assets.open(modelName).use { input ->
                    internalFile.outputStream().use { output -> input.copyTo(output) }
                }
            }
            val module = LiteModuleLoader.load(internalFile.absolutePath)

            // === Load descriptor ===
            val descFileName = "descriptor_cache.npz"
            val descFile = File(context.filesDir, descFileName)
            if (!descFile.exists()) {
                context.assets.open(descFileName).use { input ->
                    descFile.outputStream().use { output -> input.copyTo(output) }
                }
            }
            val descriptors = loadNpz(descFile)

            labelText = "Model & descriptor loaded ✅"

            // === Camera ===
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build()
                val selector = CameraSelector.DEFAULT_BACK_CAMERA
                preview.setSurfaceProvider(previewView.surfaceProvider)

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(224, 224))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                analysis.setAnalyzer(executor, ImageAnalysis.Analyzer { proxy ->
                    val bitmap = proxy.toBitmap()
                    if (bitmap != null) {
                        try {
                            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                                Bitmap.createScaledBitmap(bitmap, 224, 224, true),
                                floatArrayOf(0.485f, 0.456f, 0.406f),
                                floatArrayOf(0.229f, 0.224f, 0.225f)
                            )

                            val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
                            val features = outputTensor.dataAsFloatArray

                            // Jalankan di thread terpisah biar tidak block CameraX
                            Executors.newSingleThreadExecutor().execute {
                                val subset = descriptors.shuffled().take(200)
                                val similarities = subset.map { cosineSimilarity(features, it) }
                                val maxIdx = similarities.indices.maxByOrNull { similarities[it] } ?: 0
                                val similarity = similarities[maxIdx]

                                Handler(Looper.getMainLooper()).post {
                                    labelText = "Predicted: ${labels[maxIdx]} (${String.format("%.2f", similarity)})"
                                }
                            }

                        } catch (e: Exception) {
                            labelText = "Error: ${e.message}"
                        }
                    }
                    proxy.close()
                })

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(context as ComponentActivity, selector, preview, analysis)
            }, ContextCompat.getMainExecutor(context))
        } catch (e: Exception) {
            labelText = "Failed: ${e.message}"
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(20.dp)
                .background(Color.Black.copy(alpha = 0.5f))
                .padding(12.dp)
        ) {
            Text(
                text = labelText,
                color = Color.White,
                fontWeight = FontWeight.Medium,
                fontSize = 14.sp
            )
        }
    }
}

// === Helper Functions ===

/** Load NPZ file dari assets **/
fun loadNpz(file: File): List<FloatArray> {
    val zis = java.util.zip.ZipInputStream(FileInputStream(file))
    val descList = mutableListOf<FloatArray>()
    var entry = zis.nextEntry
    while (entry != null) {
        val bytes = zis.readBytes()
        val floats = parseNpyFloatArray(bytes)
        descList.add(floats)
        zis.closeEntry()
        entry = zis.nextEntry
    }
    zis.close()
    return descList
}

/** Parse satu file .npy ke FloatArray **/
fun parseNpyFloatArray(bytes: ByteArray): FloatArray {
    val magic = byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte())
    val headerEnd = bytes.indexOf(0x0A)
    val headerStr = bytes.copyOfRange(magic.size + 2, headerEnd).toString(Charsets.UTF_8)
    val shapeRegex = "\\((.*?)\\)".toRegex()
    val shape = shapeRegex.find(headerStr)?.groupValues?.get(1)?.split(",")?.mapNotNull { it.trim().toIntOrNull() } ?: listOf()
    val dataOffset = headerEnd + 1
    val bb = ByteBuffer.wrap(bytes, dataOffset, bytes.size - dataOffset).order(ByteOrder.LITTLE_ENDIAN)
    val floatArr = FloatArray(bb.remaining() / 4)
    bb.asFloatBuffer().get(floatArr)
    return floatArr
}

/** Hitung cosine similarity **/
fun cosineSimilarity(a: FloatArray, b: FloatArray): Double {
    val dot = a.zip(b) { x, y -> x * y }.sum()
    val normA = sqrt(a.map { it * it }.sum().toDouble())
    val normB = sqrt(b.map { it * it }.sum().toDouble())
    return dot / (normA * normB)
}

/** Konversi ImageProxy ke Bitmap **/
private fun ImageProxy.toBitmap(): Bitmap? {
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
    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
    val imageBytes = out.toByteArray()
    var bmp = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    val matrix = Matrix()
    matrix.postRotate(imageInfo.rotationDegrees.toFloat())
    bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
    return bmp
}
