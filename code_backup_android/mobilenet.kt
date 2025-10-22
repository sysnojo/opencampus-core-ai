package com.example.opencampus.ui.scanner

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
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
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
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
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.updateTransition
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import kotlin.math.abs
import com.example.opencampus.core.utils.loadNpz


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
    var selectedRoom by remember { mutableStateOf("ruang_dosen_1") }

    // State UI
    var phase by remember { mutableStateOf("hold") }
    var showMainUI by remember { mutableStateOf(false) }
    var userLocation by remember { mutableStateOf<String?>(null) }

    // Kamera & model
    var lastBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var movementDetected by remember { mutableStateOf(false) }

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

    // === Load Model + Kamera (MobileVNET2) ===
//    LaunchedEffect(Unit) {
//        try {
//            val modelName = "openibl_vgg16_netvlad_lite.pt"
//            val internalFile = File(context.filesDir, modelName)
//            if (!internalFile.exists()) {
//                context.assets.open(modelName).use { input ->
//                    internalFile.outputStream().use { output -> input.copyTo(output) }
//                }
//            }
//
//            val module = LiteModuleLoader.load(internalFile.absolutePath)
//            labelText = "Model loaded"
//
//            cameraProviderFuture.addListener({
//                val cameraProvider = cameraProviderFuture.get()
//                val preview = Preview.Builder().build()
//                val selector = CameraSelector.DEFAULT_BACK_CAMERA
//                preview.setSurfaceProvider(previewView.surfaceProvider)
//
//                val analysis = ImageAnalysis.Builder()
//                    .setTargetResolution(android.util.Size(224, 224))
//                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                    .build()
//
//                analysis.setAnalyzer(executor, ImageAnalysis.Analyzer { proxy ->
//                    val bitmap = proxy.toBitmap()
//                    if (bitmap != null) {
//                        val moved = detectMovement(lastBitmap, bitmap)
//                        if (moved && !movementDetected) {
//                            movementDetected = true
//                            Handler(Looper.getMainLooper()).postDelayed({
//                                phase = "scanning"
//                            }, 1500)
//                        }
//                        lastBitmap = bitmap
//
//                        val tensor = TensorImageUtils.bitmapToFloat32Tensor(
//                            Bitmap.createScaledBitmap(bitmap, 224, 224, true),
//                            floatArrayOf(0.485f, 0.456f, 0.406f),
//                            floatArrayOf(0.229f, 0.224f, 0.225f)
//                        )
//                        val output = module.forward(IValue.from(tensor)).toTensor()
//                        val scores = output.dataAsFloatArray
//                        val (maxIdx, maxProb) = scores.withIndex().maxByOrNull { it.value }!!
//                        val detectedLabel = labels[maxIdx]
//                        labelText = "${detectedLabel.replace('_', ' ')} (${(maxProb * 100).toInt()}%)"
//
//                        // Jika terdeteksi dekat → transisi fade ke main UI
//                        if ((detectedLabel == "ruang_dosen_1_5m" || detectedLabel == "ruang_dosen_2_5m")
//                            && phase == "scanning") {
//                            userLocation = "Selasar Lantai 4"
//
//                            // Delay untuk memberi waktu progress bar selesai → lalu fade keluar
//                            Handler(Looper.getMainLooper()).postDelayed({
//                                phase = "fadeout"
//                            }, 1500)
//
//                            // Setelah fade-out selesai baru tampilkan UI utama
//                            Handler(Looper.getMainLooper()).postDelayed({
//                                showMainUI = true
//                            }, 2500)
//                        }
//                    }
//                    proxy.close()
//                })
//
//                cameraProvider.unbindAll()
//                cameraProvider.bindToLifecycle(
//                    context as ComponentActivity,
//                    selector,
//                    preview,
//                    analysis
//                )
//            }, ContextCompat.getMainExecutor(context))
//        } catch (e: Exception) {
//            labelText = "Model failed: ${e.message}"
//            e.printStackTrace()
//        }
//    }
    // === Load model & descriptor cache (OpenIBL) ===
    LaunchedEffect(Unit) {
        try {
            val modelName = "openibl_vgg16_netvlad_lite.pt"
            val descriptorCacheName = "descriptor_cache.npz"
            val internalModelFile = File(context.filesDir, modelName)
            val internalCacheFile = File(context.filesDir, descriptorCacheName)

            // Salin model dari assets → internal storage
            if (!internalModelFile.exists()) {
                context.assets.open(modelName).use { input ->
                    internalModelFile.outputStream().use { output -> input.copyTo(output) }
                }
            }

            // Salin descriptor cache juga
            if (!internalCacheFile.exists()) {
                context.assets.open(descriptorCacheName).use { input ->
                    internalCacheFile.outputStream().use { output -> input.copyTo(output) }
                }
            }

            val module = LiteModuleLoader.load(internalModelFile.absolutePath)
            labelText = "Model loaded"

            // === Load descriptor cache (.npz) ===
            val npzData = loadNpz(internalCacheFile)
            val descriptors = npzData.descriptors
            val imagePaths = npzData.imagePaths

            // === Setup Camera ===
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build()
                val selector = CameraSelector.DEFAULT_BACK_CAMERA
                preview.setSurfaceProvider(previewView.surfaceProvider)

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(480, 640))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                analysis.setAnalyzer(executor, ImageAnalysis.Analyzer { proxy ->
                    val bitmap = proxy.toBitmap()
                    if (bitmap != null) {
                        // ==== Nonaktifkan movement detection untuk test ====
                         val moved = detectMovement(lastBitmap, bitmap)
                         if (moved && !movementDetected) {
                             movementDetected = true
                             Handler(Looper.getMainLooper()).postDelayed({
                                 phase = "scanning"
                             }, 1500)
                         }
                         lastBitmap = bitmap

                        try {
                            // === Ubah ke tensor ===
                            val tensor = TensorImageUtils.bitmapToFloat32Tensor(
                                Bitmap.createScaledBitmap(bitmap, 640, 480, true),
                                floatArrayOf(0.485f, 0.458f, 0.408f),
                                floatArrayOf(0.0039f, 0.0039f, 0.0039f)
                            )

                            // === Ekstraksi descriptor ===
                            val output = module.forward(IValue.from(tensor)).toTensor()
                            val queryDescriptor = output.dataAsFloatArray

                            // === Hitung cosine similarity ===
                            var bestScore = -1f
                            var bestLabel = "Unknown"
                            for (i in imagePaths.indices) {
                                val score = cosineSimilarity(queryDescriptor, descriptors[i])
                                if (score > bestScore) {
                                    bestScore = score
                                    bestLabel = imagePaths[i]
                                }
                            }

                            labelText = "Best match: ${bestLabel.substringAfterLast("\\")}\nScore: ${String.format("%.3f", bestScore)}"

                            // ✅ Berhenti scanning setelah dapat hasil
                            cameraProviderFuture.get().unbindAll()
                        } catch (e: Exception) {
                            labelText = "Error: ${e.message}"
                        }
                    }
                    proxy.close()
                })


                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    context as ComponentActivity,
                    selector,
                    preview,
                    analysis
                )
            }, ContextCompat.getMainExecutor(context))
        } catch (e: Exception) {
            labelText = "Model failed: ${e.message}"
            e.printStackTrace()
        }
    }

    // === Animasi Transisi ===
    val alpha by animateFloatAsState(
        targetValue = when (phase) {
            "hold" -> 0.85f
            "scanning" -> 0.85f
            "fadeout" -> 0f
            else -> 0f
        },
        animationSpec = tween(700)
    )

    val uiAlpha by animateFloatAsState(
        targetValue = if (showMainUI) 1f else 0f,
        animationSpec = tween(700)
    )

    // === UI ===
    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

        // Overlay (Hold → Scanning → Fade out)
        if (phase != "done" && alpha > 0f) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = alpha)),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    val text = when (phase) {
                        "hold" -> "Hold vertically to detect location"
                        "scanning" -> "Scanning your location..."
                        else -> ""
                    }
                    Text(
                        text = text,
                        color = Color.White,
                        fontWeight = FontWeight.Medium,
                        fontSize = 13.sp
                    )
                    if (phase == "scanning") {
                        Spacer(modifier = Modifier.height(12.dp))
                        CircularProgressIndicator(
                            color = Color.White,
                            strokeWidth = 2.dp,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }
        }

        // Fade-in UI utama setelah scanning selesai
        if (showMainUI && userLocation != null) {
            // ---- Modified UI ----
//            LocationBottomSheetUI()

            
            // ---- FOR MODEL ----
            Column(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 60.dp)
                    .graphicsLayer(alpha = uiAlpha),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Your location: ${userLocation!!}",
                    color = Color.White,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(modifier = Modifier.height(10.dp))
                Text("Pilih Ruang Dosen:", color = Color.White, fontSize = 14.sp)
                RoomSelector(selectedRoom) { selectedRoom = it }
            }

            Text(
                text = labelText,
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 14.sp,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 40.dp)
                    .graphicsLayer(alpha = uiAlpha)
            )
        }
    }
}

// ==== UI ====
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LocationBottomSheetUI() {
    val sheetState = rememberStandardBottomSheetState(
        initialValue = SheetValue.PartiallyExpanded,
        skipHiddenState = true
    )
    val scaffoldState = rememberBottomSheetScaffoldState(bottomSheetState = sheetState)

    BottomSheetScaffold(
        scaffoldState = scaffoldState,
        sheetPeekHeight = 180.dp,
        sheetContent = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp, vertical = 16.dp)
            ) {
                // Handle bar (strip kecil di atas)
                Box(
                    modifier = Modifier
                        .align(Alignment.CenterHorizontally)
                        .width(40.dp)
                        .height(5.dp)
                        .clip(RoundedCornerShape(50))
                        .background(Color.Gray.copy(alpha = 0.4f))
                )
                Spacer(modifier = Modifier.height(16.dp))

                // Header - Ilustrasi + teks lokasi
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Box(
                        modifier = Modifier
                            .size(60.dp)
                            .clip(RoundedCornerShape(16.dp))
                            .background(Color(0xFF1E1E1E)),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("🧭", fontSize = 26.sp)
                    }
                    Spacer(modifier = Modifier.width(14.dp))
                    Column {
                        Text(
                            "Ruang Dosen 1",
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp,
                            color = Color.White
                        )
                        Text(
                            "Selasar Lantai 4",
                            fontSize = 13.sp,
                            color = Color.LightGray
                        )
                    }
                }

                Spacer(modifier = Modifier.height(20.dp))
                Divider(color = Color.Gray.copy(alpha = 0.2f))
                Spacer(modifier = Modifier.height(12.dp))

                // Opsi-opsi (seperti Gojek UI)
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    SheetOptionItem(icon = "📍", title = "Scan lagi")
                    SheetOptionItem(icon = "🏫", title = "Pilih lokasi lain")
                    SheetOptionItem(icon = "ℹ️", title = "Detail lokasi ini")
                }

                Spacer(modifier = Modifier.height(40.dp))
            }
        },
        sheetContainerColor = Color(0xFF111111),
        sheetShape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)
    ) {
        // Konten utama sementara dikosongin (kamera dll dikomen)
//        Box(
//            modifier = Modifier
//                .fillMaxSize()
//                .background(Color.Black),
//            contentAlignment = Alignment.Center
//        ) {
//            Text(
//                "Camera View Placeholder",
//                color = Color.White.copy(alpha = 0.5f),
//                fontSize = 13.sp
//            )
//        }
    }
}

fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
    require(a.size == b.size) { "Arrays must have the same length" }

    var dot = 0.0
    var normA = 0.0
    var normB = 0.0
    for (i in a.indices) {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    return (dot / (Math.sqrt(normA) * Math.sqrt(normB))).toFloat()
}



@Composable
fun SheetOptionItem(icon: String, title: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .clickable { /* aksi nanti */ }
            .background(Color(0xFF1C1C1C))
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(icon, fontSize = 18.sp)
        Spacer(modifier = Modifier.width(12.dp))
        Text(
            title,
            color = Color.White,
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium
        )
    }
}



/** Deteksi gerakan sederhana berdasarkan perbedaan rata-rata piksel */
private fun detectMovement(prev: Bitmap?, current: Bitmap): Boolean {
    if (prev == null) return false
    val resizedPrev = Bitmap.createScaledBitmap(prev, 64, 64, true)
    val resizedCurr = Bitmap.createScaledBitmap(current, 64, 64, true)

    var diffSum = 0L
    val pixels1 = IntArray(64 * 64)
    val pixels2 = IntArray(64 * 64)
    resizedPrev.getPixels(pixels1, 0, 64, 0, 0, 64, 64)
    resizedCurr.getPixels(pixels2, 0, 64, 0, 0, 64, 64)

    for (i in pixels1.indices) {
        val r1 = (pixels1[i] shr 16) and 0xFF
        val g1 = (pixels1[i] shr 8) and 0xFF
        val b1 = pixels1[i] and 0xFF
        val r2 = (pixels2[i] shr 16) and 0xFF
        val g2 = (pixels2[i] shr 8) and 0xFF
        val b2 = pixels2[i] and 0xFF
        diffSum += (abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2))
    }

    val avgDiff = diffSum / (64 * 64)
    return avgDiff > 15 // threshold pergerakan
}


@Composable
fun RoomSelector(selected: String, onSelectedChange: (String) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    val options = listOf("ruang_dosen_1", "ruang_dosen_2")

    Box {
        Button(onClick = { expanded = true }) {
            Text(
                text = selected.replace('_', ' ').uppercase(),
                fontSize = 12.sp
            )
        }
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            options.forEach { option ->
                DropdownMenuItem(
                    text = { Text(option.replace('_', ' ').uppercase()) },
                    onClick = {
                        onSelectedChange(option)
                        expanded = false
                    }
                )
            }
        }
    }
}

// === Utility: Convert ImageProxy → Bitmap ===
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
    val yuvImage = android.graphics.YuvImage(
        nv21, android.graphics.ImageFormat.NV21, width, height, null
    )
    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 90, out)
    val imageBytes = out.toByteArray()
    var bmp = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    val matrix = Matrix()
    matrix.postRotate(imageInfo.rotationDegrees.toFloat())
    bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
    return bmp
}
