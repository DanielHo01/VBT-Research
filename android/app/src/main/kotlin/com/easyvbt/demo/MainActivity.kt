/*
 * MainActivity.kt — Demo 入口（Compose UI）
 * =========================================
 * 用户流程：
 *   1. 点击「选择视频」按钮 → 系统文件选择器（ACTION_OPEN_DOCUMENT）
 *   2. 选 mp4 → 拷贝到应用沙箱 → 调 EngineBridge.analyzeVideo
 *   3. 进度条 + 取消
 *   4. 结果屏：rep 列表 + 速度曲线（Canvas 自绘）
 */
package com.easyvbt.demo

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.database.Cursor
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

private const val TAG = "EasyVBT/Demo"

class ResultViewModel : ViewModel() {
    sealed class State {
        object Idle : State()

        data class Loading(
            val videoName: String,
        ) : State()

        data class Success(
            val result: EngineBridge.Result,
        ) : State()

        data class Error(
            val message: String,
        ) : State()
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    /** 从 assets 提取 best.onnx 到 filesDir，返回绝对路径 */
    private fun extractModelAsset(context: android.content.Context): File {
        val modelFile = File(context.filesDir, "best.onnx")
        if (modelFile.exists() && modelFile.length() > 0) {
            Log.i(TAG, "模型已存在: ${modelFile.absolutePath} (${modelFile.length() / 1024} KB)")
            return modelFile
        }
        Log.i(TAG, "从 assets 提取 best.onnx...")
        context.assets.open("best.onnx").use { input ->
            FileOutputStream(modelFile).use { os ->
                input.copyTo(os)
            }
        }
        Log.i(TAG, "模型提取完成: ${modelFile.absolutePath} (${modelFile.length() / 1024} KB)")
        return modelFile
    }

    fun analyze(
        srcUri: Uri,
        contentResolver: android.content.ContentResolver,
        context: android.content.Context,
    ) {
        viewModelScope.launch {
            try {
                _state.value = State.Loading(srcUri.lastPathSegment ?: "video.mp4")
                // 1. 提取模型（从 assets 到文件系统）
                val modelFile =
                    withContext(Dispatchers.IO) {
                        extractModelAsset(context)
                    }
                // 2. 使用 MediaMetadataRetriever 提取帧 → JNI → C++ 引擎
                val name = srcUri.lastPathSegment ?: "video.mp4"
                val result =
                    withContext(Dispatchers.Default) {
                        EngineBridge.analyzeVideoFromUri(
                            videoUri = srcUri,
                            contentResolver = contentResolver,
                            context = context,
                            modelPath = modelFile.absolutePath,
                            exerciseType = "squat_bench",
                            plateDiameterM = 0.45,
                            frameStep = 3, // 每隔 3 帧取一帧（30fps 视频 → ~10fps）
                        )
                    }
                _state.value = State.Success(result.copy(video = name))
            } catch (t: Throwable) {
                Log.e(TAG, "analyze failed", t)
                _state.value = State.Error(t.message ?: "未知错误")
            }
        }
    }

    fun reset() {
        _state.value = State.Idle
    }

    /** 把 ContentResolver Uri 拷贝到 cacheDir，返回 (File, 显示名) */
    private fun copyToCache(
        uri: Uri,
        cr: android.content.ContentResolver,
    ): Pair<File, String> {
        val name = queryDisplayName(uri, cr) ?: "input.mp4"
        val safe = name.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val out = File(System.getProperty("java.io.tmpdir") ?: ".", safe)
        cr.openInputStream(uri)?.use { input ->
            FileOutputStream(out).use { os ->
                input.copyTo(os)
            }
        } ?: throw IllegalStateException("无法打开输入流: $uri")
        return out to name
    }

    private fun queryDisplayName(
        uri: Uri,
        cr: android.content.ContentResolver,
    ): String? {
        val cursor: Cursor? = cr.query(uri, null, null, null, null)
        cursor?.use {
            if (it.moveToFirst()) {
                val idx = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (idx >= 0) return it.getString(idx)
            }
        }
        return null
    }
}

class MainActivity : ComponentActivity() {
    private val viewModel: ResultViewModel by viewModels()

    // Activity 级别注册，生命周期稳定，不受 Composable 重组合影响
    private val pickLauncher =
        registerForActivityResult(
            contract = ActivityResultContracts.StartActivityForResult(),
        ) { result ->
            Log.i(TAG, "ActivityResult: resultCode=${result.resultCode} RESULT_OK=${Activity.RESULT_OK}")
            if (result.resultCode == Activity.RESULT_OK) {
                val uri = result.data?.data
                Log.i(TAG, "ActivityResult: uri=$uri")
                if (uri != null) {
                    // 用户从 SAF 选择视频后，尝试获取持久化权限
                    try {
                        val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
                        contentResolver.takePersistableUriPermission(uri, flags)
                        Log.i(TAG, "持久化读权限获取成功: $uri")
                    } catch (e: SecurityException) {
                        Log.w(TAG, "持久化读权限获取失败（单次访问）: ${e.message}")
                    }
                    viewModel.analyze(uri, contentResolver, this@MainActivity)
                } else {
                    Log.w(TAG, "ActivityResult: uri is null")
                }
            } else {
                Log.w(TAG, "ActivityResult: resultCode=${result.resultCode} (not OK, user cancelled?)")
            }
        }

    // 权限请求回调
    private val mediaPermissionLauncher =
        registerForActivityResult(
            contract = ActivityResultContracts.RequestMultiplePermissions(),
        ) { results ->
            val granted = results.entries.any { it.value }
            Log.i(TAG, "媒体权限申请结果: $results granted=$granted")
            if (granted) {
                // 权限获取成功后，分析调试视频
                val debugVideoUri = Uri.parse("file:///storage/emulated/0/Download/102.5kg_0.53_0.38.mp4")
                viewModel.analyze(debugVideoUri, contentResolver, this@MainActivity)
            } else {
                _state.value = ResultViewModel.State.Error("需要 READ_MEDIA_VIDEO 权限才能读取视频")
            }
        }

    private var _state: MutableStateFlow<ResultViewModel.State> = MutableStateFlow(ResultViewModel.State.Idle)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 检查并申请媒体读取权限
        val mediaPerm =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                Manifest.permission.READ_MEDIA_VIDEO
            } else {
                Manifest.permission.READ_EXTERNAL_STORAGE
            }
        when {
            ContextCompat.checkSelfPermission(this, mediaPerm) == PackageManager.PERMISSION_GRANTED -> {
                Log.i(TAG, "媒体权限已授权，直接开始分析")
                val debugVideoUri = Uri.parse("file:///storage/emulated/0/Download/102.5kg_0.53_0.38.mp4")
                viewModel.analyze(debugVideoUri, contentResolver, this@MainActivity)
            }

            else -> {
                Log.i(TAG, "请求媒体权限: $mediaPerm")
                mediaPermissionLauncher.launch(arrayOf(mediaPerm))
            }
        }

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    DemoScreen(viewModel) { intent ->
                        pickLauncher.launch(intent)
                    }
                }
            }
        }
    }
}

@Composable
private fun DemoScreen(
    viewModel: ResultViewModel,
    onLaunchPicker: (Intent) -> Unit,
) {
    val state by viewModel.state.collectAsState()

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text(
            "EasyVBT Demo",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "VBT 速度分析器（离线导入 mp4）",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontSize = 14.sp,
        )
        Spacer(Modifier.height(24.dp))

        when (val s = state) {
            is ResultViewModel.State.Idle -> {
                IdleView {
                    val intent =
                        Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                            addCategory(Intent.CATEGORY_OPENABLE)
                            type = "video/*"
                            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("video/mp4"))
                        }
                    onLaunchPicker(intent)
                }
            }

            is ResultViewModel.State.Loading -> {
                LoadingView(s.videoName)
            }

            is ResultViewModel.State.Success -> {
                ResultView(s.result) {
                    viewModel.reset()
                }
            }

            is ResultViewModel.State.Error -> {
                ErrorView(s.message) {
                    viewModel.reset()
                }
            }
        }
    }
}

@Composable
private fun IdleView(onPick: () -> Unit) {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Button(onClick = onPick) {
                Text("选择视频", fontSize = 16.sp)
            }
            Spacer(Modifier.height(8.dp))
            Text(
                "支持 mp4 格式（建议 ≤ 60s, 720p）",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun LoadingView(videoName: String) {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        CircularProgressIndicator()
        Spacer(Modifier.height(16.dp))
        Text("正在分析…", fontSize = 16.sp)
        Spacer(Modifier.height(8.dp))
        Text(videoName, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun ErrorView(
    message: String,
    onRetry: () -> Unit,
) {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("❌ 分析失败", color = MaterialTheme.colorScheme.error, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(8.dp))
            Text(message, fontSize = 14.sp)
            Spacer(Modifier.height(16.dp))
            Button(onClick = onRetry) { Text("重试") }
        }
    }
}

@Composable
private fun ResultView(
    result: EngineBridge.Result,
    onReset: () -> Unit,
) {
    Column(modifier = Modifier.fillMaxSize()) {
        Card(modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp)) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    "状态：${result.status}",
                    fontWeight = FontWeight.Bold,
                    color =
                        if (result.isOk) {
                            Color(0xFF2E7D32)
                        } else {
                            MaterialTheme.colorScheme.error
                        },
                )
                Text("视频：${result.video}", fontSize = 12.sp)
                Text(
                    "FPS：${"%.1f".format(result.fps)}",
                    fontSize = 12.sp,
                )
                result.mpp?.let { Text("比例尺：${"%.4f".format(it)} m/px", fontSize = 12.sp) }
                @Suppress("UNCHECKED_CAST")
                (result.diagnostics["elapsed_s"] as? Number)?.let {
                    Text("耗时：${"%.1f".format(it.toDouble())}s", fontSize = 12.sp)
                }
            }
        }

        if (result.reps.isNotEmpty()) {
            VelocityChart(reps = result.reps, modifier = Modifier.fillMaxWidth().height(160.dp).padding(vertical = 8.dp))
            Text(
                "Rep 列表（${result.reps.size} 个）",
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(top = 8.dp),
            )
            LazyColumn(modifier = Modifier.fillMaxWidth().weight(1f)) {
                items(result.reps) { rep ->
                    RepRow(rep)
                }
            }
        } else {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("未检测到 rep", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }

        Button(onClick = onReset, modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
            Text("选择另一个视频")
        }
    }
}

@Composable
private fun RepRow(rep: EngineBridge.Rep) {
    Card(modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "#${rep.startIdx}-${rep.endIdx}",
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier.width(80.dp),
            )
            Spacer(Modifier.width(8.dp))
            Text(
                "MCV=${"%.2f".format(rep.mcvMps)} m/s",
                fontWeight = FontWeight.Medium,
            )
            Spacer(Modifier.width(16.dp))
            Text(
                "ROM=${"%.2f".format(rep.romM * 100)}cm",
                fontSize = 13.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.width(16.dp))
            Text(
                "峰=${"%.2f".format(rep.pcvMps)}",
                fontSize = 13.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun VelocityChart(
    reps: List<EngineBridge.Rep>,
    modifier: Modifier = Modifier,
) {
    val color = MaterialTheme.colorScheme.primary
    Canvas(modifier = modifier.background(MaterialTheme.colorScheme.surfaceVariant)) {
        if (reps.isEmpty()) return@Canvas
        val w = size.width
        val h = size.height
        val pad = 16f

        val maxVel = reps.maxOf { it.pcvMps }.coerceAtLeast(0.1)
        val xStep = (w - 2 * pad) / (reps.size + 1)

        // 网格线
        for (i in 0..4) {
            val y = pad + (h - 2 * pad) * i / 4
            drawLine(
                color = Color.LightGray,
                start = Offset(pad, y),
                end = Offset(w - pad, y),
                strokeWidth = 1f,
            )
        }
        // 速度折线
        val path = Path()
        reps.forEachIndexed { i, rep ->
            val x = pad + xStep * (i + 1)
            val y = h - pad - ((rep.pcvMps / maxVel) * (h - 2 * pad)).toFloat()
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            drawCircle(color = color, radius = 6f, center = Offset(x, y))
        }
        drawPath(
            path = path,
            color = color,
            style =
                androidx.compose.ui.graphics.drawscope
                    .Stroke(width = 3f),
        )
    }
}
