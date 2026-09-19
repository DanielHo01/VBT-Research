/*
 * EngineBridge.kt — JNI 桥接层
 * ==============================
 * 加载 libvbtcore_jni.so 并暴露 Kotlin API 调用 C++ 引擎。
 *
 * 支持两种模式：
 *   1. analyzeVideo() — 调用 nativeAnalyze (文件路径模式，桌面端使用 cv::VideoCapture)
 *   2. analyzeVideoFromUri() — 使用 MediaMetadataRetriever 提取帧，
 *      通过 nativeInitAnalyzer / nativeAddFrame / nativeFinalize
 *      (Android 端专用，绕过 cv::VideoCapture 缺失问题)
 */
package com.easyvbt.demo

import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer

object EngineBridge {
    private const val TAG = "EngineBridge"
    private const val LIB_NAME = "vbtcore_jni"

    init {
        try {
            System.loadLibrary(LIB_NAME)
            Log.i(TAG, "✓ $LIB_NAME.so 加载成功")
        } catch (t: Throwable) {
            Log.e(TAG, "✗ $LIB_NAME.so 加载失败", t)
            throw t
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // JNI 函数声明（对应 android/cpp/vbtcore_jni.cpp）
    // ══════════════════════════════════════════════════════════════════

    /** 文件路径模式（JNI vbt_analyze_video，桌面端 cv::VideoCapture） */
    @JvmStatic
    external fun nativeAnalyze(
        videoPath: String,
        modelPath: String,
        optionsJson: String,
    ): String?

    /** v2 段分割器（Java 传轨迹数组，native 执行峰值检测+自适应过零分段） */
    @JvmStatic
    external fun nativeSegmentV2(
        timestamps: DoubleArray,
        positions: DoubleArray,
        velocities: DoubleArray,
        mpp: Double,
        exerciseType: String,
    ): String

    /** 初始化帧分析会话 */
    @JvmStatic
    external fun nativeInitAnalyzer(
        fps: Double,
        modelPath: String,
        plateDiameterM: Double,
        exerciseType: String,
        redetEvery: Int,
        confThresh: Double,
    ): Boolean

    /** 喂入一帧 RGBA 数据（来自 Bitmap） */
    @JvmStatic
    external fun nativeAddFrame(
        rgbaData: ByteArray,
        width: Int,
        height: Int,
        strideBytes: Int,
        frameIdx: Int,
        timestampMs: Double,
    )

    /** 通过 DirectByteBuffer 喂入帧（零拷贝，推荐） */
    @JvmStatic
    external fun nativeAddFrameDirect(
        rgbaBuffer: ByteBuffer,
        width: Int,
        height: Int,
        strideBytes: Int,
        frameIdx: Int,
        timestampMs: Double,
    )

    /** 查询是否已完成标定 */
    @JvmStatic
    external fun nativeIsCalibrated(): Boolean

    /** 查询比例尺 m/px */
    @JvmStatic
    external fun nativeGetMpp(): Double

    /** 完成分析，返回 JSON 结果字符串 */
    @JvmStatic
    external fun nativeFinalize(): String

    // ══════════════════════════════════════════════════════════════════
    // 高层 API
    // ══════════════════════════════════════════════════════════════════

    /**
     * 文件路径模式 — 调用 nativeAnalyze（内部使用 cv::VideoCapture）。
     * Android 端 fallback 到 NOT_SUPPORTED_ON_ANDROID。
     */
    fun analyzeVideo(
        srcVideoPath: String,
        modelAssetPath: String,
        exerciseType: String = "squat_bench",
        plateDiameterM: Double = 0.45,
        outerPlate: String? = null,
    ): Result {
        val src = File(srcVideoPath)
        require(src.exists() && src.length() > 0) {
            "视频文件不存在或为空: $srcVideoPath"
        }

        val opts =
            JSONObject().apply {
                put("redet_every", 15)
                put("plate_diameter_m", plateDiameterM)
                put("exercise_type", exerciseType)
                if (outerPlate != null) put("outer_plate", outerPlate)
                put("conf_thresh", 0.40)
            }

        val jsonStr =
            nativeAnalyze(srcVideoPath, modelAssetPath, opts.toString())
                ?: throw IllegalStateException("nativeAnalyze 返回 null")

        return Result.fromJson(JSONObject(jsonStr))
    }

    /**
     * Android 帧提取模式 — 使用 MediaMetadataRetriever 提取帧，
     * 通过 JNI 传入 C++ 引擎（绕过 cv::VideoCapture 缺失问题）。
     *
     * @param videoUri  ContentResolver Uri（来自 ACTION_OPEN_DOCUMENT）
     * @param contentResolver 用于打开视频输入流
     * @param context Android Context（用于获取文件描述符）
     * @param modelPath ONNX 模型文件路径
     * @param exerciseType 练习类型："squat_bench" | "deadlift"
     * @param plateDiameterM 杠铃片直径（米）
     * @param frameStep 每隔 N 帧提取一帧（默认 3，对应 ~10fps@30fps 视频）
     */
    fun analyzeVideoFromUri(
        videoUri: android.net.Uri,
        contentResolver: android.content.ContentResolver,
        context: android.content.Context,
        modelPath: String,
        exerciseType: String = "squat_bench",
        plateDiameterM: Double = 0.45,
        outerPlate: String? = null,
        frameStep: Int = 3,
        progressCallback: ((framesExtracted: Int) -> Unit)? = null,
    ): Result {
        val retriever = MediaMetadataRetriever()
        var autoCloseFd: java.io.FileDescriptor? = null
        try {
            // file:// URIs: 先复制到 app 缓存目录（避免 scoped storage 权限问题）
            // content:// URIs: 用 FileDescriptor 模式
            if (videoUri.scheme == "file") {
                val srcPath =
                    videoUri.path
                        ?: throw IllegalStateException("file:// URI has no path: $videoUri")
                // 复制到 cache 目录
                val cacheFile = java.io.File(context.cacheDir, "debug_video.mp4")
                java.io.File(srcPath).inputStream().use { input ->
                    java.io.FileOutputStream(cacheFile).use { output ->
                        input.copyTo(output)
                    }
                }
                Log.i(TAG, "Copied video to cache: ${cacheFile.absolutePath}")
                val fd = java.io.FileInputStream(cacheFile).fd
                retriever.setDataSource(fd)
                autoCloseFd = fd // 循环结束后关闭
            } else {
                // content:// URI：尝试用 FileDescriptor
                val pfd =
                    try {
                        contentResolver.openFileDescriptor(videoUri, "r")
                    } catch (e: java.io.FileNotFoundException) {
                        // content:// 失效 → 降级：从 MediaStore 重新查 content URI
                        Log.w(TAG, "content:// URI 失效，尝试 MediaStore 降级")
                        val cursor =
                            contentResolver.query(
                                android.provider.MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                                arrayOf(android.provider.MediaStore.Video.Media._ID),
                                "${android.provider.MediaStore.Video.Media.DATA} LIKE ?",
                                arrayOf("%102.5kg_0.53_0.38%"),
                                null,
                            )
                        cursor
                            ?.use {
                                if (it.moveToFirst()) {
                                    val id = it.getLong(it.getColumnIndexOrThrow(android.provider.MediaStore.Video.Media._ID))
                                    android.net.Uri.withAppendedPath(
                                        android.provider.MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                                        id.toString(),
                                    )
                                } else {
                                    null
                                }
                            }?.let { newUri ->
                                Log.i(TAG, "MediaStore fallback URI: $newUri")
                                contentResolver.openFileDescriptor(newUri, "r")
                            }
                    }
                pfd ?: throw IllegalStateException("无法打开视频文件描述符: $videoUri")
                retriever.setDataSource(pfd.fileDescriptor)
                autoCloseFd = pfd.fileDescriptor
            }

            // 读取元数据
            val durationMsStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val fpsStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
            val rotationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)

            val durationMs = durationMsStr?.toLongOrNull() ?: 30_000L
            val fps = fpsStr?.toFloatOrNull() ?: 30.0f
            val rotationDeg = rotationStr?.toIntOrNull() ?: 0

            Log.i(TAG, "视频元数据: duration=${durationMs}ms, fps=$fps, rotation=$rotationDeg°")

            // 初始化 native 分析器
            // 关键：getFrameAtTime(OPTION_CLOSEST) 每次 decode 需要 ~300ms，
            //       这是提取速度的根本瓶颈（与 redetEvery 无关）
            // 修复：redetEvery=15 恢复原来已验证的速度，OPTION_CLOSEST 改善帧选择
            val redetEvery = 15
            val confThresh = 0.40
            val initOk =
                nativeInitAnalyzer(
                    fps.toDouble(),
                    modelPath,
                    plateDiameterM,
                    exerciseType,
                    redetEvery,
                    confThresh,
                )
            if (!initOk) {
                throw IllegalStateException("nativeInitAnalyzer 初始化失败")
            }
            Log.i(TAG, "nativeInitAnalyzer 成功 (fps=$fps, step=$frameStep)")

            // 帧提取策略
            // OPTION_CLOSEST: 取最近帧（I/P/B），但每次 seek+decode 需 300-500ms，太慢
            // OPTION_NEXT_SYNC: 取最近关键帧（I-frame），速度更快（~100ms/帧）
            // 实测：两者 YOLO detect 都能正常工作，OPTION_CLOSEST 运动信息更丰富但速度慢
            val frameDurationUs = (1_000_000L / fps).toLong()
            val stepUs = (frameDurationUs * 1).toLong()

            var frameIdx = 0
            var framesExtracted = 0
            var lastTimestampMs = -1.0
            var lastProgressReport = System.currentTimeMillis()

            var currentTimeUs = 0L

            while (currentTimeUs < durationMs * 1000) {
                // OPTION_CLOSEST: 取离 currentTimeUs 最近的视频帧（包含 I/P/B 帧）
                // 这是原来已验证的配置，OPTION_NEXT_SYNC 会导致同帧重复返回
                val bitmap =
                    retriever.getFrameAtTime(currentTimeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                        ?: break

                // 处理旋转
                val rotatedBitmap =
                    if (rotationDeg != 0) {
                        val matrix = Matrix().apply { postRotate(rotationDeg.toFloat()) }
                        Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true).also {
                            if (it != bitmap) bitmap.recycle()
                        }
                    } else {
                        bitmap
                    }

                // 提取 RGBA 像素
                val (width, height) = rotatedBitmap.width to rotatedBitmap.height
                val strideBytes = width * 4
                val pixels = IntArray(width * height)
                rotatedBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

                // ARGB → RGBA
                val rgbaBytes = ByteArray(strideBytes * height)
                for (i in pixels.indices) {
                    val px = pixels[i]
                    val offset = i * 4
                    rgbaBytes[offset + 0] = ((px shr 16) and 0xFF).toByte()
                    rgbaBytes[offset + 1] = ((px shr 8) and 0xFF).toByte()
                    rgbaBytes[offset + 2] = (px and 0xFF).toByte()
                    rgbaBytes[offset + 3] = ((px shr 24) and 0xFF).toByte()
                }

                // 计算时间戳（用实际帧位置而非请求时间，精确反映帧真实 PTS）
                val timestampMs = currentTimeUs / 1000.0

                // 跳过与上一帧时间戳相同的情况（某些视频含重复 PTS）
                if (timestampMs > lastTimestampMs + 1.0) {
                    nativeAddFrame(rgbaBytes, width, height, strideBytes, frameIdx, timestampMs)
                    framesExtracted++
                    frameIdx += frameStep
                    lastTimestampMs = timestampMs
                }

                if (rotatedBitmap != bitmap) rotatedBitmap.recycle()
                bitmap.recycle()

                // 推进到下一个同步帧（时间增量 = stepUs）
                currentTimeUs += stepUs

                // 每 0.5s 报告一次进度
                val now = System.currentTimeMillis()
                if (now - lastProgressReport > 500) {
                    val progress = (currentTimeUs * 100 / (durationMs * 1000)).coerceIn(0, 100).toInt()
                    Log.i(TAG, "帧提取进度: $framesExtracted 帧 ($progress%)")
                    progressCallback?.invoke(framesExtracted)
                    lastProgressReport = now
                }
            }

            Log.i(TAG, "帧提取完成: $framesExtracted 帧")

            // 检查标定状态
            val calibrated = nativeIsCalibrated()
            val mpp = nativeGetMpp()
            Log.i(TAG, "标定状态: calibrated=$calibrated, mpp=$mpp")

            // 完成分析
            val jsonStr =
                nativeFinalize()
                    ?: throw IllegalStateException("nativeFinalize 返回 null")

            Log.i(TAG, "nativeFinalize JSON: $jsonStr")

            return Result.fromJson(
                JSONObject(jsonStr).apply {
                    put("fps", fps.toDouble())
                    put("frames_extracted", framesExtracted)
                },
            )
        } finally {
            try {
                retriever.release()
            } catch (_: Exception) {
                // ignore
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // 数据结构
    // ══════════════════════════════════════════════════════════════════

    /**
     * 引擎结果（解析自 JSON）。
     */
    data class Result(
        val status: String,
        val video: String,
        val fps: Double,
        val mpp: Double?,
        val reps: List<Rep>,
        val diagnostics: Map<String, Any?>,
    ) {
        val isOk: Boolean get() = status == "OK"

        companion object {
            fun fromJson(json: JSONObject): Result {
                val repsArr = json.optJSONArray("reps")
                val reps = mutableListOf<Rep>()
                if (repsArr != null) {
                    for (i in 0 until repsArr.length()) {
                        reps.add(Rep.fromJson(repsArr.getJSONObject(i)))
                    }
                }
                @Suppress("UNCHECKED_CAST")
                val diagMap =
                    json.optJSONObject("diagnostics")?.let { obj ->
                        obj.keys().asSequence().associateWith { obj.get(it) }
                    } ?: emptyMap()
                return Result(
                    status = json.optString("status", "UNKNOWN"),
                    video = json.optString("video", ""),
                    fps = json.optDouble("fps", 30.0),
                    mpp = if (json.has("mpp")) json.optDouble("mpp") else null,
                    reps = reps,
                    diagnostics = diagMap,
                )
            }
        }
    }

    /**
     * 单个 rep 结果。
     */
    data class Rep(
        val startIdx: Int,
        val endIdx: Int,
        val startTime: Double,
        val endTime: Double,
        val durationS: Double,
        val romM: Double,
        val mcvMps: Double,
        val pcvMps: Double,
    ) {
        companion object {
            fun fromJson(json: JSONObject): Rep =
                Rep(
                    startIdx = json.optInt("start_idx"),
                    endIdx = json.optInt("end_idx"),
                    startTime = json.optDouble("start_time", 0.0),
                    endTime = json.optDouble("end_time", 0.0),
                    durationS = json.optDouble("duration_s", 0.0),
                    romM = json.optDouble("rom_m", 0.0),
                    mcvMps = json.optDouble("mcv_mps", 0.0),
                    pcvMps = json.optDouble("pcv_mps", 0.0),
                )
        }
    }
}
