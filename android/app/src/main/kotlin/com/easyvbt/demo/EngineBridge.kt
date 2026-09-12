/*
 * EngineBridge.kt — JNI 桥接层
 * ==============================
 * 加载 libvbtcore_jni.so 并暴露 Kotlin API 调用 C++ 引擎。
 * 输入：mp4 文件路径 + 模型路径
 * 输出：JSON 字符串（含 rep 列表 + 诊断）
 *
 * 与 C++ 端 vbtcore-cpp/src/analyze.cpp::vbt_analyze_video 对接。
 */
package com.easyvbt.demo

import android.util.Log
import org.json.JSONObject
import java.io.File

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

    /** 调用 C++ 引擎，返回 JSON 字符串 */
    @JvmStatic
    external fun nativeAnalyze(
        videoPath: String,
        modelPath: String,
        optionsJson: String,
    ): String?

    /**
     * 高层 Kotlin 包装：从 ContentResolver 选择的视频 Uri 拷贝到
     * 应用沙箱，再调用 native 分析，返回结构化结果。
     */
    fun analyzeVideo(
        srcVideoPath: String,
        modelAssetPath: String,
        exerciseType: String = "squat_bench",
        plateDiameterM: Double = 0.45,
        outerPlate: String? = null,
    ): Result {
        // 检查视频存在性
        val src = File(srcVideoPath)
        require(src.exists() && src.length() > 0) {
            "视频文件不存在或为空: $srcVideoPath"
        }

        // 构造选项 JSON
        val opts =
            JSONObject().apply {
                put("redet_every", 15)
                put("plate_diameter_m", plateDiameterM)
                put("exercise_type", exerciseType)
                if (outerPlate != null) put("outer_plate", outerPlate)
                put("conf_thresh", 0.40)
            }

        // 调用 native
        val jsonStr =
            nativeAnalyze(srcVideoPath, modelAssetPath, opts.toString())
                ?: throw IllegalStateException("nativeAnalyze 返回 null")

        return Result.fromJson(JSONObject(jsonStr))
    }

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
