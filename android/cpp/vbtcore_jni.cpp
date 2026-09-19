/*
 * android/cpp/vbtcore_jni.cpp — JNI 桥接
 * =======================================
 * 把 vbtcore-cpp 的 C-linkage 函数暴露给 Kotlin/Java 层。
 */
// Android log — no-op for LSP (NDK headers resolved at build time only)
#define __android_log_print(...) ((void)0)
#include "jni.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "vbt/frame_analyzer.hpp"

// vbtcore-cpp 头文件
#include "vbt/analyze.hpp"
#include "vbt/frame_analyzer.hpp"
#include "vbt/segmenter_v2.hpp"

// nlohmann/json（Android 构建 bundling 版本）
#include "../include/nlohmann/json.hpp"

// 全局帧分析器（每个分析会话一个）
static vbt::FrameAnalyzer* g_analyzer = nullptr;

extern "C" {

// ── 视频文件路径模式（桌面端 cv::VideoCapture，Android stub）──────────
JNIEXPORT jstring JNICALL
Java_com_easyvbt_demo_EngineBridge_nativeAnalyze(
    JNIEnv* env,
    jobject /*thiz*/,
    jstring jVideoPath,
    jstring jModelPath,
    jstring jOptionsJson
) {
    const char* video_path = env->GetStringUTFChars(jVideoPath, nullptr);
    const char* model_path = env->GetStringUTFChars(jModelPath, nullptr);
    const char* options_json = (jOptionsJson != nullptr)
        ? env->GetStringUTFChars(jOptionsJson, nullptr)
        : nullptr;

    char* result = vbt::vbt_analyze_video(video_path, model_path, options_json);

    env->ReleaseStringUTFChars(jVideoPath, video_path);
    env->ReleaseStringUTFChars(jModelPath, model_path);
    if (options_json != nullptr) {
        env->ReleaseStringUTFChars(jOptionsJson, options_json);
    }

    jstring jResult = nullptr;
    if (result != nullptr) {
        jResult = env->NewStringUTF(result);
        std::free(result);
    }
    return jResult;
}

// ── 帧输入模式（Android MediaMetadataRetriever 提取帧后调用）───────────
JNIEXPORT jboolean JNICALL
Java_com_easyvbt_demo_EngineBridge_nativeInitAnalyzer(
    JNIEnv* env,
    jobject /*thiz*/,
    jdouble fps,
    jstring jModelPath,
    jdouble plateDiameterM,
    jstring jExerciseType,
    jint redetEvery,
    jdouble confThresh
) {
    delete g_analyzer;
    g_analyzer = nullptr;

    const char* model_path = env->GetStringUTFChars(jModelPath, nullptr);
    const char* exercise_type = env->GetStringUTFChars(jExerciseType, nullptr);

    try {
        g_analyzer = new vbt::FrameAnalyzer(
            fps,
            std::string(model_path),
            plateDiameterM,
            std::string(exercise_type ? exercise_type : "squat_bench"),
            static_cast<int>(redetEvery),
            confThresh
        );
    } catch (...) {
        g_analyzer = nullptr;
    }

    env->ReleaseStringUTFChars(jModelPath, model_path);
    if (jExerciseType != nullptr) {
        env->ReleaseStringUTFChars(jExerciseType, exercise_type);
    }

    return (g_analyzer != nullptr) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_easyvbt_demo_EngineBridge_nativeAddFrame(
    JNIEnv* env,
    jobject /*thiz*/,
    jbyteArray jRgbaData,
    jint width,
    jint height,
    jint strideBytes,
    jint frameIdx,
    jdouble timestampMs
) {
    if (g_analyzer == nullptr) return;

    jbyte* rgba_ptr = env->GetByteArrayElements(jRgbaData, nullptr);
    const jsize data_len = env->GetArrayLength(jRgbaData);

    if (rgba_ptr != nullptr && data_len >= strideBytes * height) {
        g_analyzer->add_frame(
            reinterpret_cast<const uint8_t*>(rgba_ptr),
            static_cast<int>(width),
            static_cast<int>(height),
            static_cast<int>(strideBytes),
            static_cast<int>(frameIdx),
            timestampMs
        );
    }

    if (rgba_ptr != nullptr) {
        env->ReleaseByteArrayElements(jRgbaData, rgba_ptr, JNI_ABORT);
    }
}

JNIEXPORT jboolean JNICALL
Java_com_easyvbt_demo_EngineBridge_nativeIsCalibrated(
    JNIEnv* env,
    jobject /*thiz*/
) {
    return (g_analyzer != nullptr && g_analyzer->is_calibrated())
        ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jdouble JNICALL
Java_com_easyvbt_demo_EngineBridge_nativeGetMpp(
    JNIEnv* env,
    jobject /*thiz*/
) {
    return (g_analyzer != nullptr) ? g_analyzer->get_mpp() : 0.0;
}

JNIEXPORT jstring JNICALL
Java_com_easyvbt_demo_EngineBridge_nativeFinalize(
    JNIEnv* env,
    jobject /*thiz*/
) {
    std::string result_json;
    if (g_analyzer != nullptr) {
        result_json = g_analyzer->finalize();
        delete g_analyzer;
        g_analyzer = nullptr;
    } else {
        result_json = R"({"status":"NOT_INITIALIZED"})";
    }
    // 简单字符串提取关键诊断字段
    {
        const char* s = result_json.c_str();
        char status[32] = "?", mpp_str[32] = "?";
        int detected = 0, total = 0;
        double det_min = 0, det_max = 0, raw_min = 0, raw_max = 0;
        if (const char* p = strstr(s, "\"status\":\"")) strncpy(status, p + 10, 31);
        if (const char* p = strstr(s, "\"mpp\":\"")) strncpy(mpp_str, p + 7, 31);
        if (const char* p = strstr(s, "\"barbell_detected_frames\":"))
            sscanf(p + 25, "%d", &detected);
        if (const char* p = strstr(s, "\"total_frames\":"))
            sscanf(p + 16, "%d", &total);
        if (const char* p = strstr(s, "\"barbell_y_min_px\":\""))
            sscanf(p + 20, "%lf", &det_min);
        if (const char* p = strstr(s, "\"barbell_y_max_px\":\""))
            sscanf(p + 20, "%lf", &det_max);
        if (const char* p = strstr(s, "\"raw_pos_min_px\":\""))
            sscanf(p + 17, "%lf", &raw_min);
        if (const char* p = strstr(s, "\"raw_pos_max_px\":\""))
            sscanf(p + 17, "%lf", &raw_max);
        __android_log_print(ANDROID_LOG_INFO, "EngineBridge",
            "nativeFinalize: status=%s mpp=%s detected=%d/%d "
            "det_px=[%.1f,%.1f] raw_px=[%.1f,%.1f]",
            status, mpp_str, detected, total,
            det_min, det_max, raw_min, raw_max);
    }
    return env->NewStringUTF(result_json.c_str());
}

// ── v2 段分割器（Java 传轨迹数组）─────────────────────────────────
// 由 EngineBridge.kt 调用，替代 FrameAnalyzer 的旧 FSM 段分割器
JNIEXPORT jstring JNICALL
Java_com_easyvbt_demo_EngineBridge_nativeSegmentV2(
    JNIEnv* env,
    jobject /*thiz*/,
    jdoubleArray jTimestamps,
    jdoubleArray jPositions,
    jdoubleArray jVelocities,
    jdouble mpp,
    jstring jExerciseType
) {
    const char* exercise_type = env->GetStringUTFChars(jExerciseType, nullptr);
    std::string exercise = exercise_type ? exercise_type : std::string("squat_bench");

    jsize n = env->GetArrayLength(jTimestamps);
    if (n == 0 || (std::size_t)n != (std::size_t)env->GetArrayLength(jPositions) ||
        (std::size_t)n != (std::size_t)env->GetArrayLength(jVelocities)) {
        if (exercise_type) env->ReleaseStringUTFChars(jExerciseType, exercise_type);
        return env->NewStringUTF(R"({"status":"ARRAY_LENGTH_MISMATCH"})");
    }

    std::vector<double> t(n), y(n), v(n);
    env->GetDoubleArrayRegion(jTimestamps, 0, n, t.data());
    env->GetDoubleArrayRegion(jPositions,  0, n, y.data());
    env->GetDoubleArrayRegion(jVelocities, 0, n, v.data());

    vbt::PhysicalAlignedTrajectorySegmenter seg(exercise, mpp, 0.01, 8, 0.25);
    std::vector<vbt::Rep> reps = seg.segment(t.data(), y.data(), v.data(), n);

    nlohmann::json result;
    result["status"] = reps.empty() ? "NO_REP_DETECTED" : "OK";
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& r : reps) {
        nlohmann::json j;
        j["start_idx"] = r.start_idx;
        j["end_idx"]   = r.end_idx;
        j["start_time"] = r.start_time;
        j["end_time"]   = r.end_time;
        j["duration_s"]  = r.duration_s;
        j["rom_m"]       = r.rom_m;
        j["mcv_mps"]    = r.mcv_mps;
        j["pcv_mps"]    = r.pcv_mps;
        arr.push_back(j);
    }
    result["reps"] = arr;
    nlohmann::json diag;
    diag["method"] = "v2_physical_aligned_segmenter";
    diag["n_frames"] = (int)n;
    diag["n_reps"] = (int)reps.size();
    if (!v.empty()) {
        double v_min = v[0], v_max = v[0], v_sum = 0.0;
        for (double vi : v) {
            if (vi < v_min) v_min = vi;
            if (vi > v_max) v_max = vi;
            v_sum += vi;
        }
        diag["v_min"] = v_min;
        diag["v_max"] = v_max;
        diag["v_mean"] = v_sum / v.size();
    }
    result["diagnostics"] = diag;

    if (exercise_type) env->ReleaseStringUTFChars(jExerciseType, exercise_type);

    std::string json_str = result.dump();
    return env->NewStringUTF(json_str.c_str());
}

}  // extern "C"
