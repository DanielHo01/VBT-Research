/*
 * android/cpp/vbtcore_jni.cpp — JNI 桥接
 * =======================================
 * 把 vbtcore-cpp 的 C-linkage 函数暴露给 Kotlin/Java 层。
 * 对应 Kotlin 端 EngineBridge.nativeAnalyze。
 */
#include <jni.h>
#include <cstring>
#include <string>

#include "vbt/analyze.hpp"

extern "C" {

/**
 * JNI 入口：nativeAnalyze(String videoPath, String modelPath, String optionsJson) -> String
 */
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
        // vbt_analyze_video 内部 strdup，需 free
        std::free(result);
    }
    return jResult;
}

}  // extern "C"