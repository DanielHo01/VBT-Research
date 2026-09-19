// android/app/build.gradle.kts
// ============================
// VBT-Research Demo Android 应用模块。
// 集成 Compose UI + JNI (vbtcore-cpp) + ONNX Runtime Mobile。

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.easyvbt.demo"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.easyvbt.demo"
        minSdk = 24 // Android 7.0+
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0-demo"

        ndk {
            version = "28.2.13676358"
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }

        // CMake arguments: OpenCV/JSON/vbtcore-cpp paths
        externalNativeBuild {
            cmake {
                arguments.addAll(
                    listOf(
                        "-DANDROID_STL=c++_shared",
                        "-DANDROID_PLATFORM=android-24",
                        "-DVBT_BUILD_ANDROID=ON",
                        // Explicitly set NDK 28 path (AGP ndkVersion sometimes ignored on first configure)
                        "-DANDROID_NDK=D:/AndroidSdk/ndk/28.2.13676358",
                        "-DCMAKE_ANDROID_NDK=D:/AndroidSdk/ndk/28.2.13676358",
                        "-DOpenCV_DIR=C:/Users/30625/Downloads/opencv-extracted/opencv-mobile-4.11.0-android/sdk/native/jni",
                        "-DJSON_ROOT=${rootProject.projectDir}/../vbtcore-cpp",
                        "-DVBTCORE_CPP_DIR=${rootProject.projectDir}/../vbtcore-cpp",
                    ),
                )
            }
        }
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
        getByName("debug") {
            isDebuggable = true
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"
    }

    // CMakeLists.txt path (separate externalNativeBuild block, merges with above)
    externalNativeBuild {
        cmake {
            path = file("../cpp/CMakeLists.txt")
        }
    }

    packaging {
        resources {
            excludes += listOf("/META-INF/{AL2.0,LGPL2.1}")
        }
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.06.00")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.0")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.documentfile:documentfile:1.0.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")
    implementation("org.json:json:20240303")

    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
