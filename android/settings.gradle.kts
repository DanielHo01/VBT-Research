// android/settings.gradle.kts
// ==========================
// 启用 Android Gradle Plugin 仓库 + vbtcore-cpp 子模块包含。

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "EasyVBT-Demo"
include(":app")

// vbtcore-cpp 子项目（NDK 构建集成 CMake）
// 路径相对 settings.gradle.kts：android/ → ../vbtcore-cpp
include(":vbtcore-cpp")
project(":vbtcore-cpp").projectDir = file("../vbtcore-cpp/android")
