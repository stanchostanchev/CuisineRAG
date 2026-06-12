plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.hilt)
    alias(libs.plugins.ksp)
}

android {
    namespace  = "com.cuisine.rag.core"
    compileSdk = 35

    defaultConfig { minSdk = 26 }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    // Package the prebuilt llama.cpp .so libraries
    sourceSets["main"].jniLibs.srcDirs("src/main/jniLibs")

    // Build the JNI bridge only when the prebuilt libllama.so is present.
    // Obtain libllama.so from https://github.com/ggerganov/llama.cpp/releases
    // and place it under src/main/jniLibs/<abi>/libllama.so before building.
    val hasLlamaLibs = file("src/main/jniLibs/arm64-v8a/libllama.so").exists() ||
                       file("src/main/jniLibs/x86_64/libllama.so").exists()
    if (hasLlamaLibs) {
        externalNativeBuild {
            cmake {
                path    = file("src/main/cpp/CMakeLists.txt")
                version = "3.22.1"
            }
        }
    }

    defaultConfig {
        if (hasLlamaLibs) {
            externalNativeBuild {
                cmake {
                    abiFilters += listOf("arm64-v8a", "x86_64")
                    arguments += "-DANDROID_STL=c++_shared"
                }
            }
        }
    }
}

dependencies {
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.androidx.lifecycle.viewmodel)

    implementation(libs.hilt.android)
    ksp(libs.hilt.compiler)

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}
