# CuisineRAG — Android

On-device European cuisine Q&A using vector similarity search + Qwen3 fallback.
Both the embedding model (all-MiniLM-L6-v2) and the generative model (Qwen3-0.6B)
run entirely on-device via llama.cpp JNI — no internet connection required.

---

## Architecture

```
User question
      │
      ▼
LlamaCppEngine.embed()          all-minilm-l6-v2.Q8_0.gguf
      │
      ▼
VectorIndex.search()            cuisine_index.bin  (400 × 384 floats)
      │
      ├─ score ≥ 0.82 ──────────► Return stored answer directly
      │
      ├─ score 0.60–0.82 ───────► Inject top-3 as context → Qwen3
      │
      └─ score < 0.60 ──────────► Qwen3 (no context)
```

## Module structure

```
CuisineRAG/
├── app/                        # App shell — MainActivity, ChatScreen, ChatViewModel
└── core/
    └── rag/                    # All RAG logic — no Android UI dependencies
        ├── llama/
        │   ├── LlamaCppEngine.kt   JNI bridge (embedding + generation)
        │   └── PromptBuilder.kt    Qwen3 ChatML prompt construction
        ├── vector/
        │   └── VectorIndex.kt      Loads cuisine_index.bin, cosine search
        ├── repository/
        │   └── RagRepository.kt    Orchestrates the full pipeline
        ├── model/
        │   ├── RagState.kt         Sealed interface for UI state
        │   └── VectorEntry.kt      Index entry data class
        └── di/
            └── RagModule.kt        Hilt providers
```

## Setup

### Step 1 — Build the index (Colab)

1. Open `Build_Cuisine_Index.ipynb` in Google Colab
2. Upload `Qwen_cuisine_qa.txt`
3. Run all cells
4. Download `cuisine_index.bin` and `all-minilm-l6-v2.Q8_0.gguf`

### Step 2 — Get Qwen3 GGUF

Either:
- Use the GGUF exported by the `Qwen3_0_6B_CuisineQA.ipynb` notebook
- Or download `Qwen3-0.6B-Instruct-GGUF` from HuggingFace

Rename to `qwen3-0.6b-cuisine.Q4_K_M.gguf`.

### Step 3 — Get llama.cpp prebuilt .so

Download the Android arm64-v8a prebuilt from llama.cpp releases, or build from source:

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-26 \
      -DGGML_OPENMP=OFF \
      -B build-android
cmake --build build-android --target llama -j4
```

Place `libllama.so` in:
```
core/rag/src/main/jniLibs/arm64-v8a/libllama.so
```

### Step 4 — Implement llama_jni.cpp

`LlamaCppEngine.kt` declares four native methods. Implement them in C++ using the llama.cpp C API:
- `nativeLoadModel` → `llama_load_model_from_file` + `llama_new_context_with_model`
- `nativeFreeModel` → `llama_free` + `llama_free_model`
- `nativeEmbed`     → `llama_encode` + `llama_get_embeddings`
- `nativeGenerate`  → `llama_decode` loop + token callback via JNI

### Step 5 — Push files to device

```bash
adb push cuisine_index.bin              /data/data/com.cuisine.rag/files/
adb push all-minilm-l6-v2.Q8_0.gguf    /data/data/com.cuisine.rag/files/
adb push qwen3-0.6b-cuisine.Q4_K_M.gguf /data/data/com.cuisine.rag/files/
```

### Step 6 — Build and run

```bash
./gradlew installDebug
```

---

## Similarity thresholds

| Score | Action |
|-------|--------|
| ≥ 0.82 | Return stored answer (deterministic, instant) |
| 0.60 – 0.82 | Qwen3 with top-3 Q&A as context |
| < 0.60 | Qwen3 with no context |

Tune in `VectorIndex.kt`:
```kotlin
const val THRESHOLD_DIRECT  = 0.82f
const val THRESHOLD_CONTEXT = 0.60f
```

---

## Files required in app internal storage

| File | Size | Purpose |
|------|------|---------|
| `cuisine_index.bin` | ~1 MB | 400 pre-embedded Q&A vectors |
| `all-minilm-l6-v2.Q8_0.gguf` | ~22 MB | Embedding model |
| `qwen3-0.6b-cuisine.Q4_K_M.gguf` | ~400 MB | Generative fallback model |
