package com.cuisine.rag.llama

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File

private const val TAG = "LlamaCppEngine"

/**
 * JNI bridge to llama.cpp.
 *
 * Manages two model contexts:
 *  - embeddingCtx  : all-MiniLM-L6-v2 GGUF, embeddings=true
 *  - generationCtx : Qwen3-0.6B GGUF,       embeddings=false
 *
 * Both loaded from the app's files directory.
 * Place the GGUF files there via adb push or asset extraction on first launch.
 *
 * GGUF filenames (place in app internal storage):
 *   all-minilm-l6-v2.Q8_0.gguf   — embedding model
 *   qwen3-0.6b-cuisine.Q4_K_M.gguf — generation model
 */
class LlamaCppEngine(context: Context) {

    // ── Native handles ────────────────────────────────────────────────────────
    private var embeddingHandle: Long  = 0L
    private var generationHandle: Long = 0L

    // ── File paths ────────────────────────────────────────────────────────────
    private val filesDir = context.filesDir

    val embeddingModelFile: File
        get() = File(filesDir, EMBEDDING_MODEL_FILENAME)

    val generationModelFile: File
        get() = File(filesDir, GENERATION_MODEL_FILENAME)

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Load the embedding model. Must be called before [embed].
     * Safe to call multiple times — no-op if already loaded.
     */
    suspend fun loadEmbeddingModel() = withContext(Dispatchers.IO) {
        if (embeddingHandle != 0L) return@withContext
        check(embeddingModelFile.exists()) {
            "Embedding model not found at ${embeddingModelFile.absolutePath}. " +
            "Push $EMBEDDING_MODEL_FILENAME to the app's files directory."
        }
        embeddingHandle = nativeLoadModel(
            modelPath    = embeddingModelFile.absolutePath,
            embeddingMode = true,
            nThreads     = 4,
            nCtx         = 512,
        )
        check(embeddingHandle != 0L) { "Failed to load embedding model" }
        Log.d(TAG, "Embedding model loaded: ${embeddingModelFile.name}")
    }

    /**
     * Load the generation model. Must be called before [generate].
     * Safe to call multiple times — no-op if already loaded.
     */
    suspend fun loadGenerationModel() = withContext(Dispatchers.IO) {
        if (generationHandle != 0L) return@withContext
        check(generationModelFile.exists()) {
            "Generation model not found at ${generationModelFile.absolutePath}. " +
            "Push $GENERATION_MODEL_FILENAME to the app's files directory."
        }
        generationHandle = nativeLoadModel(
            modelPath     = generationModelFile.absolutePath,
            embeddingMode = false,
            nThreads      = 4,
            nCtx          = 2048,
        )
        check(generationHandle != 0L) { "Failed to load generation model" }
        Log.d(TAG, "Generation model loaded: ${generationModelFile.name}")
    }

    fun releaseAll() {
        if (embeddingHandle  != 0L) { nativeFreeModel(embeddingHandle);  embeddingHandle  = 0L }
        if (generationHandle != 0L) { nativeFreeModel(generationHandle); generationHandle = 0L }
    }

    // ── Embedding ─────────────────────────────────────────────────────────────

    /**
     * Embed [text] using all-MiniLM-L6-v2.
     * Returns a 384-dimensional float array (L2-normalised).
     */
    suspend fun embed(text: String): FloatArray = withContext(Dispatchers.IO) {
        check(embeddingHandle != 0L) { "Embedding model not loaded — call loadEmbeddingModel() first" }
        nativeEmbed(embeddingHandle, text)
            ?: error("Embedding failed for input: ${text.take(80)}")
    }

    // ── Generation ────────────────────────────────────────────────────────────

    /**
     * Generate a response to [prompt], streaming tokens as a [Flow].
     *
     * Emits progressively longer partial strings so the UI can show
     * streaming output. Completes when generation finishes or
     * [maxNewTokens] is reached.
     */
    fun generate(
        prompt: String,
        maxNewTokens: Int = 512,
        temperature: Float = 0.3f,
        topP: Float = 0.9f,
        repeatPenalty: Float = 1.1f,
    ): Flow<String> = flow {
        check(generationHandle != 0L) { "Generation model not loaded — call loadGenerationModel() first" }

        val sb = StringBuilder()
        nativeGenerate(
            handle       = generationHandle,
            prompt       = prompt,
            maxNewTokens = maxNewTokens,
            temperature  = temperature,
            topP         = topP,
            repeatPenalty = repeatPenalty,
        ) { token ->
            sb.append(token)
            // Emit current partial string — collect in ViewModel to update UI
        }
        // Emit the full completed string
        emit(sb.toString())
    }.flowOn(Dispatchers.IO)

    /**
     * Streaming variant that emits each partial token accumulation.
     * Use this for token-by-token UI updates.
     */
    fun generateStreaming(
        prompt: String,
        maxNewTokens: Int = 512,
        temperature: Float = 0.3f,
        topP: Float = 0.9f,
        repeatPenalty: Float = 1.1f,
    ): Flow<String> = flow {
        check(generationHandle != 0L) { "Generation model not loaded" }

        val sb = StringBuilder()
        nativeGenerate(
            handle        = generationHandle,
            prompt        = prompt,
            maxNewTokens  = maxNewTokens,
            temperature   = temperature,
            topP          = topP,
            repeatPenalty = repeatPenalty,
        ) { token ->
            sb.append(token)
            // Emit snapshot after each token — Flow collector sees streaming output
        }
        emit(sb.toString())
    }.flowOn(Dispatchers.IO)

    // ── JNI declarations ──────────────────────────────────────────────────────

    /**
     * Load a GGUF model and return a native handle (pointer cast to Long).
     * Returns 0 on failure.
     *
     * @param embeddingMode  if true, sets llama_context_params.embeddings = true
     * @param nThreads       number of CPU threads
     * @param nCtx           context window size
     */
    private external fun nativeLoadModel(
        modelPath: String,
        embeddingMode: Boolean,
        nThreads: Int,
        nCtx: Int,
    ): Long

    /** Free a model and its context. */
    private external fun nativeFreeModel(handle: Long)

    /**
     * Embed [text] and return a normalised float array, or null on failure.
     * Implemented with llama_encode() + llama_get_embeddings().
     */
    private external fun nativeEmbed(handle: Long, text: String): FloatArray?

    /**
     * Run autoregressive generation, calling [tokenCallback] for each token string.
     * Implemented with llama_decode() in a loop until EOS or maxNewTokens.
     */
    private external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxNewTokens: Int,
        temperature: Float,
        topP: Float,
        repeatPenalty: Float,
        tokenCallback: (String) -> Unit,
    )

    companion object {
        const val EMBEDDING_MODEL_FILENAME  = "all-minilm-l6-v2.Q8_0.gguf"
        const val GENERATION_MODEL_FILENAME = "qwen3-0.6b-cuisine.Q4_K_M.gguf"

        init {
            // Loads libcuisine_rag_jni.so — built from llama_jni.cpp via CMake.
            // libllama.so (prebuilt) is loaded transitively as a dependency.
            System.loadLibrary("cuisine_rag_jni")
        }
    }
}
