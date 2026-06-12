#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>

#include "llama.h"

#define LOG_TAG "LlamaCppJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Convert a jstring to std::string, releasing the JNI local ref.
 */
static std::string jstring_to_std(JNIEnv* env, jstring js) {
    if (!js) return {};
    const char* chars = env->GetStringUTFChars(js, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(js, chars);
    return result;
}

/**
 * Internal context bundle — one per loaded model.
 * Stored as a jlong handle so Kotlin can hold a reference.
 */
struct LlamaContext {
    llama_model*   model   = nullptr;
    llama_context* ctx     = nullptr;
    bool           is_embedding = false;
    int            n_embd  = 0;    // embedding dimension (set for embedding models)
};

static LlamaContext* handle_to_ctx(jlong handle) {
    return reinterpret_cast<LlamaContext*>(static_cast<uintptr_t>(handle));
}

// ── nativeLoadModel ───────────────────────────────────────────────────────────

extern "C"
JNIEXPORT jlong JNICALL
Java_com_cuisine_rag_llama_LlamaCppEngine_nativeLoadModel(
        JNIEnv*   env,
        jobject   /* this */,
        jstring   j_model_path,
        jboolean  embedding_mode,
        jint      n_threads,
        jint      n_ctx)
{
    std::string model_path = jstring_to_std(env, j_model_path);
    LOGI("Loading model: %s  embedding=%d  threads=%d  ctx=%d",
         model_path.c_str(), (int)embedding_mode, (int)n_threads, (int)n_ctx);

    // ── Model params ──────────────────────────────────────────────────────────
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;    // CPU only on Android — adjust if Vulkan available

    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model) {
        LOGE("Failed to load model from: %s", model_path.c_str());
        return 0L;
    }

    // ── Context params ────────────────────────────────────────────────────────
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx         = static_cast<uint32_t>(n_ctx);
    ctx_params.n_threads     = static_cast<uint32_t>(n_threads);
    ctx_params.n_threads_batch = static_cast<uint32_t>(n_threads);
    ctx_params.embeddings    = static_cast<bool>(embedding_mode);

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        LOGE("Failed to create context for: %s", model_path.c_str());
        llama_free_model(model);
        return 0L;
    }

    auto* bundle         = new LlamaContext();
    bundle->model        = model;
    bundle->ctx          = ctx;
    bundle->is_embedding = static_cast<bool>(embedding_mode);
    bundle->n_embd       = embedding_mode ? llama_model_n_embd(model) : 0;

    LOGI("Model loaded OK. Handle=%p  n_embd=%d", bundle, bundle->n_embd);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(bundle));
}

// ── nativeFreeModel ───────────────────────────────────────────────────────────

extern "C"
JNIEXPORT void JNICALL
Java_com_cuisine_rag_llama_LlamaCppEngine_nativeFreeModel(
        JNIEnv* /* env */,
        jobject /* this */,
        jlong   handle)
{
    auto* bundle = handle_to_ctx(handle);
    if (!bundle) return;

    LOGI("Freeing model handle=%p", bundle);
    if (bundle->ctx)   llama_free(bundle->ctx);
    if (bundle->model) llama_free_model(bundle->model);
    delete bundle;
}

// ── nativeEmbed ───────────────────────────────────────────────────────────────

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_cuisine_rag_llama_LlamaCppEngine_nativeEmbed(
        JNIEnv* env,
        jobject /* this */,
        jlong   handle,
        jstring j_text)
{
    auto* bundle = handle_to_ctx(handle);
    if (!bundle || !bundle->is_embedding) {
        LOGE("nativeEmbed: invalid handle or not an embedding context");
        return nullptr;
    }

    std::string text = jstring_to_std(env, j_text);
    if (text.empty()) return nullptr;

    llama_context* ctx   = bundle->ctx;
    llama_model*   model = bundle->model;
    int            n_embd = bundle->n_embd;

    // Tokenise
    std::vector<llama_token> tokens(text.size() + 4);
    int n_tokens = llama_tokenize(
        model,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        /*add_special=*/true,
        /*parse_special=*/false
    );

    if (n_tokens < 0) {
        LOGE("Tokenisation failed for embed input (len=%zu)", text.size());
        return nullptr;
    }
    tokens.resize(n_tokens);

    // Build a single-sequence batch
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);

    // Clear KV cache and run the encode pass
    llama_kv_cache_clear(ctx);

    if (llama_encode(ctx, batch) != 0) {
        LOGE("llama_encode failed");
        return nullptr;
    }

    // Retrieve embeddings for sequence 0
    const float* emb_ptr = llama_get_embeddings_seq(ctx, 0);
    if (!emb_ptr) {
        // Fallback: get embeddings for last token
        emb_ptr = llama_get_embeddings_ith(ctx, n_tokens - 1);
    }
    if (!emb_ptr) {
        LOGE("llama_get_embeddings returned null");
        return nullptr;
    }

    // L2-normalise
    std::vector<float> emb(emb_ptr, emb_ptr + n_embd);
    float norm = 0.f;
    for (float v : emb) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-9f) {
        for (float& v : emb) v /= norm;
    }

    // Copy to Java float[]
    jfloatArray result = env->NewFloatArray(n_embd);
    if (!result) return nullptr;
    env->SetFloatArrayRegion(result, 0, n_embd, emb.data());
    return result;
}

// ── nativeGenerate ────────────────────────────────────────────────────────────

extern "C"
JNIEXPORT void JNICALL
Java_com_cuisine_rag_llama_LlamaCppEngine_nativeGenerate(
        JNIEnv* env,
        jobject /* this */,
        jlong   handle,
        jstring j_prompt,
        jint    max_new_tokens,
        jfloat  temperature,
        jfloat  top_p,
        jfloat  repeat_penalty,
        jobject token_callback)   // Kotlin (String) -> Unit
{
    auto* bundle = handle_to_ctx(handle);
    if (!bundle || bundle->is_embedding) {
        LOGE("nativeGenerate: invalid handle or embedding-mode context");
        return;
    }

    std::string prompt = jstring_to_std(env, j_prompt);
    if (prompt.empty()) return;

    llama_context* ctx   = bundle->ctx;
    llama_model*   model = bundle->model;

    // ── Resolve the Kotlin callback ───────────────────────────────────────────
    // token_callback is a Kotlin lambda — (String) -> Unit — which the JVM
    // represents as kotlin.jvm.functions.Function1
    jclass  cb_class  = env->GetObjectClass(token_callback);
    jmethodID cb_invoke = env->GetMethodID(
        cb_class, "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");
    if (!cb_invoke) {
        LOGE("Could not find invoke() on token callback");
        return;
    }

    // ── Tokenise prompt ───────────────────────────────────────────────────────
    int max_prompt_tokens = static_cast<int>(llama_n_ctx(ctx)) - max_new_tokens - 4;
    std::vector<llama_token> prompt_tokens(max_prompt_tokens);
    int n_prompt = llama_tokenize(
        model,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        prompt_tokens.data(),
        max_prompt_tokens,
        /*add_special=*/true,
        /*parse_special=*/true
    );
    if (n_prompt < 0) {
        LOGE("Prompt tokenisation failed");
        return;
    }
    prompt_tokens.resize(n_prompt);

    // ── Sampler chain ─────────────────────────────────────────────────────────
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
        /*penalty_last_n=*/64,
        /*penalty_repeat=*/repeat_penalty,
        /*penalty_freq=*/0.0f,
        /*penalty_present=*/0.0f
    ));
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // ── Prefill pass ──────────────────────────────────────────────────────────
    llama_kv_cache_clear(ctx);

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);
    if (llama_decode(ctx, batch) != 0) {
        LOGE("Prefill decode failed");
        llama_sampler_free(sampler);
        return;
    }

    // ── Autoregressive generation loop ────────────────────────────────────────
    llama_token eos_token = llama_token_eos(model);
    int         n_pos     = n_prompt;

    for (int i = 0; i < max_new_tokens; ++i) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, new_token);

        if (new_token == eos_token) break;

        // Decode token → string
        char token_buf[256] = {};
        int  n_chars = llama_token_to_piece(
            model, new_token, token_buf, sizeof(token_buf) - 1,
            /*lstrip=*/0, /*special=*/false);

        if (n_chars > 0) {
            token_buf[n_chars] = '\0';

            // Invoke Kotlin callback with the token string
            jstring j_token = env->NewStringUTF(token_buf);
            env->CallObjectMethod(token_callback, cb_invoke, j_token);
            env->DeleteLocalRef(j_token);

            // Check for JVM exception (e.g. coroutine cancellation)
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                break;
            }
        }

        // Decode the new token to advance KV cache
        llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx, next_batch) != 0) {
            LOGE("Decode failed at step %d", i);
            break;
        }
        n_pos++;
    }

    llama_sampler_free(sampler);
}
