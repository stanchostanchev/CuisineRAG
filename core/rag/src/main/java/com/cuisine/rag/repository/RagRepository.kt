package com.cuisine.rag.repository

import com.cuisine.rag.llama.LlamaCppEngine
import com.cuisine.rag.llama.PromptBuilder
import com.cuisine.rag.model.RagState
import com.cuisine.rag.vector.VectorIndex
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Orchestrates the full RAG pipeline:
 *
 *   1. Emit [RagState.Embedding]
 *   2. Embed the user question with all-MiniLM via [LlamaCppEngine]
 *   3. Emit [RagState.Searching] with top match score
 *   4a. score ≥ 0.82 → emit [RagState.DirectAnswer] (stored answer, no LLM)
 *   4b. score 0.60–0.82 → build RAG prompt, stream Qwen3, emit [RagState.Generating]
 *       then [RagState.GeneratedAnswer]
 *   4c. score < 0.60 → fallback prompt (no context), same stream path
 *   5. On any exception → emit [RagState.Error]
 */
@Singleton
class RagRepository @Inject constructor(
    private val llamaEngine: LlamaCppEngine,
    private val vectorIndex: VectorIndex,
) {

    /**
     * Answer [question] as a cold [Flow] of [RagState] transitions.
     * Collect in a ViewModel's viewModelScope.
     */
    fun answer(question: String): Flow<RagState> = flow {
        try {
            // ── 1. Embed ──────────────────────────────────────────────────────
            emit(RagState.Embedding)
            val queryEmbedding = llamaEngine.embed(question)

            // ── 2. Search ─────────────────────────────────────────────────────
            val results = vectorIndex.search(queryEmbedding, topK = 3)
            val (topEntry, topScore) = results.first()

            emit(RagState.Searching(score = topScore, matched = topEntry.question))

            // ── 3. Route by score ─────────────────────────────────────────────
            when {
                topScore >= VectorIndex.THRESHOLD_DIRECT -> {
                    // High confidence — return stored answer directly
                    emit(RagState.DirectAnswer(answer = topEntry.answer, score = topScore))
                }

                topScore >= VectorIndex.THRESHOLD_CONTEXT -> {
                    // Medium confidence — inject top-3 as RAG context
                    val contextEntries = results.map { it.first }
                    val prompt = PromptBuilder.ragPrompt(question, contextEntries)
                    streamGeneration(prompt, usedContext = true)
                }

                else -> {
                    // Low confidence — pure generation, no context
                    val prompt = PromptBuilder.fallbackPrompt(question)
                    streamGeneration(prompt, usedContext = false)
                }
            }

        } catch (e: Exception) {
            emit(RagState.Error(e.message ?: "Unknown error in RAG pipeline"))
        }
    }

    // ── Private ───────────────────────────────────────────────────────────────

    private suspend fun kotlinx.coroutines.flow.FlowCollector<RagState>.streamGeneration(
        prompt: String,
        usedContext: Boolean,
    ) {
        val sb = StringBuilder()

        llamaEngine.generateStreaming(
            prompt       = prompt,
            maxNewTokens = 512,
            temperature  = 0.3f,
            topP         = 0.9f,
            repeatPenalty = 1.1f,
        ).collect { partial ->
            sb.clear()
            sb.append(partial)
            emit(RagState.Generating(partialAnswer = partial, usedContext = usedContext))
        }

        emit(
            RagState.GeneratedAnswer(
                answer      = sb.toString().trim(),
                usedContext = usedContext,
            )
        )
    }
}
