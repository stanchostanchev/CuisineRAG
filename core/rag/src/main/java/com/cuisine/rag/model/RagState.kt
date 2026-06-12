package com.cuisine.rag.model

/**
 * Represents every state the RAG pipeline can be in.
 * Collected by the ViewModel and mapped to UI.
 */
sealed interface RagState {

    /** Initial idle state — no query running. */
    data object Idle : RagState

    /** Embedding the user's question. */
    data object Embedding : RagState

    /**
     * Vector search complete.
     * @param score   cosine similarity 0..1
     * @param matched the closest stored question
     */
    data class Searching(
        val score: Float,
        val matched: String,
    ) : RagState

    /**
     * High-confidence match found (score ≥ 0.82).
     * Answer is returned directly from the index — no LLM call.
     * @param answer  the stored answer text
     * @param score   similarity score
     */
    data class DirectAnswer(
        val answer: String,
        val score: Float,
    ) : RagState

    /**
     * No confident match — falling back to Qwen3 GGUF generation.
     * Streams tokens progressively.
     * @param partialAnswer tokens generated so far
     * @param usedContext   whether retrieved Q&A was injected as context
     */
    data class Generating(
        val partialAnswer: String,
        val usedContext: Boolean,
    ) : RagState

    /**
     * Generation complete.
     * @param answer      full generated answer
     * @param usedContext whether retrieved Q&A context was injected
     */
    data class GeneratedAnswer(
        val answer: String,
        val usedContext: Boolean,
    ) : RagState

    /** An error occurred at any stage. */
    data class Error(val message: String) : RagState
}
