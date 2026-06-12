package com.cuisine.rag.llama

import com.cuisine.rag.model.VectorEntry

/**
 * Builds Qwen3 ChatML-formatted prompts for the two generation modes.
 *
 * Qwen3 ChatML format (no_think — skip reasoning block):
 *
 *   <|im_start|>system
 *   {system}
 *   <|im_end|>
 *   <|im_start|>user
 *   {user} /no_think
 *   <|im_end|>
 *   <|im_start|>assistant
 */
object PromptBuilder {

    private const val SYSTEM_PROMPT =
        "You are a knowledgeable European cuisine expert. " +
        "Answer questions about European food, dishes, ingredients, and culinary traditions " +
        "accurately and with detail."

    /**
     * RAG prompt: injects retrieved Q&A context before the user question.
     * Used when similarity is between [VectorIndex.THRESHOLD_CONTEXT] and
     * [VectorIndex.THRESHOLD_DIRECT].
     */
    fun ragPrompt(question: String, context: List<VectorEntry>): String {
        val contextBlock = context.joinToString("\n\n") { entry ->
            "Q: ${entry.question}\nA: ${entry.answer}"
        }
        val userMessage = """
            |Use the following reference information to answer the question.
            |
            |--- Reference ---
            |$contextBlock
            |--- End reference ---
            |
            |Question: $question
        """.trimMargin()

        return buildChatML(userMessage)
    }

    /**
     * Fallback prompt: no retrieved context, model answers from its own knowledge.
     * Used when similarity is below [VectorIndex.THRESHOLD_CONTEXT].
     */
    fun fallbackPrompt(question: String): String =
        buildChatML(question)

    // ── Private ───────────────────────────────────────────────────────────────

    private fun buildChatML(userContent: String): String = buildString {
        append("<|im_start|>system\n")
        append(SYSTEM_PROMPT)
        append("\n<|im_end|>\n")
        append("<|im_start|>user\n")
        append(userContent.trim())
        append(" /no_think")          // suppress Qwen3 think block
        append("\n<|im_end|>\n")
        append("<|im_start|>assistant\n")
    }
}
