package com.cuisine.rag.model

/**
 * One entry in the cuisine vector index.
 *
 * @param question  the original question text
 * @param answer    the stored answer text
 * @param embedding 384-dimensional float vector from all-MiniLM-L6-v2
 */
data class VectorEntry(
    val question: String,
    val answer: String,
    val embedding: FloatArray,
) {
    // FloatArray doesn't implement equals/hashCode by default
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is VectorEntry) return false
        return question == other.question &&
               answer == other.answer &&
               embedding.contentEquals(other.embedding)
    }

    override fun hashCode(): Int {
        var result = question.hashCode()
        result = 31 * result + answer.hashCode()
        result = 31 * result + embedding.contentHashCode()
        return result
    }
}
