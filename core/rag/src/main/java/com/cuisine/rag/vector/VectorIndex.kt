package com.cuisine.rag.vector

import android.content.Context
import android.util.Log
import com.cuisine.rag.model.VectorEntry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.DataInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

private const val TAG          = "VectorIndex"
private const val MAGIC        = 0x43555349  // "CUSI" in hex
private const val VERSION      = 1
private const val EMBEDDING_DIM = 384

/**
 * Loads and queries the cuisine vector index produced by the Colab notebook.
 *
 * Binary format of cuisine_index.bin:
 * ┌─────────────────────────────────────────────┐
 * │  Header                                      │
 * │    magic   : Int32  (0x43555349)             │
 * │    version : Int32  (1)                      │
 * │    count   : Int32  (number of entries)      │
 * │    dim     : Int32  (embedding dimensions)   │
 * ├─────────────────────────────────────────────┤
 * │  Entry × count                               │
 * │    questionLen : Int32                       │
 * │    question    : UTF-8 bytes                 │
 * │    answerLen   : Int32                       │
 * │    answer      : UTF-8 bytes                 │
 * │    embedding   : Float32 × dim (little-end.) │
 * └─────────────────────────────────────────────┘
 */
class VectorIndex {

    private val entries = mutableListOf<VectorEntry>()

    val size: Int get() = entries.size
    val isLoaded: Boolean get() = entries.isNotEmpty()

    // ── Loading ───────────────────────────────────────────────────────────────

    /**
     * Load the index from [file].
     * Typically called once on app start, from a coroutine.
     */
    suspend fun load(file: File) = withContext(Dispatchers.IO) {
        require(file.exists()) {
            "cuisine_index.bin not found at ${file.absolutePath}. " +
            "Copy it to the app's files directory."
        }

        entries.clear()

        DataInputStream(file.inputStream().buffered()).use { stream ->
            // Header
            val magic   = stream.readInt()
            val version = stream.readInt()
            val count   = stream.readInt()
            val dim     = stream.readInt()

            check(magic == MAGIC)     { "Invalid index file — wrong magic number" }
            check(version == VERSION) { "Unsupported index version: $version" }
            check(dim == EMBEDDING_DIM) {
                "Dimension mismatch: index has $dim, expected $EMBEDDING_DIM"
            }

            val floatBuf = ByteArray(dim * 4)

            repeat(count) {
                val qLen     = stream.readInt()
                val question = String(stream.readNBytes(qLen), Charsets.UTF_8)
                val aLen     = stream.readInt()
                val answer   = String(stream.readNBytes(aLen), Charsets.UTF_8)

                stream.readFully(floatBuf)
                val embedding = FloatArray(dim)
                ByteBuffer.wrap(floatBuf)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                    .get(embedding)

                entries.add(VectorEntry(question, answer, embedding))
            }

            Log.d(TAG, "Loaded $count entries (dim=$EMBEDDING_DIM) from ${file.name}")
        }
    }

    /**
     * Convenience: load from app internal storage using [context].
     */
    suspend fun load(context: Context) {
        load(File(context.filesDir, INDEX_FILENAME))
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /**
     * Find the [topK] closest entries to [queryEmbedding] by cosine similarity.
     *
     * @return list of (entry, score) sorted descending by score
     */
    suspend fun search(
        queryEmbedding: FloatArray,
        topK: Int = 3,
    ): List<Pair<VectorEntry, Float>> = withContext(Dispatchers.Default) {
        check(isLoaded) { "Index not loaded — call load() first" }
        require(queryEmbedding.size == EMBEDDING_DIM) {
            "Query embedding has ${queryEmbedding.size} dims, expected $EMBEDDING_DIM"
        }

        val queryNorm = l2norm(queryEmbedding)

        entries
            .map { entry ->
                val score = cosineSimilarity(queryEmbedding, queryNorm, entry.embedding)
                entry to score
            }
            .sortedByDescending { it.second }
            .take(topK)
    }

    /**
     * Return the single best match.
     */
    suspend fun searchTop1(queryEmbedding: FloatArray): Pair<VectorEntry, Float> =
        search(queryEmbedding, topK = 1).first()

    // ── Math helpers ──────────────────────────────────────────────────────────

    private fun cosineSimilarity(
        a: FloatArray,
        aNorm: Float,
        b: FloatArray,
    ): Float {
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        val bNorm = l2norm(b)
        return if (aNorm == 0f || bNorm == 0f) 0f
        else dot / (aNorm * bNorm)
    }

    private fun l2norm(v: FloatArray): Float {
        var sum = 0f
        for (x in v) sum += x * x
        return sqrt(sum)
    }

    companion object {
        const val INDEX_FILENAME = "cuisine_index.bin"

        // Similarity thresholds
        const val THRESHOLD_DIRECT   = 0.82f  // return stored answer directly
        const val THRESHOLD_CONTEXT  = 0.60f  // inject as RAG context for Qwen3
        // below THRESHOLD_CONTEXT → call Qwen3 with no retrieved context
    }
}
