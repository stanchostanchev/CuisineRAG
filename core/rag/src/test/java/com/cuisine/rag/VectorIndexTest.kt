package com.cuisine.rag

import com.cuisine.rag.model.VectorEntry
import com.cuisine.rag.vector.VectorIndex
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class VectorIndexTest {

    private val dim = 384

    // ── Helpers ───────────────────────────────────────────────────────────────

    /** Write a minimal valid index binary to a temp file. */
    private fun writeTempIndex(entries: List<VectorEntry>): File {
        val baos = ByteArrayOutputStream()
        val dos  = DataOutputStream(baos)

        // Header
        dos.writeInt(0x43555349) // MAGIC
        dos.writeInt(1)          // VERSION
        dos.writeInt(entries.size)
        dos.writeInt(dim)

        for (entry in entries) {
            val qBytes = entry.question.toByteArray(Charsets.UTF_8)
            val aBytes = entry.answer.toByteArray(Charsets.UTF_8)
            dos.writeInt(qBytes.size)
            dos.write(qBytes)
            dos.writeInt(aBytes.size)
            dos.write(aBytes)

            val floatBytes = ByteBuffer.allocate(dim * 4)
                .order(ByteOrder.LITTLE_ENDIAN)
                .also { buf -> entry.embedding.forEach { buf.putFloat(it) } }
                .array()
            dos.write(floatBytes)
        }
        dos.flush()

        val file = File.createTempFile("test_index", ".bin")
        file.writeBytes(baos.toByteArray())
        return file
    }

    private fun unitVector(index: Int, size: Int = dim): FloatArray {
        return FloatArray(size).also { it[index] = 1f }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    @Test
    fun `load reads correct entry count`() = runTest {
        val entries = listOf(
            VectorEntry("What is paella?", "Paella is...", unitVector(0)),
            VectorEntry("What is pizza?",  "Pizza is...",  unitVector(1)),
        )
        val file  = writeTempIndex(entries)
        val index = VectorIndex()
        index.load(file)

        assertEquals(2, index.size)
    }

    @Test
    fun `searchTop1 returns exact match for identical vector`() = runTest {
        val entries = listOf(
            VectorEntry("What is paella?",  "Paella is a Valencian rice dish.", unitVector(0)),
            VectorEntry("What is risotto?", "Risotto is an Italian rice dish.", unitVector(1)),
            VectorEntry("What is fondue?",  "Fondue is a Swiss cheese dish.",   unitVector(2)),
        )
        val file  = writeTempIndex(entries)
        val index = VectorIndex()
        index.load(file)

        val (match, score) = index.searchTop1(unitVector(1))

        assertEquals("What is risotto?", match.question)
        assertEquals(1.0f, score, 0.001f)
    }

    @Test
    fun `search returns topK results sorted descending`() = runTest {
        val entries = (0 until 10).map { i ->
            VectorEntry("Q$i", "A$i", unitVector(i))
        }
        val file  = writeTempIndex(entries)
        val index = VectorIndex()
        index.load(file)

        val results = index.search(unitVector(3), topK = 3)

        assertEquals(3, results.size)
        // First result should be the exact match
        assertEquals("Q3", results[0].first.question)
        // Results should be sorted descending
        assertTrue(results[0].second >= results[1].second)
        assertTrue(results[1].second >= results[2].second)
    }

    @Test
    fun `cosine similarity of orthogonal vectors is zero`() = runTest {
        val entries = listOf(
            VectorEntry("Q0", "A0", unitVector(0)),
            VectorEntry("Q1", "A1", unitVector(1)),
        )
        val file  = writeTempIndex(entries)
        val index = VectorIndex()
        index.load(file)

        // unitVector(0) vs unitVector(1) → cosine = 0
        val results = index.search(unitVector(0), topK = 2)
        val (_, scoreQ1) = results.first { it.first.question == "Q1" }

        assertEquals(0f, scoreQ1, 0.001f)
    }

    @Test
    fun `threshold constants have sensible ordering`() {
        assertTrue(VectorIndex.THRESHOLD_DIRECT  > VectorIndex.THRESHOLD_CONTEXT)
        assertTrue(VectorIndex.THRESHOLD_CONTEXT > 0f)
        assertTrue(VectorIndex.THRESHOLD_DIRECT  < 1f)
    }
}
