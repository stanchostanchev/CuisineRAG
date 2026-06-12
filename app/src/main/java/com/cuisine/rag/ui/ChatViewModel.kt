package com.cuisine.rag.ui

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.cuisine.rag.model.RagState
import com.cuisine.rag.repository.RagRepository
import com.cuisine.rag.vector.VectorIndex
import com.cuisine.rag.llama.LlamaCppEngine
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ChatMessage(
    val text: String,
    val isUser: Boolean,
    val isStreaming: Boolean = false,
    val metadata: String? = null,   // e.g. "Direct match (0.91)" or "RAG context"
)

data class ChatUiState(
    val messages: List<ChatMessage>   = emptyList(),
    val isLoading: Boolean            = false,
    val modelsReady: Boolean          = false,
    val modelsError: String?          = null,
    val inputEnabled: Boolean         = true,
)

@HiltViewModel
class ChatViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val ragRepository: RagRepository,
    private val llamaEngine: LlamaCppEngine,
    private val vectorIndex: VectorIndex,
) : ViewModel() {

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    init {
        loadModels()
    }

    // ── Model loading ─────────────────────────────────────────────────────────

    private fun loadModels() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, modelsError = null) }
            try {
                llamaEngine.loadEmbeddingModel()
                llamaEngine.loadGenerationModel()
                vectorIndex.load(context)
                _uiState.update { it.copy(isLoading = false, modelsReady = true) }
                appendSystemMessage("Models loaded. ${vectorIndex.size} cuisine entries indexed.")
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isLoading    = false,
                        modelsError  = e.message,
                        modelsReady  = false,
                    )
                }
            }
        }
    }

    // ── User actions ──────────────────────────────────────────────────────────

    fun onSendQuestion(question: String) {
        if (question.isBlank() || !_uiState.value.modelsReady) return

        // Append user message
        appendMessage(ChatMessage(text = question, isUser = true))
        _uiState.update { it.copy(inputEnabled = false) }

        viewModelScope.launch {
            ragRepository.answer(question).collect { state ->
                handleRagState(state)
            }
            _uiState.update { it.copy(inputEnabled = true) }
        }
    }

    // ── RAG state → UI ────────────────────────────────────────────────────────

    private fun handleRagState(state: RagState) {
        when (state) {
            is RagState.Embedding -> {
                // Replace or add a "thinking" placeholder
                upsertAssistantMessage(ChatMessage(
                    text = "Searching cuisine knowledge…",
                    isUser = false,
                    isStreaming = true,
                ))
            }

            is RagState.Searching -> {
                upsertAssistantMessage(ChatMessage(
                    text = "Found match (score: ${"%.2f".format(state.score)}): \"${state.matched}\"",
                    isUser = false,
                    isStreaming = true,
                ))
            }

            is RagState.DirectAnswer -> {
                upsertAssistantMessage(ChatMessage(
                    text      = state.answer,
                    isUser    = false,
                    metadata  = "Direct match (${"%.2f".format(state.score)})",
                ))
            }

            is RagState.Generating -> {
                upsertAssistantMessage(ChatMessage(
                    text        = state.partialAnswer,
                    isUser      = false,
                    isStreaming  = true,
                    metadata    = if (state.usedContext) "RAG + Qwen3" else "Qwen3",
                ))
            }

            is RagState.GeneratedAnswer -> {
                upsertAssistantMessage(ChatMessage(
                    text     = state.answer,
                    isUser   = false,
                    metadata = if (state.usedContext) "RAG + Qwen3" else "Qwen3 (no context)",
                ))
            }

            is RagState.Error -> {
                upsertAssistantMessage(ChatMessage(
                    text   = "Error: ${state.message}",
                    isUser = false,
                ))
            }

            RagState.Idle -> Unit
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun appendMessage(message: ChatMessage) {
        _uiState.update { state ->
            state.copy(messages = state.messages + message)
        }
    }

    private fun appendSystemMessage(text: String) {
        appendMessage(ChatMessage(text = text, isUser = false, metadata = "System"))
    }

    /** Replace the last assistant message if it's still streaming, else append. */
    private fun upsertAssistantMessage(message: ChatMessage) {
        _uiState.update { state ->
            val messages = state.messages.toMutableList()
            val lastIndex = messages.indexOfLast { !it.isUser }
            if (lastIndex != -1 && messages[lastIndex].isStreaming) {
                messages[lastIndex] = message
            } else {
                messages.add(message)
            }
            state.copy(messages = messages)
        }
    }

    override fun onCleared() {
        super.onCleared()
        llamaEngine.releaseAll()
    }
}
