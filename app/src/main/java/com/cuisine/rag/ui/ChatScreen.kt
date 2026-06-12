@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.cuisine.rag.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun ChatRoute(viewModel: ChatViewModel = hiltViewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    ChatScreen(
        uiState       = uiState,
        onSendQuestion = viewModel::onSendQuestion,
    )
}

@Composable
fun ChatScreen(
    uiState: ChatUiState,
    onSendQuestion: (String) -> Unit,
) {
    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    // Scroll to bottom when new message arrives
    LaunchedEffect(uiState.messages.size) {
        if (uiState.messages.isNotEmpty()) {
            listState.animateScrollToItem(uiState.messages.size - 1)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Cuisine RAG", fontWeight = FontWeight.Bold)
                        Text(
                            text = if (uiState.modelsReady)
                                "Ready · ${uiState.messages.size} messages"
                            else if (uiState.isLoading) "Loading models…"
                            else uiState.modelsError ?: "Initialising",
                            style = MaterialTheme.typography.labelSmall,
                            color = if (uiState.modelsReady) Color(0xFF4CAF50) else Color.Gray,
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                )
            )
        },
        bottomBar = {
            InputBar(
                text           = inputText,
                onTextChange   = { inputText = it },
                onSend         = {
                    if (inputText.isNotBlank() && uiState.inputEnabled) {
                        onSendQuestion(inputText.trim())
                        inputText = ""
                    }
                },
                enabled        = uiState.inputEnabled && uiState.modelsReady,
            )
        }
    ) { padding ->
        if (uiState.isLoading) {
            Box(
                Modifier.fillMaxSize().padding(padding),
                contentAlignment = Alignment.Center,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator()
                    Spacer(Modifier.height(12.dp))
                    Text("Loading models…", style = MaterialTheme.typography.bodyMedium)
                }
            }
        } else {
            LazyColumn(
                state            = listState,
                modifier         = Modifier.fillMaxSize().padding(padding),
                contentPadding   = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(uiState.messages) { message ->
                    MessageBubble(message)
                }
            }
        }
    }
}

@Composable
private fun MessageBubble(message: ChatMessage) {
    val isUser    = message.isUser
    val bubbleColor = if (isUser)
        MaterialTheme.colorScheme.primaryContainer
    else
        MaterialTheme.colorScheme.surfaceVariant

    Column(
        modifier  = Modifier.fillMaxWidth(),
        horizontalAlignment = if (isUser) Alignment.End else Alignment.Start,
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 320.dp)
                .clip(
                    RoundedCornerShape(
                        topStart    = if (isUser) 16.dp else 4.dp,
                        topEnd      = if (isUser) 4.dp  else 16.dp,
                        bottomStart = 16.dp,
                        bottomEnd   = 16.dp,
                    )
                )
                .background(bubbleColor)
                .padding(horizontal = 12.dp, vertical = 8.dp)
        ) {
            Column {
                Text(
                    text  = message.text,
                    style = MaterialTheme.typography.bodyMedium,
                )
                if (message.isStreaming) {
                    Spacer(Modifier.height(4.dp))
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
            }
        }

        // Metadata badge (e.g. "Direct match (0.91)" or "RAG + Qwen3")
        message.metadata?.let { meta ->
            Text(
                text     = meta,
                style    = MaterialTheme.typography.labelSmall,
                color    = Color.Gray,
                fontStyle = FontStyle.Italic,
                modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp),
            )
        }
    }
}

@Composable
private fun InputBar(
    text: String,
    onTextChange: (String) -> Unit,
    onSend: () -> Unit,
    enabled: Boolean,
) {
    Surface(
        tonalElevation = 4.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            OutlinedTextField(
                value            = text,
                onValueChange    = onTextChange,
                modifier         = Modifier.weight(1f),
                placeholder      = { Text("Ask about European cuisine…") },
                enabled          = enabled,
                maxLines         = 4,
                keyboardOptions  = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions  = KeyboardActions(onSend = { onSend() }),
            )
            Spacer(Modifier.width(8.dp))
            IconButton(
                onClick  = onSend,
                enabled  = enabled && text.isNotBlank(),
            ) {
                Icon(
                    imageVector  = Icons.Default.Send,
                    contentDescription = "Send",
                    tint = if (enabled && text.isNotBlank())
                        MaterialTheme.colorScheme.primary
                    else Color.Gray,
                )
            }
        }
    }
}
