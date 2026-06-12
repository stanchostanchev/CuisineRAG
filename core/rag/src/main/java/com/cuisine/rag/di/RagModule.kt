package com.cuisine.rag.di

import android.content.Context
import com.cuisine.rag.llama.LlamaCppEngine
import com.cuisine.rag.vector.VectorIndex
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object RagModule {

    @Provides
    @Singleton
    fun provideLlamaCppEngine(
        @ApplicationContext context: Context,
    ): LlamaCppEngine = LlamaCppEngine(context)

    @Provides
    @Singleton
    fun provideVectorIndex(): VectorIndex = VectorIndex()
}
