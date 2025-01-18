import { pipeline } from '@huggingface/transformers';
import { ConversationEntry } from './models/LLM';

class LLMModule {
  private modelName: string;
  private pipelineInstance: any | null; // To hold the pipeline instance
  private conversationHistory: ConversationEntry[]; // To store the conversation history

  constructor(model: string) {
    this.modelName = model;
    this.pipelineInstance = null;
    this.conversationHistory = [];

  }

  /**
   * Initializes the pipeline with the specified model and enables WebGPU.
   */
  async init(): Promise<void> {
    try {
      console.log(`Loading model: ${this.modelName}...`);
      this.pipelineInstance = await pipeline('text-generation', this.modelName);
      console.log('Model loaded successfully!');
    } catch (error) {
      console.error(`Error loading model ${this.modelName}:`, error);
      throw error;
    }
  }

  /**
   * Adds a user message to the history, generates a response, and adds it to the history.
   * @param userMessage - The user's input.
   * @returns The assistant's response.
   */
  async interact(userMessage: string): Promise<ConversationEntry[]> {
    if (!this.pipelineInstance) {
      throw new Error('Pipeline not initialized. Call `init()` first.');
    }

    // Add the user message to the history
    this.conversationHistory.push({ role: 'user', content: userMessage });

    try {
      // Generate the assistant's response
      console.log('Generating response...');
      const output = await this.pipelineInstance(this.conversationHistory, {
        max_length: 2048, // Adjust for the desired output length
        temperature: 0.7,
        top_k: 50,
        top_p: 0.95,
      });
      console.log('Response generated!');
      const response = output[0]?.generated_text ?? this.conversationHistory;

      // Add the assistant's response to the history
      this.conversationHistory = response;

      return this.conversationHistory;
    } catch (error) {
      console.error('Error during LLM inference:', error);
      throw error;
    }
  }
}

export default LLMModule;
