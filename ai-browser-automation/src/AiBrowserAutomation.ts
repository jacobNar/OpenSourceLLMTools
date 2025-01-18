import LLMModule from './LLMModule';
import { ConversationEntry } from './models/LLM';
class AiBrowserAutomation {
  private llmModule: LLMModule | null;

  constructor() {
    this.llmModule = null; // The LLMModule instance will be initialized later
  }

  /**
   * Initializes the LLMModule with a specified model.
   * @param modelName - The model to initialize the LLMModule with.
   */
  async initializeLLM(modelName: string): Promise<void> {
    this.llmModule = new LLMModule(modelName);
    await this.llmModule.init(); // Initialize the LLM module
    console.log(`LLMModule initialized with model: ${modelName}`);
  }

  /**
   * Prompts the model with a user message and gets a response.
   * @param userMessage - The message the user inputs.
   * @returns The model's response.
   */
  async promptModel(userMessage: string): Promise<ConversationEntry[]> {
    if (!this.llmModule) {
      throw new Error('LLMModule not initialized. Call `initializeLLM()` first.');
    }

    // Interact with the LLMModule and get the model's response
    const response = await this.llmModule.interact(userMessage);
    return response;
  }
}

export default AiBrowserAutomation;
