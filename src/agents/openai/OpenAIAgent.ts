import { GoogleGenAI } from "@google/genai";
import type { Channel, DefaultGenerics, Event, StreamChat, MessageResponse } from "stream-chat";
import type { AIAgent } from "../types";
import { GeminiResponseHandler } from "./GeminiResponseHandler"; 

export class OpenAIAgent implements AIAgent {
  private genai?: GoogleGenAI;
  private lastInteractionTs = Date.now();
  private handlers: GeminiResponseHandler[] = [];

  constructor(
    readonly chatClient: StreamChat,
    readonly channel: Channel
  ) {}

  dispose = async () => {
    this.chatClient.off("message.new", this.handleMessage);
    await this.chatClient.disconnectUser();
  };

  get user() {
    return this.chatClient.user;
  }

  getLastInteraction = (): number => this.lastInteractionTs;

  init = async () => {
    const apiKey = process.env.GEMINI_API_KEY as string | undefined;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is required");
    }

    this.genai = new GoogleGenAI({ apiKey }); 

    this.chatClient.on("message.new", this.handleMessage);
  };

  private getWritingAssistantPrompt = (context?: string): string => {
    const date = new Date().toLocaleDateString("en-US");
    return `
You are an expert writing assistant.
Current Date: ${date}
Context: ${context || "General writing assistance."}
Write clearly and professionally. No disclaimers.
`;
  };

  private handleMessage = async (e: Event<DefaultGenerics>) => {
    if (!this.genai) {
      console.log("Gemini client not initialized");
      return;
    }

    if (!e.message || e.message.ai_generated) {
      return;
    }

    const message = e.message.text;
    if (!message) return;

    this.lastInteractionTs = Date.now();

    const writingTask = (e.message.custom as { writingTask?: string })
      ?.writingTask;
    const context = writingTask ? `Writing Task: ${writingTask}` : undefined;
    const instructions = this.getWritingAssistantPrompt(context);
    
    const fullContent = `${instructions}\n\nUser Message: ${message}`;

    const { message: channelMessage } = await this.channel.sendMessage({
      text: "",
      ai_generated: true,
    }) as { message: MessageResponse };

    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_THINKING",
      cid: channelMessage.cid,
      message_id: channelMessage.id,
    });

    const runStream = await this.genai.models.generateContentStream({
      model: "gemini-2.5-flash", 
      contents: [{ role: "user", parts: [{ text: fullContent }] }],
      config: {
        temperature: 0.7,
      }
    });

    const handler = new GeminiResponseHandler( 
      runStream,
      this.chatClient,
      this.channel,
      channelMessage,
      () => this.removeHandler(handler)
    );
    this.handlers.push(handler);
    void handler.run();
  };

  private removeHandler = (handlerToRemove: GeminiResponseHandler) => {
    this.handlers = this.handlers.filter(
      (handler) => handler !== handlerToRemove
    );
  };
}