import { GoogleGenAI, GenerateContentResponse, FunctionCall } from "@google/genai";
import type { Channel, Event, MessageResponse, StreamChat } from "stream-chat";

export class GeminiResponseHandler {
  private message_text = "";
  private chunk_counter = 0;
  private run_id = "";
  private is_done = false;
  private last_update_time = 0;
  
  constructor(
    private readonly contentStream: AsyncGenerator<GenerateContentResponse>, 
    private readonly chatClient: StreamChat,
    private readonly channel: Channel,
    private readonly message: MessageResponse,
    private readonly onDispose: () => void
  ) {
    this.chatClient.on("ai_indicator.stop", this.handleStopGenerating);
  }

  run = async () => {
    const { cid, id: message_id } = this.message;
    
    try {
      // Non-null assertion for cid to satisfy TypeScript's string requirement
      const channelId: string = cid!; 
        
      for await (const chunk of this.contentStream) {
        if (chunk.functionCalls && chunk.functionCalls.length > 0) {
          await this.handleFunctionCalls(
            chunk.functionCalls as FunctionCall[], // Cast to resolve type incompatibility (Error 2345)
            channelId, 
            message_id
          );
        }
        
        if (chunk.text) {
          this.message_text += chunk.text;
          const now = Date.now();
          if (now - this.last_update_time > 1000) { 
            this.chatClient.partialUpdateMessage(message_id, {
              set: { text: this.message_text },
            });
            this.last_update_time = now;
          }
          this.chunk_counter += 1;
        }
      }
      
      this.chatClient.partialUpdateMessage(message_id, {
        set: { text: this.message_text },
      });
      
      this.channel.sendEvent({
        type: "ai_indicator.clear",
        cid: this.message.cid,
        message_id: message_id,
      });

    } catch (error) {
      console.error("An error occurred during the Gemini run:", error);
      await this.handleError(error as Error);
    } finally {
      await this.dispose();
    }
  };

  private handleFunctionCalls = async (
    functionCalls: FunctionCall[],
    cid: string,
    message_id: string
  ) => {
    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_EXTERNAL_SOURCES",
      cid: cid, // Use the passed string variable `cid`
      message_id: message_id,
    });
    
    const toolOutputs = [];
    
    for (const call of functionCalls) {
      // Crucial check: Ensure call.name and call.args exist before accessing them
      if (call.name === "web_search" && call.args) {
        try {
          // Ensure query is a string to satisfy performWebSearch signature
          let queryForSearch: string;
          if (typeof call.args === "object" && call.args !== null) {
            const maybeQuery = (call.args as any).query;
            if (typeof maybeQuery === "string") {
              queryForSearch = maybeQuery;
            } else if (maybeQuery != null) {
              queryForSearch = String(maybeQuery);
            } else {
              // Fallback: serialize the whole args object if no explicit query field
              queryForSearch = JSON.stringify(call.args);
            }
          } else {
            // Fallback: coerce primitive args to string
            queryForSearch = String(call.args);
          }

          const searchResult = await this.performWebSearch(queryForSearch);
          
          toolOutputs.push({
            functionCall: call,
            output: { name: "web_search", response: JSON.parse(searchResult) },
          });

        } catch (e) {
          console.error("Error performing web search in Gemini handler", e);
          toolOutputs.push({
            functionCall: call,
            output: { name: "web_search", response: { error: "failed to call tool" } },
          });
        }
      }
    }
    
    return toolOutputs;
  };
  
  dispose = async () => {
    if (this.is_done) {
      return;
    }
    this.is_done = true;
    this.chatClient.off("ai_indicator.stop", this.handleStopGenerating);
    this.onDispose();
  };

  private handleStopGenerating = async (event: Event) => {
    if (this.is_done || event.message_id !== this.message.id) {
      return;
    }

    console.log("Stop generating for message", this.message.id);
    
    await this.channel.sendEvent({
      type: "ai_indicator.clear",
      cid: this.message.cid,
      message_id: this.message.id,
    });
    await this.dispose();
  };

  private handleError = async (error: Error) => {
    if (this.is_done) {
      return;
    }
    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_ERROR",
      cid: this.message.cid,
      message_id: this.message.id,
    });
    await this.chatClient.partialUpdateMessage(this.message.id, {
      set: {
        text: `Error: ${error.message ?? "Error generating the message"}`,
        message: error.toString(),
      },
    });
    await this.dispose();
  };

  private performWebSearch = async (query: string): Promise<string> => {
    const TAVILY_API_KEY = process.env.TAVILY_API_KEY;

    if (!TAVILY_API_KEY) {
      return JSON.stringify({
        error: "Web search is not available. API key not configured.",
      });
    }

    console.log(`Performing web search for: "${query}"`);

    try {
      const response = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${TAVILY_API_KEY}`,
        },
        body: JSON.stringify({
          query: query,
          search_depth: "advanced",
          max_results: 5,
          include_answer: true,
          include_raw_content: false,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Tavily search failed for query "${query}":`, errorText);
        return JSON.stringify({
          error: `Search failed with status: ${response.status}`,
          details: errorText,
        });
      }

      const data = await response.json();
      console.log(`Tavily search successful for query "${query}"`);

      return JSON.stringify(data);
    } catch (error) {
      console.error(
        `An exception occurred during web search for "${query}":`,
        error
      );
      return JSON.stringify({
        error: "An exception occurred during the search.",
        message: error instanceof Error ? error.message : "Unknown error",
      });
    }
  };
}