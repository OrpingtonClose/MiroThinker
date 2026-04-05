/**
 * SSE Client Helper for MiroThinker ADK Dashboard.
 *
 * Wraps the EventSource API with:
 *   - Typed event dispatching
 *   - Auto-reconnect with backoff
 *   - Clean shutdown
 */

class SSEClient {
  /**
   * @param {string} url - SSE endpoint URL
   * @param {function} onEvent - Callback receiving parsed event objects
   * @param {function} [onEnd] - Called when the stream ends
   * @param {function} [onError] - Called on connection errors
   */
  constructor(url, onEvent, onEnd = null, onError = null) {
    this.url = url;
    this.onEvent = onEvent;
    this.onEnd = onEnd;
    this.onError = onError;
    this._source = null;
    this._closed = false;
    this._reconnectDelay = 1000;
    this._maxReconnectDelay = 10000;
    this._consecutiveErrors = 0;
    this._maxConsecutiveErrors = 3;
  }

  connect() {
    if (this._closed) return;

    this._source = new EventSource(this.url);

    this._source.onmessage = (raw) => {
      try {
        const event = JSON.parse(raw.data);
        if (event.type === 'end') {
          this.close();
          if (this.onEnd) this.onEnd();
          return;
        }
        this.onEvent(event);
        // Reset reconnect delay on successful message
        this._reconnectDelay = 1000;
      } catch (e) {
        console.error('SSE parse error:', e, raw.data);
      }
    };

    this._source.onerror = (err) => {
      if (this._closed) return;
      this._consecutiveErrors++;
      console.warn('SSE connection error', this._consecutiveErrors, '/', this._maxConsecutiveErrors);
      this._source.close();
      if (this._consecutiveErrors >= this._maxConsecutiveErrors) {
        // Server is gone (agent process exited) — treat as session end
        console.info('SSE: max reconnect attempts reached, treating as session end');
        this.close();
        if (this.onEnd) this.onEnd();
        return;
      }
      if (this.onError) this.onError(err);
      setTimeout(() => this.connect(), this._reconnectDelay);
      this._reconnectDelay = Math.min(
        this._reconnectDelay * 2,
        this._maxReconnectDelay
      );
    };
  }

  close() {
    this._closed = true;
    if (this._source) {
      this._source.close();
      this._source = null;
    }
  }

  get connected() {
    return this._source && this._source.readyState === EventSource.OPEN;
  }
}

// Export for use in other modules
window.SSEClient = SSEClient;
