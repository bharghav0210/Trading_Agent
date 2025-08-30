text
# 📈 AI Trading Agent with Real-Time Market Analysis

An intelligent trading agent that leverages real-time stock data, local AI models, and automated decision-making. Supports both US and Indian stocks with robust fallback to mock data for complete reliability and privacy.

---

## ✨ Features

- **🤖 AI-Powered Sentiment Analysis:** Uses local Ollama models (Gemma2, Llama3) for financial news analysis
- **🌍 Multi-Market Support:** US stocks (AAPL, MSFT) and Indian NSE stocks (RELIANCE.NS, TCS.NS)
- **📊 Real-Time Data:** Polygon.io API integration for live prices
- **📰 News Integration:** NewsAPI delivers up-to-date financial news
- **💼 Portfolio Management:** Virtual trading with balance and holdings tracking
- **⚡ Local Processing:** All analytics run locally; total data privacy
- **🔧 Fallback System:** Automatic switch to high-quality mock data if APIs are unavailable

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Ollama ([install from ollama.ai](https://ollama.ai))
- API keys (optional): Polygon.io & NewsAPI

### Installation

git clone <your-repo>
cd trading-agent
pip install -r requirements.txt

text

Install Ollama model:

ollama pull gemma2:2b

or
ollama pull llama3.2

text

Configure API keys (optional):

Create .env file
echo "POLYGON_API_KEY=your_polygon_key_here" > .env
echo "NEWS_API_KEY=your_newsapi_key_here" >> .env

text

Run the application:

python trading_agent.py

text

---

## 🎯 Usage

1. Enter a stock ticker (e.g., `AAPL` or `RELIANCE.NS`)
2. Click "Analyze Sentiment"—AI analyzes news and market sentiment
3. Review trading decision—BUY/SELL/HOLD with confidence score
4. Execute trades—Virtual trading with $10,000 starting capital
5. Monitor portfolio—Track performance and transaction history

---

## 🔧 Configuration

The agent automatically detects available APIs and adjusts:
- ✅ Live data when APIs are available
- ✅ High-quality mock data as fallback
- ✅ Local AI processing with Ollama
- ✅ Rule-based analysis if Ollama unavailable

---

## 📊 Supported Stocks

**US Stocks:**  
`AAPL`, `MSFT`, `GOOGL`, `TSLA`, `NVDA`, `JPM`, `V`

**Indian Stocks:**  
`RELIANCE.NS`, `TATAMOTORS.NS`, `HDFCBANK.NS`, `INFY.NS`, `TCS.NS`

---

## 🤖 AI Capabilities

- Natural language understanding of financial news
- Sentiment scoring with confidence levels
- Context-aware financial analysis
- Explainable AI decisions with reasoning

---

## 📝 License

MIT License — you are welcome to use this project for personal or educational purposes.

---

## ⚠️ Disclaimer

This is a simulation tool for educational purposes only. Not financial advice. Real trading involves risk. Always conduct your own research before investing.
