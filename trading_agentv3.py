import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from datetime import datetime, timedelta
import random
import requests
import ollama
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class FinancialDataProcessor:
    def __init__(self):
        self.categories_mapping = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC'],
            'Finance': ['JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'PYPL'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABT', 'TMO'],
            'Consumer': ['AMZN', 'WMT', 'TGT', 'HD', 'NKE', 'MCD'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'Indian': ['RELIANCE.NS', 'TATAMOTORS.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ICICIBANK.NS', 'SBIN.NS']
        }
        
    def get_stock_category(self, ticker):
        # Remove exchange suffix for categorization
        base_ticker = ticker.replace('.NS', '') if '.NS' in ticker else ticker
        for category, tickers in self.categories_mapping.items():
            if base_ticker in tickers:
                return category
        return "Other"
    
    def process_market_data(self, ticker, price, sentiment):
        category = self.get_stock_category(ticker)
        volatility = random.uniform(0.1, 0.3)
        
        return {
            'ticker': ticker,
            'price': price,
            'category': category,
            'volatility': volatility,
            'sentiment': sentiment,
            'timestamp': datetime.now()
        }

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "gemma2:2b"
        self.ollama_available = self._check_ollama()
        self.labels = ['Negative', 'Neutral', 'Positive']
        
    def _check_ollama(self):
        """Check if Ollama is available and the model exists"""
        try:
            response = ollama.list()
            models = [model['name'] for model in response['models']]
            
            if self.model_name in models:
                print(f"✓ Ollama model '{self.model_name}' is available")
                return True
            else:
                print(f"✗ Model '{self.model_name}' not found. Available models: {models}")
                print("Please run: ollama pull gemma2:2b")
                return False
                
        except Exception as e:
            print(f"✗ Ollama not available: {e}")
            print("Please install Ollama from https://ollama.ai/")
            return False
    
    def analyze_sentiment(self, text):
        if not text or not isinstance(text, str):
            return {'sentiment': 'Neutral', 'confidence': 0.5}
        
        if not self.ollama_available:
            return self._mock_sentiment_analysis(text)
        
        try:
            prompt = f"""
            Analyze the sentiment of this financial news text and respond ONLY with JSON format:
            {{
                "sentiment": "Positive", "Neutral", or "Negative",
                "confidence": 0.0 to 1.0,
                "reason": "brief explanation"
            }}
            
            Text: "{text}"
            """
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1}
            )
            
            result_text = response['message']['content'].strip()
            
            try:
                result = json.loads(result_text)
                sentiment = result.get('sentiment', 'Neutral')
                confidence = float(result.get('confidence', 0.5))
                
                if sentiment not in self.labels:
                    sentiment = 'Neutral'
                
                return {
                    'sentiment': sentiment,
                    'confidence': max(0.1, min(confidence, 1.0)),
                    'reason': result.get('reason', '')
                }
                
            except json.JSONDecodeError:
                return self._analyze_sentiment_from_text(result_text)
                
        except Exception as e:
            print(f"Ollama sentiment analysis error: {e}")
            return self._mock_sentiment_analysis(text)
    
    def _analyze_sentiment_from_text(self, text):
        text_lower = text.lower()
        
        positive_words = ['positive', 'bullish', 'buy', 'recommend', 'strong', 'growth', 'profit', 'gain', 'up']
        negative_words = ['negative', 'bearish', 'sell', 'avoid', 'weak', 'decline', 'loss', 'down', 'drop']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {'sentiment': 'Positive', 'confidence': 0.7, 'reason': 'Positive keywords detected'}
        elif negative_count > positive_count:
            return {'sentiment': 'Negative', 'confidence': 0.7, 'reason': 'Negative keywords detected'}
        else:
            return {'sentiment': 'Neutral', 'confidence': 0.5, 'reason': 'Neutral or mixed signals'}
    
    def _mock_sentiment_analysis(self, text):
        words = text.lower().split()
        positive_words = ['rise', 'up', 'strong', 'good', 'great', 'positive', 'buy', 'upgrade', 'profit', 'growth']
        negative_words = ['fall', 'down', 'weak', 'bad', 'poor', 'negative', 'sell', 'downgrade', 'loss', 'decline']
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            return {'sentiment': 'Positive', 'confidence': min(0.7 + positive_count * 0.05, 0.95), 'reason': 'Positive keywords'}
        elif negative_count > positive_count:
            return {'sentiment': 'Negative', 'confidence': min(0.7 + negative_count * 0.05, 0.95), 'reason': 'Negative keywords'}
        else:
            return {'sentiment': 'Neutral', 'confidence': 0.5, 'reason': 'Neutral content'}

class StockDataFetcher:
    def __init__(self):
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.last_call_time = 0
        self.min_call_interval = 1  # Polygon free tier allows 5 requests per minute
        self.use_mock_data = not self.polygon_key
        
        if self.use_mock_data:
            print("Using mock stock data (no Polygon API key found)")
        else:
            print("Polygon API key found - attempting real data")
    
    def get_stock_price(self, ticker):
        if self.use_mock_data:
            return self._get_mock_price(ticker)
        
        current_time = time.time()
        if current_time - self.last_call_time < self.min_call_interval:
            time.sleep(self.min_call_interval)  # Respect rate limits
        
        self.last_call_time = current_time
        
        try:
            # For Indian stocks, we need to use different endpoints
            if '.NS' in ticker:
                # For NSE stocks, use the snapshot endpoint
                url = f"https://api.polygon.io/v2/snapshot/locale/global/markets/stocks/tickers/{ticker}?apiKey={self.polygon_key}"
            else:
                # For US stocks, use the previous close endpoint
                url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={self.polygon_key}"
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if response.status_code == 200:
                if '.NS' in ticker and 'ticker' in data:
                    # Parse NSE stock response
                    price = data['ticker']['day']['c']  # Current price
                elif 'results' in data and len(data['results']) > 0:
                    # Parse US stock response
                    price = data['results'][0]['c']  # Close price
                else:
                    raise ValueError("Invalid response format")
                
                print(f"✓ Live price for {ticker}: ${price:.2f}")
                return price
            else:
                print(f"Polygon API error for {ticker}: {data.get('error', 'Unknown error')}")
                return self._get_mock_price(ticker)
                
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching {ticker} price: {e}")
            return self._get_mock_price(ticker)
        except Exception as e:
            print(f"Error fetching {ticker} price: {e}")
            return self._get_mock_price(ticker)
    
    def _get_mock_price(self, ticker):
        mock_prices = {
            'AAPL': random.uniform(170, 190),
            'MSFT': random.uniform(320, 350),
            'GOOGL': random.uniform(130, 150),
            'TSLA': random.uniform(180, 220),
            'AMZN': random.uniform(140, 160),
            'NVDA': random.uniform(420, 480),
            'JPM': random.uniform(170, 190),
            'V': random.uniform(250, 270),
            'RELIANCE.NS': random.uniform(2500, 2800),
            'TATAMOTORS.NS': random.uniform(800, 950),
            'HDFCBANK.NS': random.uniform(1600, 1800),
            'INFY.NS': random.uniform(1500, 1700),
            'TCS.NS': random.uniform(3500, 3800),
            'ICICIBANK.NS': random.uniform(900, 1100),
            'SBIN.NS': random.uniform(550, 650),
        }
        price = mock_prices.get(ticker, random.uniform(100, 200))
        print(f"Mock price for {ticker}: ${price:.2f}")
        return price

class NewsFetcher:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.use_mock_data = not self.api_key
        
        if self.use_mock_data:
            print("Using mock news data (no News API key found)")
        else:
            print("News API key found - attempting real news")
    
    def fetch_news(self, ticker, days=1):
        if self.use_mock_data:
            return self._get_mock_news(ticker)
        
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            from_str = from_date.strftime('%Y-%m-%d')
            to_str = to_date.strftime('%Y-%m-%d')
            
            # For Indian stocks, search without the .NS suffix
            search_ticker = ticker.replace('.NS', '') if '.NS' in ticker else ticker
            
            url = f"https://newsapi.org/v2/everything?q={search_ticker}&from={from_str}&to={to_str}&sortBy=publishedAt&apiKey={self.api_key}"
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('status') == 'ok' and data.get('articles'):
                articles = data['articles'][:3]
                news_items = []
                for article in articles:
                    title = article.get('title', 'No title')
                    description = article.get('description', 'No description')
                    news_items.append(f"{title}: {description}")
                print(f"✓ Live news fetched for {ticker}")
                return news_items
            else:
                print(f"News API error for {ticker}")
                return self._get_mock_news(ticker)
                
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching news for {ticker}: {e}")
            return self._get_mock_news(ticker)
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return self._get_mock_news(ticker)
    
    def _get_mock_news(self, ticker):
        # Remove .NS suffix for news lookup
        base_ticker = ticker.replace('.NS', '') if '.NS' in ticker else ticker
        
        mock_news_db = {
            'AAPL': [
                "Apple announces new iPhone with revolutionary AI features",
                "Apple Q3 earnings beat expectations by 15%, stock surges",
                "Apple partners with OpenAI for iOS 18 integration"
            ],
            'MSFT': [
                "Microsoft Azure cloud revenue grows 25% year-over-year",
                "Microsoft announces major Windows 12 update with AI copilot",
                "Satya Nadella: AI will transform every industry"
            ],
            'GOOGL': [
                "Google Gemini AI outperforms competitors in latest benchmarks",
                "Google Cloud expands AI services to new markets",
                "Sundar Pichai discusses AI future at Google IO"
            ],
            'TSLA': [
                "Tesla unveils new Robotaxi prototype with full autonomy",
                "Elon Musk: Tesla will achieve Level 5 autonomy this year",
                "Tesla energy storage business grows 40% quarterly"
            ],
            'RELIANCE': [
                "Reliance Jio announces 5G rollout across India completed",
                "Mukesh Ambani: Reliance to invest $10B in green energy",
                "Reliance Retail expands to 1000 new locations"
            ],
            'TATAMOTORS': [
                "Tata Motors electric vehicle sales double year-over-year",
                "Tata Steel announces new sustainable manufacturing initiative",
                "Tata Group to acquire major e-commerce platform"
            ],
            'HDFCBANK': [
                "HDFC Bank reports strong quarterly results with 18% growth",
                "HDFC launches new digital banking platform with AI features",
                "HDFC expands rural banking services across India"
            ]
        }
        
        news = mock_news_db.get(base_ticker, [
            f"{base_ticker} shows strong performance in recent market trading",
            f"Analysts upgrade {base_ticker} rating based on strong fundamentals",
            f"Market sentiment for {base_ticker} remains positive amid volatility"
        ])
        print(f"Mock news for {ticker}: {len(news)} articles")
        return news

class TradingAgent:
    def __init__(self, analyzer, news_fetcher, data_processor):
        self.analyzer = analyzer
        self.news_fetcher = news_fetcher
        self.data_processor = data_processor
        self.stock_fetcher = StockDataFetcher()
        self.portfolio = {}
        self.balance = 10000.0
        self.transaction_history = []
        self.market_data = []
    
    def analyze_news_for_ticker(self, ticker):
        news_articles = self.news_fetcher.fetch_news(ticker)
        sentiments = []
        
        for news in news_articles:
            sentiment = self.analyzer.analyze_sentiment(news)
            sentiments.append(sentiment)
        
        return news_articles, sentiments
    
    def make_trading_decision(self, ticker, sentiments):
        if not sentiments:
            return "HOLD", 0.5
        
        # Use weighted average based on confidence
        positive_score = sum(s['confidence'] for s in sentiments if s['sentiment'] == 'Positive') / len(sentiments)
        negative_score = sum(s['confidence'] for s in sentiments if s['sentiment'] == 'Negative') / len(sentiments)
        
        if positive_score > 0.65:
            decision = "BUY"
            confidence = positive_score
        elif negative_score > 0.65:
            decision = "SELL"
            confidence = negative_score
        else:
            decision = "HOLD"
            confidence = max(positive_score, negative_score, 0.5)
            
        return decision, confidence
    
    def execute_trade(self, ticker, decision, confidence):
        current_price = self.stock_fetcher.get_stock_price(ticker)
        
        if decision == "BUY" and self.balance >= 500:
            amount = min(self.balance * 0.1, 1000)
            shares = amount / current_price
            self.portfolio[ticker] = self.portfolio.get(ticker, 0) + shares
            self.balance -= amount
            
            self.transaction_history.append({
                'date': datetime.now(),
                'ticker': ticker,
                'action': 'BUY',
                'shares': shares,
                'price': current_price,
                'amount': amount
            })
            
            market_data = self.data_processor.process_market_data(ticker, current_price, 'Positive')
            self.market_data.append(market_data)
            
            return f"Bought {shares:.2f} shares of {ticker} at ${current_price:.2f}"
            
        elif decision == "SELL" and ticker in self.portfolio and self.portfolio[ticker] > 0:
            shares_to_sell = min(self.portfolio[ticker] * min(confidence, 0.5), self.portfolio[ticker])
            amount = shares_to_sell * current_price
            
            if shares_to_sell > 0:
                self.portfolio[ticker] -= shares_to_sell
                if self.portfolio[ticker] < 0.001:
                    self.portfolio.pop(ticker)
                
                self.balance += amount
                
                self.transaction_history.append({
                    'date': datetime.now(),
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'amount': amount
                })
                
                market_data = self.data_processor.process_market_data(ticker, current_price, 'Negative')
                self.market_data.append(market_data)
                
                return f"Sold {shares_to_sell:.2f} shares of {ticker} at ${current_price:.2f}"
            
        return f"No trade executed for {ticker}"

class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Trading Agent with Polygon.io API")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        print("Initializing trading agent components...")
        self.data_processor = FinancialDataProcessor()
        self.analyzer = SentimentAnalyzer()
        self.news_fetcher = NewsFetcher()
        self.agent = TradingAgent(self.analyzer, self.news_fetcher, self.data_processor)
        
        self.setup_styles()
        self.create_gui()
        
        self.update_interval = 30000
        self.schedule_updates()
        print("Trading agent initialized successfully!")
    
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        self.style.configure('Normal.TLabel', font=('Arial', 10), background='#f0f0f0')
        self.style.configure('Positive.TLabel', font=('Arial', 10), background='#f0f0f0', foreground='green')
        self.style.configure('Negative.TLabel', font=('Arial', 10), background='#f0f0f0', foreground='red')
        self.style.configure('Buy.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0', foreground='green')
        self.style.configure('Sell.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0', foreground='red')
        self.style.configure('Hold.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0', foreground='blue')
        self.style.configure('TFrame', background='#f0f0f0')
    
    def create_gui(self):
        main_frame = ttk.Frame(self.root, padding="10", style='TFrame')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        title_label = ttk.Label(main_frame, text="AI Trading Agent with Polygon.io API", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # API status
        api_status = "Using Live Data" if not self.agent.stock_fetcher.use_mock_data else "Using Mock Data"
        news_status = "Live News" if not self.news_fetcher.use_mock_data else "Mock News"
        ollama_status = "✓ Ollama Ready" if self.analyzer.ollama_available else "✗ Ollama Offline"
        
        status_label = ttk.Label(main_frame, text=f"Status: {api_status} | {news_status} | {ollama_status}", style='Header.TLabel')
        status_label.grid(row=0, column=1, sticky=tk.E, padx=10)
        
        ttk.Label(main_frame, text="Stock Ticker:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ticker_var = tk.StringVar(value="AAPL")
        ticker_entry = ttk.Entry(main_frame, textvariable=self.ticker_var, width=15)
        ticker_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Add hint for Indian stocks
        ttk.Label(main_frame, text="For Indian stocks, use .NS suffix (e.g., RELIANCE.NS)", 
                 style='Normal.TLabel', foreground='gray').grid(row=1, column=1, sticky=tk.E, pady=5)
        
        button_frame = ttk.Frame(main_frame, style='TFrame')
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        analyze_btn = ttk.Button(button_frame, text="Analyze Sentiment", command=self.analyze_sentiment)
        analyze_btn.pack(side=tk.LEFT, padx=5)
        
        trade_btn = ttk.Button(button_frame, text="Execute Trade", command=self.execute_trade)
        trade_btn.pack(side=tk.LEFT, padx=5)
        
        debug_btn = ttk.Button(button_frame, text="Debug Info", command=self.show_debug_info)
        debug_btn.pack(side=tk.LEFT, padx=5)
        
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.news_text = scrolledtext.ScrolledText(results_frame, width=80, height=10, wrap=tk.WORD)
        self.news_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        decision_frame = ttk.Frame(results_frame)
        decision_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(decision_frame, text="Trading Decision:", style='Header.TLabel').pack(side=tk.LEFT)
        self.decision_var = tk.StringVar(value="HOLD")
        self.decision_label = ttk.Label(decision_frame, textvariable=self.decision_var, style='Hold.TLabel')
        self.decision_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(decision_frame, text="Confidence:", style='Header.TLabel').pack(side=tk.LEFT, padx=(20, 0))
        self.confidence_var = tk.StringVar(value="0.0%")
        ttk.Label(decision_frame, textvariable=self.confidence_var, style='Normal.TLabel').pack(side=tk.LEFT, padx=5)
        
        portfolio_frame = ttk.LabelFrame(main_frame, text="Portfolio", padding="10")
        portfolio_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        portfolio_frame.columnconfigure(1, weight=1)
        
        ttk.Label(portfolio_frame, text="Balance:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        self.balance_var = tk.StringVar(value=f"${self.agent.balance:.2f}")
        ttk.Label(portfolio_frame, textvariable=self.balance_var, style='Normal.TLabel').grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(portfolio_frame, text="Holdings:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        self.holdings_var = tk.StringVar(value="None")
        ttk.Label(portfolio_frame, textvariable=self.holdings_var, style='Normal.TLabel').grid(row=1, column=1, sticky=tk.W, pady=2)
        
        history_frame = ttk.LabelFrame(main_frame, text="Transaction History", padding="10")
        history_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        
        columns = ('Date', 'Ticker', 'Action', 'Shares', 'Price', 'Amount')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100)
        
        self.history_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        self.news_text.tag_configure('positive', foreground='green')
        self.news_text.tag_configure('negative', foreground='red')
        self.news_text.tag_configure('neutral', foreground='blue')
    
    def show_debug_info(self):
        info = [
            f"Ollama Available: {'Yes' if self.analyzer.ollama_available else 'No'}",
            f"Ollama Model: {self.analyzer.model_name}",
            f"Polygon API: {'Live' if not self.agent.stock_fetcher.use_mock_data else 'Mock'}",
            f"News API: {'Live' if not self.news_fetcher.use_mock_data else 'Mock'}",
            f"Balance: ${self.agent.balance:.2f}",
            f"Holdings: {len(self.agent.portfolio)} stocks",
            f"Transactions: {len(self.agent.transaction_history)}"
        ]
        messagebox.showinfo("Debug Information", "\n".join(info))
    
    def analyze_sentiment(self):
        ticker = self.ticker_var.get().upper().strip()
        if not ticker:
            messagebox.showerror("Error", "Please enter a stock ticker")
            return
        
        self.news_text.delete(1.0, tk.END)
        self.news_text.insert(tk.END, f"Analyzing news for {ticker} using Ollama...\n")
        self.decision_var.set("ANALYZING...")
        self.confidence_var.set("0.0%")
        
        thread = threading.Thread(target=self._perform_analysis, args=(ticker,))
        thread.daemon = True
        thread.start()
    
    def _perform_analysis(self, ticker):
        try:
            price = self.agent.stock_fetcher.get_stock_price(ticker)
            news_articles, sentiments = self.agent.analyze_news_for_ticker(ticker)
            decision, confidence = self.agent.make_trading_decision(ticker, sentiments)
            
            self.root.after(0, self._update_analysis_results, ticker, news_articles, sentiments, decision, confidence, price)
        except Exception as e:
            self.root.after(0, self._show_error, f"Analysis error: {str(e)}")
    
    def _update_analysis_results(self, ticker, news_articles, sentiments, decision, confidence, price):
        self.news_text.delete(1.0, tk.END)
        
        data_source = "Live Data" if not self.agent.stock_fetcher.use_mock_data else "Mock Data"
        news_source = "Live News" if not self.news_fetcher.use_mock_data else "Mock News"
        ollama_source = "Ollama AI" if self.analyzer.ollama_available else "Rule-Based"
        
        self.news_text.insert(tk.END, f"Data Source: {data_source} | News: {news_source} | Analysis: {ollama_source}\n")
        self.news_text.insert(tk.END, f"News for {ticker} (Current Price: ${price:.2f}):\n\n")
        
        for i, (news, sentiment) in enumerate(zip(news_articles, sentiments)):
            self.news_text.insert(tk.END, f"{i+1}. {news}\n")
            sentiment_text = f"   Sentiment: {sentiment['sentiment']} (Confidence: {sentiment['confidence']*100:.1f}%)\n"
            if sentiment.get('reason'):
                sentiment_text += f"   Reason: {sentiment['reason']}\n"
            sentiment_text += "\n"
            
            if sentiment['sentiment'] == 'Positive':
                self.news_text.insert(tk.END, sentiment_text, 'positive')
            elif sentiment['sentiment'] == 'Negative':
                self.news_text.insert(tk.END, sentiment_text, 'negative')
            else:
                self.news_text.insert(tk.END, sentiment_text, 'neutral')
        
        self.decision_var.set(decision)
        self.confidence_var.set(f"{confidence*100:.1f}%")
        
        if decision == "BUY":
            self.decision_label.configure(style='Buy.TLabel')
        elif decision == "SELL":
            self.decision_label.configure(style='Sell.TLabel')
        else:
            self.decision_label.configure(style='Hold.TLabel')
    
    def _show_error(self, message):
        messagebox.showerror("Error", message)
    
    def execute_trade(self):
        ticker = self.ticker_var.get().upper().strip()
        if not ticker:
            messagebox.showerror("Error", "Please enter a stock ticker")
            return
        
        decision = self.decision_var.get()
        if decision == "ANALYZING...":
            messagebox.showerror("Error", "Please analyze sentiment first")
            return
        
        confidence_str = self.confidence_var.get().replace('%', '')
        try:
            confidence = float(confidence_str) / 100
        except:
            confidence = 0.5
        
        try:
            result = self.agent.execute_trade(ticker, decision, confidence)
            self.update_portfolio_display()
            messagebox.showinfo("Trade Executed", result)
        except Exception as e:
            messagebox.showerror("Trade Error", f"Error executing trade: {str(e)}")
    
    def update_portfolio_display(self):
        self.balance_var.set(f"${self.agent.balance:.2f}")
        
        if self.agent.portfolio:
            holdings_text = ", ".join([f"{k}: {v:.2f} shares" for k, v in self.agent.portfolio.items()])
        else:
            holdings_text = "None"
        self.holdings_var.set(holdings_text)
        
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        for transaction in self.agent.transaction_history[-10:]:
            self.history_tree.insert('', 'end', values=(
                transaction['date'].strftime('%Y-%m-%d %H:%M'),
                transaction['ticker'],
                transaction['action'],
                f"{transaction['shares']:.2f}",
                f"${transaction['price']:.2f}",
                f"${transaction['amount']:.2f}"
            ))
    
    def schedule_updates(self):
        self.periodic_update()
        self.root.after(self.update_interval, self.schedule_updates)
    
    def periodic_update(self):
        ticker = self.ticker_var.get().upper().strip()
        if ticker:
            thread = threading.Thread(target=self._perform_analysis, args=(ticker,))
            thread.daemon = True
            thread.start()

def main():
    root = tk.Tk()
    app = TradingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()