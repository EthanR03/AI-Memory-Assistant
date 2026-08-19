"""NFL game predictor built on the 2026 Record & Fact Book.

Stage 1 (this package) turns the Fact Book PDF into a SQLite feature
store and fits a baseline ratings model against it. The store's schema
already carries the market columns Stage 2 will fill from an odds feed,
so adding market data later is a join rather than a rewrite.
"""
