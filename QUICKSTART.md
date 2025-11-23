# Quick Start Guide - Book Explorer

## 5-Minute Setup

### 1. Install & Setup

```bash
# Install PostgreSQL (macOS)
brew install postgresql@15
brew services start postgresql@15

# Create database
createdb booksdb


# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env - set your DB_PASSWORD
# For local development, you might not need to change anything else
```

### 3. Run Your First Search

```bash
# Search for books (this will auto-create database tables)
python explorer.py search "python programming"
```

## First Commands to Try

```bash
# 1. Basic search
python explorer.py search "machine learning"

# 2. See it cache (run same search twice)
python explorer.py search "machine learning"
# Second time should be instant! ⚡

# 3. Get more results in JSON format
python explorer.py search "data science" --limit 20 --format json

# 4. Use async mode for parallel fetching
python explorer.py search "artificial intelligence" --async --parallel 5

# 5. Check your database stats
python explorer.py stats

# 6. Export your data
python explorer.py export --format json --output my_books.json
```

## Verify Installation

Run the test scripts:

```bash
# Test 1: API client (no database required)
python test_lab1.py

# Test 2: Database + caching
python test_lab2.py

# Unit tests
pytest tests/
```

## Common Issues

**"Database connection failed"**

```bash
# Make sure PostgreSQL is running
brew services list

# Start if needed
brew services start postgresql@15
```

**"Import errors"**

```bash
# Activate virtual environment
source venv/bin/activate

# Reinstall
pip install -r requirements.txt
```

**"No such database"**

```bash
# Create it
createdb booksdb
```

## What to Explore

1. **Lab Guide**: Read `LAB_GUIDE.md` for detailed explanations
2. **Source Code**: Check `src/` folder for implementation
3. **Tests**: Look at `tests/test_parse.py` for examples
4. **Full README**: See `README.md` for complete documentation

## Lab Progression

- **Lab 1** (Session A): API client with resilience → Run `test_lab1.py`
- **Lab 2** (Session B): Database + caching → Run `test_lab2.py`
- **Lab 3** (Session C): Async + CLI → Use `explorer.py`

## Success Indicators

✅ You can search for books  
✅ Second search is instant (cache hit)  
✅ Database statistics show data  
✅ Can export to JSON/CSV  
✅ Tests pass

## Next Steps

Once basics work:

1. Try different search queries
2. Experiment with cache TTL (`--cache-ttl 60`)
3. Compare sync vs async performance
4. Export data and analyze it
5. Read the Lab Guide for deeper understanding

## Getting Help

- Read error messages carefully (they're informative!)
- Check logs - they show what's happening
- Review `LAB_GUIDE.md` for concept explanations
- Make sure PostgreSQL is running
- Verify `.env` file exists and has correct credentials
