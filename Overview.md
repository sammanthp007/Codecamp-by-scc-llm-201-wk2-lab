# Week 2 — Databases, APIs, and Resilience (Accelerated Foundations)

**What this week unlocks:**
Before touching any model, you will master the foundational data infrastructure that powers every production LLM application. This week bridges **persistent storage** with **real-time data access**, teaching you how to build resilient, scalable systems that handle both structured databases and external APIs.

You'll start by understanding **databases**—the persistent backbone of any application. You'll learn when to use relational databases (PostgreSQL) versus NoSQL solutions (MongoDB, Redis), how to design schemas that scale, and how to execute efficient queries that won't bottleneck your LLM pipeline. Understanding database fundamentals is critical because every API you build, every user interaction you log, and every cached response you store ultimately lives in a database.

Then you'll layer on **API mastery**: learning how to talk to web APIs, shape complex JSON responses, and handle network failures gracefully using caching, timeouts, retries, and concurrency caps. This isn't just about making HTTP requests—it's about building **production-grade data pipelines** that never hang, recover from failures automatically, and respect rate limits while maximizing throughput.

---

## The mental model (Accelerated Pipeline)

Modern LLM applications require **two complementary data layers**:

1. **Persistent Storage (Databases):** Where your application state, user data, conversation history, and cached embeddings live. This is your source of truth—durable, queryable, and transaction-safe.

2. **Dynamic Data (APIs):** Real-time information from external sources that enriches your LLM's context—weather data, search results, user profiles, or third-party services.

An LLM application is fundamentally a **text pipeline** with database-backed state management, where each API interaction follows a critical sequence: **Query DB → Ask API → Wait → Parse/Clean → Decide/Normalize → Try Again (Politely) → Cache to DB**

| Step        | Skills Required (W1)               | Skills Applied (W2)                                         |
| ----------- | ---------------------------------- | ----------------------------------------------------------- |
| Store       | File I/O, JSON                     | SQL queries (SELECT, INSERT, UPDATE), Indexes, Transactions |
| Query DB    | Data structures                    | Schema design, JOIN operations, Query optimization          |
| Ask/Wait    | Basic HTTP / Status Codes          | Timeouts (Never hang forever)                               |
| Parse/Clean | Strings, Regex, JSON Serialization | Normalization (Turn messy nested JSON into clean lists)     |
| Decide      | Typed Exceptions, Error Handling   | Retries (on 429/5xx), Backoff + Jitter                      |
| Integrate   | Dictionaries, Sets, Functions      | Async Fan-out with Semaphore (Concurrency cap)              |
| Reuse       | File I/O, JSON Lines               | Database Caching with TTL, Connection Pooling               |

Key Concepts You Will Master

• **Database Fundamentals:** Understanding **relational vs. NoSQL** databases, when to use each, and how to design **schemas** that support efficient queries. You'll work with **PostgreSQL** for both local development and production deployment.

• **SQL Mastery:** Writing **SELECT, INSERT, UPDATE, DELETE** queries with confidence. Understanding **JOINs**, **indexes**, and **transactions** to ensure data integrity. Learning to optimize queries to avoid the N+1 problem that plagues many applications.

• **Data Modeling:** Designing database schemas that match your application's needs. Understanding **normalization** (reducing redundancy) vs. **denormalization** (optimizing for read performance), and choosing the right trade-offs for LLM applications where read-heavy workloads dominate.

• **Connection Management:** Using **connection pooling** to reuse database connections efficiently, avoiding the overhead of establishing new connections for every query. This is critical for high-throughput applications.

• **API Protocol:** Understanding **HTTP status codes** (200, 4xx, 5xx), headers, and why **idempotency** matters when designing robust integrations with external services.

• **Resilience Toolkit:** Implementing **Timeouts** on every request (both database and API calls), and using **Retries with Exponential Backoff + Jitter** to recover politely from network hiccups, rate limits, or transient database locks.

• **High-Performance Plumbing:** Using **asyncio + Semaphore** to cap **async fan-out** and prevent stampeding servers. Understanding how to combine async API calls with synchronous database operations safely.

• **Data Integration:** Querying **databases and multiple APIs**, parsing and **normalizing** their responses into a single, de-duplicated dataset using a common schema. This teaches you how to build unified views from disparate data sources.

• **Reproducibility & Caching:** Implementing a **Database-backed Cache** with TTL (time-to-live) to ensure reruns are instant and cheap. Understanding when to cache at the database level vs. application level (Source: **net** vs **cache** vs **db** logging).

- **Secrets & config:** Never hard-code keys; read them from a `.env` file. ([python-dotenv](https://pypi.org/project/python-dotenv/))

- **Logging (lite):** One line per request makes debugging painless. ([`logging`](https://docs.python.org/3/library/logging.html))

---

## What you'll build by Sunday

A production-ready **Data Integration CLI** that demonstrates database-backed API orchestration:

- **Store and query** structured data using PostgreSQL with proper schema design and indexes,
- **Query multiple public APIs** in parallel with resilience patterns,
- Parse & normalize JSON responses into a **unified data model**,
- **Persist results to database** with automatic deduplication and timestamps,
- **Fetch in parallel** with a concurrency cap to respect rate limits,
- **Retry** politely on errors with exponential backoff,
- **Cache** API responses intelligently (database-backed with TTL),
- **Generate reports** from the database: clean **tables**, raw **JSON**, or aggregated statistics,
- Handle **data migrations** as your schema evolves.

This isn't a toy—it's the exact architecture pattern you'll use for RAG document ingestion, agent tool implementations, and production LLM applications. The database becomes your application's memory, while APIs provide fresh context.

---

## Sessions map (how we’ll get there)

**Session A — Theory + live code (2h)**
We’ll walk through:

- The HTTP “happy path” → then the “sad paths” you must handle.
- How to **shape JSON** with tiny pure functions.
- Why **timeouts + retries + backoff** are non-negotiable.
- An intro to **async fan-out** (what “await” really does).
- A 10-minute micro-quiz to lock in the concepts.

**Session B — Lab 1 (2h)**
We scaffold a small **API toolkit**: `client.py` (requests with timeout+retry), `parse.py` (normalizers with tests), `cli.py` (arguments, pretty print). No keys needed; we’ll pick keyless providers like **PokéAPI** or **REST Countries**.

**Session C — Lab 2 (2h)**
We add **async** with `httpx.AsyncClient`, **Semaphore** to cap concurrency, and a simple **on-disk cache**. You’ll see “cache hit” logs and feel the speed-up immediately.

---

## When you get stuck (self-help flow)

**“My call hangs forever.”**
Set both connect/read **timeouts** (e.g., 5s/10s). If it still hangs, print the URL you’re actually calling and try it in the browser.

**“I keep getting 429.”**
You’re too fast. Add **backoff + jitter**, reduce concurrency (`--parallel 3`), and **cache**. Read the response headers for rate-limit hints if present.

**“KeyError while parsing JSON.”**
APIs change. Use `.get()` with defaults, guard for missing fields, and write a **unit test** for your parser’s happy path and one missing-key path.

**“Async is confusing.”**
Start sync. Make it work once. Then wrap the fetch in `async def`, use a **Semaphore**, and `await` everything that does I/O. Mix sync/async carefully—don’t block the event loop with pure CPU loops.

**“My laptop is slow.”**
Lower `--parallel`, rely on **cache**, and print fewer rows (`--limit 5`). Async with a small cap is usually _faster and lighter_ than firing everything at once.

---

## Why this matters (connect to the bigger picture)

- **RAG (Weeks 5–6):** You'll store document chunks in a **vector database** (which builds on SQL fundamentals), fetch source docs via APIs, and answer questions by querying both. The **same database patterns**—schema design, indexes, transactions—keep your vector store performant. The **same API skills**—timeouts, retries, parsing—keep your doc fetching trustworthy.

- **Agents (Week 7):** Every agent tool ("search the web", "query customer database", "fetch weather") is either a **database query** or an **HTTP client**. Agents that can't handle rate limits or database locks will fail in production. Build reliable data access now; thank yourself later.

- **Production (Week 8):** Real applications need **audit trails** (database logs), **conversation history** (database storage), **usage analytics** (database queries), and **cost tracking** (database aggregations). Meanwhile, costs and rate limits are real—intelligent caching with database backing + API backoff is the difference between a $5/month app and a $500/month disaster, between a smooth demo and a blank screen.

- **Scaling:** Databases give you **ACID guarantees** (Atomicity, Consistency, Isolation, Durability) that file-based caching can't match. When you eventually deploy to the cloud, understanding connection pooling, read replicas, and query optimization will determine whether your app handles 10 users or 10,000.

---

## Light reading (optional but helpful)

**Databases:**

- PostgreSQL: [Official Tutorial](https://www.postgresql.org/docs/current/tutorial.html), [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- SQL: [SQL Tutorial (W3Schools)](https://www.w3schools.com/sql/), [Use The Index, Luke](https://use-the-index-luke.com/) (query optimization)
- Python: [`psycopg2`](https://www.psycopg.org/docs/) or [`asyncpg`](https://magicstack.github.io/asyncpg/), [Context managers for DB connections](https://docs.python.org/3/library/contextlib.html)
- Design: [Database Normalization](https://en.wikipedia.org/wiki/Database_normalization), [ACID properties](https://en.wikipedia.org/wiki/ACID)

**APIs & Networking:**

- MDN: [HTTP Overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview), [Status Codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status)
- Python: [`asyncio`](https://docs.python.org/3/library/asyncio.html), [`logging`](https://docs.python.org/3/library/logging.html), [`json`](https://docs.python.org/3/library/json.html)
- httpx: [Async requests & timeouts](https://www.python-httpx.org/async/) • [Retries](https://www.python-httpx.org/advanced/#retrying-requests)
- Backoff: [Exponential backoff](https://en.wikipedia.org/wiki/Exponential_backoff) • Why add [jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)

---

## What "done" looks like by Sunday

You can run:

```bash
# First run: fetch from APIs, store in database
python explorer.py search "everest" --limit 5 --format table --parallel 5 --cache-ttl 3600

# Second run: instant results from database cache
python explorer.py search "everest" --limit 5 --format table

# Query database directly for analytics
python explorer.py stats --show-cache-hit-rate --show-api-usage

# Export database results
python explorer.py export --format json --output results.json
```

…and you see:

- A neat table (or JSON with `--format json`) pulled from **database-backed cache**,
- "db cache hit" on the second run (instant, no network calls),
- SQL queries logged for transparency (`SELECT * FROM cache WHERE key=? AND expires_at > ?`),
- Proper **transactions** ensuring data consistency even if the program crashes mid-fetch,
- Database **indexes** making lookups fast even with thousands of cached entries,
- Friendly error messages when you break a URL or connection string,
- No crashes, even under 429/5xx or database locks,
- A clean **schema** that you can inspect with any PostgreSQL client (pgAdmin, psql, etc.).

> **Integrity note:** You should write the code yourself (no GenAI code writing). Cite any references you read.

If Week 1 built your problem-solving muscles, Week 2 gives you the network superpowers your future LLM apps will rely on.
