# Contributing to Kernel Olympics

Thanks for your interest! This project was built for the AMD Developer Hackathon ACT II.

## Project Structure

```
src/
├── router.py              # Main orchestration (4-LLM pipeline)
├── main.py                # CLI entry point
├── verification/          # Verifier: compile, run, diff
├── pattern_memory/        # Trigram caching (60,000× speedup)
├── risk_classifier/       # RED/YELLOW/GREEN kernel classifier
├── scanner/               # CUDA source analysis
├── report_generator/      # Pipeline output reports
├── debug_session/         # Debug mode tooling
└── prompt_evolution/      # Prompt optimization
```

## How to Contribute

1. Fork the repo
2. Create a branch (`git checkout -b feature/your-feature`)
3. Run tests: `make test`
4. Commit (`git commit -m "add: your feature"`)
5. Push and open a PR

## Testing

```bash
make test              # full suite (665 tests)
make test-verbose      # with progress
```

## Code Style

- Python 3.11+
- 120 char line length
- Type hints required for public functions
- Docstrings for all modules and classes

## License

MIT
