# Procedure acquisition and reuse

Read [README.md](README.md) first. It owns the starting question, initial
expectation, background, and links to this study's eventual findings.

This ancillary study investigates when repeated use justifies learning a
procedure into parameters rather than retaining and consulting explicit
experience. Begin with a small tool-use procedure and compare acquisition,
application to new inputs, and costs across repeated uses.

The study owns its methods, implementation, experiments, analysis, and local
publication. Revise the question when evidence warrants it and explain material
changes. Keep work within the user's stated resources and expectation; return
to the user when those boundaries need to change. The parent project's study
map supplies background, not a mandatory experimental protocol.

### Research practice

- Start with the closest papers and available implementations. Distinguish
  existing conclusions, replications, and proposed extensions; record exact
  paper and implementation versions.
- Separate exploratory observations from prospective tests of a developed
  claim. Preserve inputs, outputs, configurations, failures, and the evidence
  needed to reproduce reported comparisons.
- Judge acquisition through behavior on new inputs, not training loss alone.
  Distinguish recalling examples, following a supplied lesson, and applying a
  procedure retained in parameters. Keep information access and costs explicit,
  including lesson construction, training, retrieval, reasoning, and retries.
- Keep working documentation proportional to the investigation. Publish a
  local research note or paper with supporting evidence and reproduction
  instructions; link it from the README. Negative and inconclusive results
  are valid outcomes. External submission is a separate user decision.

### Model resources

- Open weight models with `docker model` (preferred)
- OpenAI models with `codex`
- SpaceXAI models with `agent`

### Dependency management

- Use `uv` for Python package and project management.
- Use `docker` for local models and `compose`/Dockerfile(s) for complex resources, if/when needed.
