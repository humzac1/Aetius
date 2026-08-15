# Repository conventions

## Commit authorship: no Claude / AI co-author trailers

Standing instruction for this repository:

**Never add a `Co-Authored-By` trailer attributing Claude (or any AI assistant) to a commit.**

- Do not include `Co-Authored-By: Claude <noreply@anthropic.com>`, or any variant of it (any name containing "Claude", or any `@anthropic.com` address), in any commit message.
- This applies to every commit going forward, not only commits made interactively in a session. Any automated or agentic commit workflow that touches this repository must follow the same rule.
- Commit messages are authored purely as the repository owner's own work.

A local `commit-msg` hook at `.git/hooks/commit-msg` enforces this as a backstop: it strips any `Co-Authored-By:` line attributing Claude or Anthropic while leaving legitimate human co-author trailers for real collaborators intact. That hook is local to the clone and not version-controlled, so this file is the durable, shared statement of the rule.
