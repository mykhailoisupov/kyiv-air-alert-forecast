# Conversation logs

The AI conversation used to build this project, as plain text, in four parts.

| File | Covers |
|---|---|
| [01-research.txt](01-research.txt) | Research into existing work on alert data and forecasting |
| [02-plan-and-build.txt](02-plan-and-build.txt) | Plan, build, final model, text experiment, two questions |
| [03-review-follow-up.txt](03-review-follow-up.txt) | Trees correction, live forecast, remaining time, status page |
| [04-presentation-and-log.txt](04-presentation-and-log.txt) | Presentation README, these logs, the Kyiv scope note |

## How the logs were made

They are a mechanical export of the Claude Code session transcript stored on disk, not a retelling.

- **Verbatim:** user messages, assistant replies and subagent reports.
- **Shortened:** tool inputs, and tool output after 40 lines.
- **Left out:** images, the assistant's private reasoning, harness system messages, internal identifiers
  and the account email.

Two limits:

- The stored transcript does not hold every short progress note the assistant wrote between tool calls,
  so those notes are absent here.
- The last part ends at the moment of the final export, just before the last commit and push.
