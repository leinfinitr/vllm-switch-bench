# Security Policy

## Supported code

Security fixes are applied to the default branch. Historical research branches are not
maintained.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's **Security advisories → Report a vulnerability** flow for this repository. Do not open a public issue containing credentials, remote-code-execution details, or host-identifying benchmark logs.

Include the affected commit, reproduction steps, impact, and any proposed mitigation. You should receive an acknowledgement within seven days. If GitHub private reporting is unavailable, contact the maintainer through the address in the repository's Git commit metadata and mark the message as a private security report.

## Benchmark-specific risks

GPU experiments may launch third-party containers and source checkouts, execute model code, bind local ports, and collect process/environment metadata. Review external code and images, use least-privilege rootless runtimes where possible, keep control endpoints on loopback, and inspect artifacts for secrets before publication. Never commit API keys, tokens, private model paths that disclose credentials, or unsanitized environment dumps.
