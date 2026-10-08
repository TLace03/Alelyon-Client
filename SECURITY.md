# Security policy

## Reporting a vulnerability

Please report a vulnerability **privately**, through GitHub: open this repository's **Security** tab and choose
**Report a vulnerability** (a private security advisory). Do not open a public issue, pull request or discussion for
it.

Please include what is affected (crate, file, version or commit), how to reproduce it, and what an attacker gains.
We will acknowledge the report, keep you informed, and credit you in the advisory unless you ask us not to.

## Scope

In scope: the code in this repository, for example the identity client's handling of tokens, the DPAPI vault, the
loopback listener and PKCE, and the parsing of the service's replies.

The identity service itself is not in this repository, but a vulnerability in it that you find through this client
is welcome through the same private channel.

Please do not test against other people's accounts, try to degrade the service, or run automated scans against
Alelyon's production services.

## Supported versions

Only the latest export on the default branch is supported.
