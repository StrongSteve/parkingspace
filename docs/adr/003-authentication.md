# ADR 003: Authentication

## Status
Accepted

## Context
We need to protect the admin interface while keeping the viewer public. Options considered:
1. Firebase Authentication
2. OAuth (Google, GitHub)
3. Static password in config
4. Generated password at startup

## Decision
Generate a random 64-character password at server startup, displayed in logs. Admin authenticates once and receives a session token.

## Rationale
- **No External Dependencies**: No Firebase, OAuth providers, or databases
- **Secure**: 64-char random password is effectively unguessable
- **Simple**: No user management, email verification, etc.
- **Accessible**: Only someone with server log access can get the password
- **Self-Contained**: Works with single Docker container

## Flow

```
1. Server starts
2. Generate random 64-char password
3. Log password to stdout (visible in render.io logs)

4. User opens /admin
5. Prompted for password
6. POST /api/auth with password
7. If correct: return session token, store in browser localStorage
8. If wrong: return 401

9. All admin API calls include session token in header
10. Server validates token matches stored session
```

## Session Management
- Single session at a time (new login invalidates old)
- Session token is 32-char random string
- Token stored in browser localStorage
- No expiration (valid until server restart or new login)

## Consequences

### Positive
- Zero configuration needed
- No secrets to manage or leak
- Password rotates on every restart (security feature)
- Works offline (no external auth providers)

### Negative
- Must check logs to get password
- Password changes on restart (minor inconvenience)
- No multi-user support
- No password recovery

## Security Considerations
- Password only shown in logs (secure in render.io)
- HTTPS required in production (render.io provides this)
- Session token sent via header, not URL
- Public endpoints (`/`, `/api/status`) require no auth
