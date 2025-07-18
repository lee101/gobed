#!/usr/bin/env python3

import json

# Read the debug tokens (simple format)
with open('model/debug_tokens.json', 'r') as f:
    debug_tokens = json.load(f)

# Convert to the expected Go format
go_tokens = {}
for sentence, token_ids in debug_tokens.items():
    go_tokens[sentence] = {
        "token_ids": token_ids,
        "length": len(token_ids)
    }

# Save in the expected format
with open('model/debug_tokens_go.json', 'w') as f:
    json.dump(go_tokens, f, indent=2)

print("Converted debug tokens to Go format: model/debug_tokens_go.json")
