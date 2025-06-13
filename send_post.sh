#!/bin/bash

# URL of your running FastAPI app
URL="http://localhost:8000/how_many_upvotes"

# JSON payload
read -r -d '' PAYLOAD << EOM
{
  "title": "Amazing new post wow",
  "user_days": 5
}
EOM

# Send the POST request
curl -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"