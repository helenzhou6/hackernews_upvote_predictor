#!/bin/bash

# URL of your running FastAPI app
URL="http://localhost:8000/how_many_upvotes"

# JSON payload
read -r -d '' PAYLOAD << EOM
{
  "title": "Production is great",
  "user_created": 1098367025,
  "time": 1217418551
}
EOM

# Send the POST request
curl -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"