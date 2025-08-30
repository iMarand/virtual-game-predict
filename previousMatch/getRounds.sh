#!/bin/bash

# BetPawa Virtual Games Round Downloader
# Usage: ./bash.sh <start_round> <end_round>

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <start_round> <end_round>"
    echo "Example: $0 1234 1250"
    exit 1
fi

start_round=$1
end_round=$2

# Validate that arguments are numbers
if ! [[ "$start_round" =~ ^[0-9]+$ ]] || ! [[ "$end_round" =~ ^[0-9]+$ ]]; then
    echo "Error: Both arguments must be positive integers"
    exit 1
fi

# Validate that start_round is not greater than end_round
if [ $start_round -gt $end_round ]; then
    echo "Error: Start round ($start_round) cannot be greater than end round ($end_round)"
    exit 1
fi

echo "Downloading virtual games data from round $start_round to $end_round..."

# Loop through the range of rounds
for ((round=$start_round; round<=end_round; round++)); do
    echo "Downloading round $round..."
    
    # Calculate previous round for the API call
    previous_round=$((round - 1))
    
    # Make the curl request
    curl -X GET "https://www.betpawa.rw/api/sportsbook/virtual/v2/events/list/by-round/$previous_round" \
        -H "Host: www.betpawa.rw" \
        -H "Cookie: x-pawa-token=3cffa0e58617129b-8880507c633919b4;" \
        -H "X-Pawa-Language: en" \
        -H "Sec-Ch-Ua-Platform: \"Windows\"" \
        -H "Accept-Language: en-US,en;q=0.9" \
        -H "Sec-Ch-Ua: \"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\"" \
        -H "Sec-Ch-Ua-Mobile: ?0" \
        -H "X-Pawa-Brand: betpawa-rwanda" \
        -H "Devicetype: web" \
        -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36" \
        -H "Vuejs: true" \
        -H "Accept: */*" \
        -H "Sec-Fetch-Site: same-origin" \
        -H "Sec-Fetch-Mode: cors" \
        -H "Sec-Fetch-Dest: empty" \
        -H "Referer: https://www.betpawa.rw/" \
        -H "Accept-Encoding: gzip, deflate, br" \
        --compressed \
        --output "${round}.json" \
        --silent \
        --show-error
    
    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded ${round}.json"
    else
        echo "✗ Failed to download round $round"
    fi
    
    # Add a small delay to avoid overwhelming the server
    sleep 0.5
done

echo "Download completed! Files saved as: ${start_round}.json to ${end_round}.json"