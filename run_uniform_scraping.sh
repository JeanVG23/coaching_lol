#!/bin/bash
echo "Starting uniform scrape loop"

for i in {1..5}; do
    echo "Uniform Iteration $i"
    poetry run python src/collection/build_referential.py --rank challenger,grandmaster,master,diamond --players 150 --games 5 --skip-known --start-page 0
    echo "Uniform Iteration $i complete, taking a short break"
    sleep 10
done
echo "Uniform scraping complete"
