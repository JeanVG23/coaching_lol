#!/bin/bash
echo "Starting scrape loop"

for i in {1..5}; do
    echo "Iteration $i"
    poetry run python src/collection/build_referential.py --rank grandmaster --players 300 --games 5 --skip-known --start-page 0
    poetry run python src/collection/build_referential.py --rank master --players 500 --games 5 --skip-known --start-page 0
    echo "Iteration $i complete, taking a short break"
    sleep 10
done
echo "Scraping complete"
