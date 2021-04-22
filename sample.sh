#! /bin/bash

# See https://sqlite.org/lang_datefunc.html#modifiers
PERIOD="-1 months"

if [ -z "$1" -o -z "$2" ]; then
  echo "Usage: sample.sh <places.sqlite> <directory>"
fi

mkdir -p "$2"

SQL="SELECT DISTINCT title,url FROM moz_historyvisits JOIN moz_places ON moz_places.id=moz_historyvisits.place_id WHERE visit_date > strftime('%s', 'now', '${PERIOD}') * 1000;"
IFS=$'\n'

while read -r title
do
  read -r url
  if [ ! -z "$title" ]; then
    filename=$(echo $title | sed -e "s/[^[:alnum:] ]/-/g")
    echo "Downloading $url ..."
    curl -s -o "${2}/${filename}.html" "$url"
  fi
done < <(echo "$SQL" | sqlite3 -line $1 | sed -E \
  -e '/^$/d' \
  -e "s/title = (.*)/\\1/" \
  -e "s/  url = //")

echo "Classifying..."
time python tab_clusters.py "$2"
