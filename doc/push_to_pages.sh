#!/usr/bin/env bash
# Push HTML files to gh-pages automatically.

if [[ $TRAVIS_BRANCH != "master" || $TRAVIS_PULL_REQUEST != false ]]; then exit; fi

# Fill this out with the correct org/repo
ORG=BigDataRepublic
REPO=bdr-analytics-py
# This probably should match an email for one of your users.
EMAIL=info@bigdatarepublic.nl

set -e

# Script is located in and initiated from project root.
# Clone the gh-pages branch outside of the repo and cd into it.
cd ..
git clone -b gh-pages "https://$GH_TOKEN@github.com/$ORG/$REPO.git" gh-pages
cd gh-pages

# Update git configuration so I can push.
if [ "$1" != "dry" ]; then
    # Update git config.
    git config user.name "Travis Builder"
    git config user.email "$EMAIL"
fi

# Copy in the HTML.  You may want to change this with your documentation path.
cp -R ../$REPO/doc/_build/html/* ./

# Add and commit changes.
git add -A .
git commit -m "[ci skip] Autodoc commit for $COMMIT."
if [ "$1" != "dry" ]; then
    # -q is very important, otherwise you leak your GH_TOKEN
    git push -q origin gh-pages
fi

# Move back into project root.
cd ../$REPO
